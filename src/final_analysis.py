"""
Final results tables, figures, and failure analysis for all retrieval methods.

Inputs (all produced by the other scripts)
    results/<split>/runs/dense_<model>_<set>_top50.json        from dense_retrieval.py
    results/<split>/runs/rerank_<model>_<set>_depth<D>.json    from rerank.py
    results/<split>/dense_results_*.json, rerank_results_*.json

TF-IDF and BM25 are re-run here with the SAME top-50 output format and the same
self-exclusion rule, so every method is scored by exactly the same code on exactly
the same queries (the baseline scripts only report Recall@10 and MRR@10).

Outputs (results/<split>/, where <split> is dev or test)
    results_table_<split>.md / .csv        final comparison of all methods
    breakdown_<split>.md                   MRR@10 by query length and by word overlap
    failure_analysis_<split>.md            failure categories + examples for the final system
    failure_cases_<split>.csv              every top-10 miss of the final system
    qualitative_examples_<split>.md        dense-vs-keyword wins, reranker fixes and regressions
    figures/final_comparison_<split>.png, figures/breakdown_<split>.png

Usage
    python final_analysis.py                     # dev split, analysed on the held-out val queries
    python final_analysis.py --split test        # FINAL numbers, run once at the end
    python final_analysis.py --skip-tfidf        # skip the slower TF-IDF re-run
"""

import argparse
import csv
import json
import random
import re
import time
from pathlib import Path

import numpy as np

from data_loader import load_dataset
from data_split import split_dataset
from metrics import KS, compute_metrics, per_query_metrics


PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_DIR / "results"


def results_dir(split):
    """Outputs of each split live in results/dev/ or results/test/."""
    return RESULTS_ROOT / split


def runs_dir(split):
    return results_dir(split) / "runs"


def figures_dir(split):
    return results_dir(split) / "figures"

DENSE_NAMES = {"minilm": "MiniLM", "mpnet": "MPNet"}
TOP_K = 50

LENGTH_BINS = [("short (<=7 words)", 0, 7), ("medium (8-11)", 8, 11), ("long (12+)", 12, 10**6)]
OVERLAP_BINS = [("low overlap (<0.30)", 0.0, 0.30),
                ("medium (0.30-0.60)", 0.30, 0.60),
                ("high (>=0.60)", 0.60, 1.01)]


# ----------------------------------------------------------------------------
# 1. Method runs
# ----------------------------------------------------------------------------

def load_or_compute(path, compute):
    """Runs are cached to disk so slow baselines are only computed once."""
    if path.exists():
        data = json.loads(path.read_text())
        return data["run"], data["ms_per_query"]
    run, ms_per_query = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"run": run, "ms_per_query": ms_per_query}))
    return run, ms_per_query


def run_bm25(corpus_ids, corpus_texts, query_ids, query_texts):
    """BM25 (same settings as baseline_bm25.py) but with the query itself excluded."""

    import bm25s

    tokens = bm25s.tokenize(corpus_texts, show_progress=False)
    bm25 = bm25s.BM25(method="lucene")
    bm25.index(tokens, show_progress=False)

    t0 = time.perf_counter()
    results, _ = bm25.retrieve(
        bm25s.tokenize(query_texts, show_progress=False), k=TOP_K + 1, show_progress=False
    )
    seconds = time.perf_counter() - t0

    run = {}
    for qid, row in zip(query_ids, results):
        ids = [corpus_ids[i] for i in row if corpus_ids[i] != qid]
        run[qid] = ids[:TOP_K]

    return run, seconds / len(query_ids) * 1000


def run_tfidf(corpus_ids, corpus_texts, query_ids, query_texts, batch_size=50):
    """TF-IDF + cosine similarity (same settings as baseline_tfidf.py), top 50."""

    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer()
    corpus_vectors = vectorizer.fit_transform(corpus_texts).T.tocsr()
    id_to_index = {doc_id: i for i, doc_id in enumerate(corpus_ids)}

    query_vectors = vectorizer.transform(query_texts)
    run = {}

    t0 = time.perf_counter()
    for start in range(0, len(query_ids), batch_size):

        scores = (query_vectors[start:start + batch_size] @ corpus_vectors).toarray()

        for row, qid in enumerate(query_ids[start:start + batch_size]):
            if qid in id_to_index:
                scores[row, id_to_index[qid]] = -1          # never retrieve itself
            top = np.argpartition(-scores[row], TOP_K)[:TOP_K]
            top = top[np.argsort(-scores[row][top], kind="stable")]
            run[qid] = [corpus_ids[i] for i in top]

    seconds = time.perf_counter() - t0
    return run, seconds / len(query_ids) * 1000


def choose_first_stage():
    """Same rule as rerank.py: higher DEV Recall@20 (the test set is never used to choose)."""

    best_key, best = None, -1.0
    for key in DENSE_NAMES:
        path = results_dir("dev") / f"dense_results_{key}_dev.json"
        if path.exists():
            recall = json.loads(path.read_text())["eval"]["dev"]["recall@20"]
            if recall > best:
                best_key, best = key, recall
    if best_key is None:
        raise SystemExit("No dense results found. Run dense_retrieval.py first.")
    return best_key


def pick_final_system(first_stage, depths):
    """
    The final system is the first stage alone or one reranking depth, whichever has the
    highest MRR@10 on the DEV queries (4000 queries), not on val or test.
    """

    path = results_dir("dev") / f"rerank_results_{first_stage}_dev.json"
    if not path.exists():
        return DENSE_NAMES[first_stage]

    res = json.loads(path.read_text())
    best_name = DENSE_NAMES[first_stage]
    best = res["first_stage"]["dev"]["mrr@10"]

    for d in depths:
        mrr = res["rerank"].get(f"depth{d}", {}).get("dev", {}).get("mrr@10", -1)
        if mrr > best:
            best_name, best = f"{DENSE_NAMES[first_stage]}+CE@{d}", mrr

    print(f"Final system (chosen on dev MRR@10): {best_name}")
    return best_name


# ----------------------------------------------------------------------------
# 2. Results table and figure
# ----------------------------------------------------------------------------

def results_table(runs, qrels, latencies, set_name, split):

    names = list(runs)
    metrics = {name: compute_metrics(runs[name], qrels) for name in names}
    cols = [f"recall@{k}" for k in KS] + ["mrr@10", "ndcg@10"]
    pretty = [f"R@{k}" for k in KS] + ["MRR@10", "nDCG@10"]
    n = next(iter(metrics.values()))["num_queries"]

    lines = [f"# Final results ({set_name} queries, n = {n})", "",
             "| Method | " + " | ".join(pretty) + " | Latency (ms/query) |",
             "|---|" + "---:|" * (len(cols) + 1)]

    best = {c: max(metrics[m][c] for m in names) for c in cols}

    for name in names:
        cells = []
        for c in cols:
            text = f"{metrics[name][c]:.4f}"
            cells.append(f"**{text}**" if metrics[name][c] == best[c] else text)
        lat = latencies.get(name)
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {lat:.1f} |" if lat is not None
                     else f"| {name} | " + " | ".join(cells) + " | - |")

    lines += ["",
              "Bold = best in column. TF-IDF and BM25 latency is the average over a batch of queries; "
              "dense latency is single-query (encode + search); reranked latency adds the "
              "cross-encoder time for one query to the first-stage latency.",
              "All methods use the same corpus and queries, the same K values and the same "
              "metric code, and a query never retrieves itself."]

    results_dir(split).mkdir(parents=True, exist_ok=True)
    (results_dir(split) / f"results_table_{split}.md").write_text("\n".join(lines) + "\n")

    with open(results_dir(split) / f"results_table_{split}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method"] + cols + ["latency_ms"])
        for name in names:
            writer.writerow([name] + [round(metrics[name][c], 4) for c in cols]
                            + [None if latencies.get(name) is None else round(latencies[name], 1)])

    print("\n".join(lines))
    return metrics


def plot_comparison(metrics, set_name, split):

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(metrics)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    width = 0.8 / len(names)
    shown = ["recall@10", "mrr@10", "ndcg@10"]
    for i, name in enumerate(names):
        axes[0].bar(np.arange(len(shown)) + i * width, [metrics[name][m] for m in shown],
                    width, label=name)
    axes[0].set_xticks(np.arange(len(shown)) + width * (len(names) - 1) / 2)
    axes[0].set_xticklabels(["Recall@10", "MRR@10", "nDCG@10"])
    axes[0].set_ylim(0.5, 1.0)
    axes[0].set_title(f"Retrieval quality ({set_name} queries)")
    axes[0].legend(fontsize=8)

    for name in names:
        axes[1].plot(KS, [metrics[name][f"recall@{k}"] for k in KS], marker="o", label=name)
    axes[1].set_xscale("log")
    axes[1].set_xticks(KS)
    axes[1].set_xticklabels(KS)
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("Recall@K")
    axes[1].set_title("Recall as K grows")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    figures_dir(split).mkdir(parents=True, exist_ok=True)
    path = figures_dir(split) / f"final_comparison_{split}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved figure:", path)


# ----------------------------------------------------------------------------
# 3. Breakdown by query length and word overlap
# ----------------------------------------------------------------------------

def words(text):
    return re.findall(r"\w+", text.lower())


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else 0.0


def query_features(query_ids, queries, qrels, corpus):
    """Length in words, and mean Jaccard word overlap with the query's relevant questions."""

    features = {}
    for qid in query_ids:
        q_words = words(queries[qid])
        overlaps = [jaccard(q_words, words(corpus[d]["text"]))
                    for d in qrels[qid] if d in corpus]
        features[qid] = {"length": len(q_words),
                         "overlap": float(np.mean(overlaps)) if overlaps else 0.0}
    return features


def breakdown(per_query, features, set_name, split):

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = list(per_query)
    lines = [f"# MRR@10 by query characteristics ({set_name} queries)", ""]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))

    for ax, (title, key, bins) in zip(axes, [("Query length", "length", LENGTH_BINS),
                                             ("Word overlap with relevant question", "overlap", OVERLAP_BINS)]):

        lines += [f"## By {title.lower()}", "",
                  "| Group | Queries | " + " | ".join(methods) + " |",
                  "|---|---:|" + "---:|" * len(methods)]

        width = 0.8 / len(methods)
        for b, (label, low, high) in enumerate(bins):
            if key == "length":      # whole-word counts, both ends inclusive
                qids = [q for q, f in features.items() if low <= f["length"] <= high]
            else:                    # overlap in [low, high)
                qids = [q for q, f in features.items() if low <= f["overlap"] < high]
            row = [np.mean([per_query[m][q]["mrr@10"] for q in qids]) if qids else float("nan")
                   for m in methods]
            lines.append(f"| {label} | {len(qids)} | " + " | ".join(f"{v:.4f}" for v in row) + " |")
            for i, value in enumerate(row):
                ax.bar(b + i * width, value, width,
                       label=methods[i] if b == 0 else None, color=f"C{i}")

        ax.set_xticks(np.arange(len(bins)) + width * (len(methods) - 1) / 2)
        ax.set_xticklabels([b[0] for b in bins], fontsize=8)
        ax.set_ylabel("MRR@10")
        ax.set_title(title)
        lines.append("")

    axes[0].legend(fontsize=7)
    plt.tight_layout()
    figures_dir(split).mkdir(parents=True, exist_ok=True)
    plt.savefig(figures_dir(split) / f"breakdown_{split}.png", dpi=300, bbox_inches="tight")
    plt.close()

    lines.append("Word overlap = mean Jaccard similarity between the query's words and each "
                 "of its relevant questions (lower-cased words).")
    (results_dir(split) / f"breakdown_{split}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# ----------------------------------------------------------------------------
# 4. Failure analysis
# ----------------------------------------------------------------------------

def categorize(qid, run, qrels, queries, corpus, features):
    """Heuristic reason a duplicate was missed in the top 10 (checked in this order)."""

    q_words = words(queries[qid])
    top1 = run[qid][0]
    top1_overlap = jaccard(q_words, words(corpus[top1]["text"]))

    if top1_overlap >= 0.5:
        return "Look-alike distractor"       # a very similar question outranks the duplicate
    if features[qid]["overlap"] < 0.25:
        return "Paraphrase (low word overlap)"

    rel_lengths = [len(words(corpus[d]["text"])) for d in qrels[qid] if d in corpus]
    if rel_lengths:
        ratio = len(q_words) / max(1, np.mean(rel_lengths))
        if ratio >= 2 or ratio <= 0.5:
            return "Length mismatch"

    return "Other"


def show_example(qid, queries, corpus, qrels, run, system, top_n=3):

    relevant = set(qrels[qid])
    ranked = run[qid]
    lines = [f"**Query:** {queries[qid]}", ""]

    for doc_id in sorted(relevant):
        if doc_id in corpus:
            rank = ranked.index(doc_id) + 1 if doc_id in ranked else f">{len(ranked)}"
            lines.append(f"- Relevant (rank {rank} in {system}): {corpus[doc_id]['text']}")

    lines.append(f"- Top {top_n} retrieved by {system}:")
    for r, doc_id in enumerate(ranked[:top_n], start=1):
        mark = " (relevant)" if doc_id in relevant else ""
        lines.append(f"  {r}. {corpus[doc_id]['text']}{mark}")

    return "\n".join(lines) + "\n"


def failure_analysis(system, run, qrels, queries, corpus, features, per_query,
                     set_name, split, examples_per_category):

    misses = [q for q in run if per_query[system][q]["first_rank"] is None
              or per_query[system][q]["first_rank"] > 10]

    total = len(run)
    categories = {}
    for qid in misses:
        categories.setdefault(categorize(qid, run, qrels, queries, corpus, features), []).append(qid)

    lines = [f"# Failure analysis: {system} ({set_name} queries)", "",
             f"{len(misses)} of {total} queries ({100 * len(misses) / total:.1f}%) have no relevant "
             "question in the top 10. Reasons are assigned by simple rules (first match):", "",
             "| Category | Queries | Share of misses |", "|---|---:|---:|"]

    for cat, qids in sorted(categories.items(), key=lambda x: -len(x[1])):
        lines.append(f"| {cat} | {len(qids)} | {100 * len(qids) / max(1, len(misses)):.1f}% |")

    lines += ["",
              "- *Look-alike distractor*: the top-ranked question shares at least half of its words "
              "with the query but is not a labelled duplicate. These are often questions that differ "
              "in one key detail, or true duplicates that the dataset did not label.",
              "- *Paraphrase*: the labelled duplicate shares few words with the query.",
              "- *Length mismatch*: query and duplicate differ in length by 2x or more.", ""]

    rng = random.Random(42)
    for cat, qids in sorted(categories.items(), key=lambda x: -len(x[1])):
        lines += [f"## {cat}", ""]
        for qid in rng.sample(qids, min(examples_per_category, len(qids))):
            lines += [show_example(qid, queries, corpus, qrels, run, system), ""]

    (results_dir(split) / f"failure_analysis_{split}.md").write_text("\n".join(lines) + "\n")

    with open(results_dir(split) / f"failure_cases_{split}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "query", "category", "first_relevant_rank",
                         "relevant_question", "top1_retrieved"])
        for cat, qids in categories.items():
            for qid in qids:
                rel = [corpus[d]["text"] for d in qrels[qid] if d in corpus]
                writer.writerow([qid, queries[qid], cat, per_query[system][qid]["first_rank"],
                                 " || ".join(rel), corpus[run[qid][0]]["text"]])

    print(f"\nFailure analysis: {len(misses)}/{total} top-10 misses")
    for cat, qids in categories.items():
        print(f"  {cat}: {len(qids)}")


# ----------------------------------------------------------------------------
# 5. Qualitative examples
# ----------------------------------------------------------------------------

def qualitative_examples(runs, per_query, qrels, queries, corpus, features,
                         dense_name, rerank_name, set_name, split, n):

    def rank(method, qid):
        r = per_query[method][qid]["first_rank"]
        return r if r is not None else 10**6

    qids = list(runs[dense_name])
    rng = random.Random(42)
    lines = [f"# Qualitative examples ({set_name} queries)", ""]

    def section(title, description, selected, method_a, method_b):
        lines.extend([f"## {title}", "", description, "", f"*{len(selected)} queries match.*", ""])
        for qid in rng.sample(selected, min(n, len(selected))):
            lines.append(show_example(qid, queries, corpus, qrels, runs[method_b], method_b))
            lines.append(f"*{method_a} first relevant rank: "
                         f"{per_query[method_a][qid]['first_rank'] or 'not in top 50'}; "
                         f"{method_b}: {per_query[method_b][qid]['first_rank'] or 'not in top 50'}; "
                         f"word overlap {features[qid]['overlap']:.2f}*\n")

    if "BM25" in runs:
        wins = [q for q in qids if rank("BM25", q) > 10 and rank(dense_name, q) == 1]
        section("Dense retrieval finds it, BM25 misses it",
                "The labelled duplicate is not in BM25's top 10 but is ranked first by dense retrieval.",
                wins, "BM25", dense_name)

    summary = ""
    if rerank_name in runs:
        fixed = [q for q in qids if rank(dense_name, q) > 1 and rank(rerank_name, q) == 1]
        hurt = [q for q in qids if rank(dense_name, q) == 1 and rank(rerank_name, q) > 1]
        same = len(qids) - len(fixed) - len(hurt)

        improved = sum(rank(rerank_name, q) < rank(dense_name, q) for q in qids)
        worse = sum(rank(rerank_name, q) > rank(dense_name, q) for q in qids)

        summary = (f"Reranking changed the first relevant rank for {improved + worse} of {len(qids)} "
                   f"queries: {improved} improved, {worse} got worse.")

        section("The cross-encoder fixes the ranking",
                f"The duplicate was not ranked first by {dense_name} but is ranked first after reranking.",
                fixed, dense_name, rerank_name)
        section("The cross-encoder breaks a correct ranking",
                f"The duplicate was ranked first by {dense_name} but falls lower after reranking.",
                hurt, dense_name, rerank_name)

    if summary:
        lines.insert(2, summary + "\n")
        print(summary)

    (results_dir(split) / f"qualitative_examples_{split}.md").write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test"], default="dev",
                        help="'test' is for the final evaluation only")
    parser.add_argument("--first-stage", choices=["auto", *DENSE_NAMES], default="auto")
    parser.add_argument("--depths", nargs="+", type=int, default=[20, 50])
    parser.add_argument("--skip-tfidf", action="store_true")
    parser.add_argument("--examples", type=int, default=5, help="examples shown per category")
    args = parser.parse_args()

    # 1. Data: analysis is done on the held-out val queries (dev split) or on the test queries
    corpus, queries, qrels = load_dataset(args.split)
    if args.split == "dev":
        _, val_q, _, val_qrels = split_dataset(queries, qrels)
        query_set, qrel_set, set_name = val_q, val_qrels, "val"
    else:
        query_set, qrel_set, set_name = queries, qrels, "test"

    query_ids = [q for q in query_set if qrel_set.get(q)]
    query_texts = [query_set[q] for q in query_ids]
    corpus_ids = list(corpus.keys())
    corpus_texts = [corpus[d]["text"] for d in corpus_ids]

    first_stage = choose_first_stage() if args.first_stage == "auto" else args.first_stage
    dense_name = DENSE_NAMES[first_stage]

    # 2. Collect runs and latencies for every method
    runs, latencies = {}, {}

    if not args.skip_tfidf:
        runs["TF-IDF"], latencies["TF-IDF"] = load_or_compute(
            runs_dir(args.split) / f"tfidf_{set_name}_top{TOP_K}.json",
            lambda: run_tfidf(corpus_ids, corpus_texts, query_ids, query_texts))

    runs["BM25"], latencies["BM25"] = load_or_compute(
        runs_dir(args.split) / f"bm25_{set_name}_top{TOP_K}.json",
        lambda: run_bm25(corpus_ids, corpus_texts, query_ids, query_texts))

    dense_latency = {}
    for key, name in DENSE_NAMES.items():
        run_path = runs_dir(args.split) / f"dense_{key}_{set_name}_top{TOP_K}.json"
        res_path = results_dir(args.split) / f"dense_results_{key}_{args.split}.json"
        if run_path.exists() and res_path.exists():
            runs[name] = json.loads(run_path.read_text())
            dense_latency[key] = json.loads(res_path.read_text())["latency"]["mean_ms"]
            latencies[name] = dense_latency[key]

    rerank_path = results_dir(args.split) / f"rerank_results_{first_stage}_{args.split}.json"
    rerank_latency = json.loads(rerank_path.read_text())["latency"] if rerank_path.exists() else {}

    for d in args.depths:
        run_path = runs_dir(args.split) / f"rerank_{first_stage}_{set_name}_depth{d}.json"
        if run_path.exists():
            name = f"{dense_name}+CE@{d}"
            runs[name] = json.loads(run_path.read_text())
            if f"depth{d}" in rerank_latency and first_stage in dense_latency:
                latencies[name] = dense_latency[first_stage] + rerank_latency[f"depth{d}"]["mean_ms"]

    if dense_name not in runs:
        raise SystemExit(f"Missing dense run for {dense_name}. Run dense_retrieval.py first.")
    if not any("+CE@" in n for n in runs):
        print("Warning: no reranked runs found. Run rerank.py to include them.")

    # Evaluate every method on exactly the same queries
    common = set(query_ids)
    for run in runs.values():
        common &= set(run)
    query_ids = [q for q in query_ids if q in common]
    runs = {n: {q: r[q] for q in query_ids} for n, r in runs.items()}
    print(f"\nAnalysing {len(query_ids)} {set_name} queries, methods: {', '.join(runs)}")

    # 3. Final table and figure
    metrics = results_table(runs, qrel_set, latencies, set_name, args.split)
    plot_comparison(metrics, set_name, args.split)

    # 4. Per-query metrics, breakdowns
    per_query = {name: per_query_metrics(run, qrel_set) for name, run in runs.items()}
    features = query_features(query_ids, query_set, qrel_set, corpus)
    breakdown(per_query, features, set_name, args.split)

    # 5. Failure analysis of the final system (chosen on dev)
    depths_present = [d for d in args.depths if f"{dense_name}+CE@{d}" in runs]
    final_system = pick_final_system(first_stage, depths_present)
    if final_system not in runs:
        final_system = dense_name
    failure_analysis(final_system, runs[final_system], qrel_set, query_set, corpus, features,
                     per_query, set_name, args.split, args.examples)

    # 6. Qualitative examples
    rerank_name = (f"{dense_name}+CE@{max(depths_present)}" if depths_present else "")
    qualitative_examples(runs, per_query, qrel_set, query_set, corpus, features,
                         dense_name, rerank_name, set_name, args.split, args.examples)

    print("\nDone. Outputs are in", results_dir(args.split))


if __name__ == "__main__":
    main()
