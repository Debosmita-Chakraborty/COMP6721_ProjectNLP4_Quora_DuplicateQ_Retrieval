"""
Retrieve-and-rerank on BEIR Quora with a pretrained MS MARCO cross-encoder.

Pipeline
    1. Load the top-50 first-stage run saved by dense_retrieval.py
    2. Score every (query, candidate) pair with the cross-encoder ONCE at the maximum depth
       (scores are cached to disk, so a Colab disconnect only loses the current block)
    3. Rerank the top-D candidates for each depth D (default 20 and 50); candidates
       below rank D keep their first-stage order
    4. Report Recall@K, MRR@10, nDCG@10 for the first stage and every depth,
       plus single-query reranking latency, and save a depth-study figure

Because a cross-encoder scores each pair independently, the depth-20 result is
exactly the top-20 subset of the depth-50 scores, so one scoring pass gives the
whole depth study.

Usage
    python rerank.py                                 # auto-pick first stage on dev, dev + val queries
    python rerank.py --first-stage mpnet --depths 10 20 30 50
    python rerank.py --split test                    # FINAL evaluation only, run once at the end
    python rerank.py --max-queries 200               # quick smoke test
    python rerank.py --cache-dir /content/drive/MyDrive/quora_cache   # Colab: keep scores on Drive

Prerequisite
    python dense_retrieval.py [--split test]         # writes results/runs/dense_<model>_<set>_top50.json
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from data_loader import load_dataset
from data_split import split_dataset
from metrics import KS, compute_metrics


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
RUNS_DIR = RESULTS_DIR / "runs"

DENSE_MODELS = ("minilm", "mpnet")
CROSS_ENCODERS = {
    "minilm-l6": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "minilm-l12": "cross-encoder/ms-marco-MiniLM-L-12-v2",
}
MAX_FIRST_STAGE_DEPTH = 50   # dense_retrieval.py saves the top 50


# ----------------------------------------------------------------------------
# 1. First-stage runs
# ----------------------------------------------------------------------------

def choose_first_stage():
    """
    Pick the first-stage model with the higher DEV Recall@20 (the pool the reranker
    can reach). Only dev queries are used for this choice, so the test set is never
    touched while selecting.
    """

    best_key, best_recall = None, -1.0

    for key in DENSE_MODELS:
        path = RESULTS_DIR / f"dense_results_{key}_dev.json"
        if not path.exists():
            continue
        recall = json.loads(path.read_text())["eval"]["dev"]["recall@20"]
        print(f"  first-stage candidate {key}: dev Recall@20 = {recall:.4f}")
        if recall > best_recall:
            best_key, best_recall = key, recall

    if best_key is None:
        raise SystemExit(
            "No dense results found. Run `python dense_retrieval.py` first, "
            "or pass --first-stage explicitly."
        )

    return best_key


def load_first_stage_run(key, set_name):

    path = RUNS_DIR / f"dense_{key}_{set_name}_top{MAX_FIRST_STAGE_DEPTH}.json"

    if not path.exists():
        raise SystemExit(
            f"Missing {path}.\n"
            f"Run `python dense_retrieval.py --models {key}"
            f"{' --split test' if set_name == 'test' else ''}` first."
        )

    return json.loads(path.read_text())


# ----------------------------------------------------------------------------
# 2. Cross-encoder scoring (cached, resumable)
# ----------------------------------------------------------------------------

def score_run(cross_encoder, ce_name, run, queries, corpus, depth,
              cache_path, batch_size, block_size=250):
    """
    Returns {query_id: [score for each of the first `depth` candidates]}.

    Scores are saved every few blocks. Cached queries are reused only if the same
    cross-encoder produced them and they cover at least `depth` candidates.
    """

    scores = {}

    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        if cached.get("cross_encoder") == ce_name:
            scores = {qid: s for qid, s in cached["scores"].items() if len(s) >= depth}
            if scores:
                print(f"  Reusing {len(scores)} cached query scores from {cache_path.name}")

    todo = [qid for qid in run if qid not in scores]

    for start in range(0, len(todo), block_size):

        block = todo[start:start + block_size]

        pairs, owners = [], []
        for qid in block:
            for doc_id in run[qid][:depth]:
                pairs.append((queries[qid], corpus[doc_id]["text"]))
                owners.append(qid)

        predictions = cross_encoder.predict(
            pairs, batch_size=batch_size, show_progress_bar=False
        )

        for qid in block:
            scores[qid] = []
        for qid, value in zip(owners, predictions):
            scores[qid].append(float(value))

        done = start + len(block)
        print(f"  Scored {done}/{len(todo)} queries")

        if (start // block_size) % 4 == 3 or done == len(todo):
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps({"cross_encoder": ce_name, "scores": scores}))

    return scores


def rerank_run(run, scores, depth):
    """Reorder the top `depth` candidates by cross-encoder score; keep the rest as is."""

    reranked = {}

    for qid, ranked in run.items():
        top = ranked[:depth]
        # Stable sort on the negated score keeps first-stage order for exact ties
        order = np.argsort(-np.array(scores[qid][:depth]), kind="stable")
        reranked[qid] = [top[i] for i in order] + ranked[depth:]

    return reranked


# ----------------------------------------------------------------------------
# 3. Latency
# ----------------------------------------------------------------------------

def measure_latency(cross_encoder, run, queries, corpus, depth,
                    num_queries=100, warmup=5):
    """Time to rerank ONE query's top-`depth` candidates (one batch), like a live system."""

    import torch

    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    times = []

    for i, qid in enumerate(list(run)[:num_queries + warmup]):

        pairs = [(queries[qid], corpus[d]["text"]) for d in run[qid][:depth]]

        sync()
        t0 = time.perf_counter()
        cross_encoder.predict(pairs, batch_size=depth, show_progress_bar=False)
        sync()
        elapsed = (time.perf_counter() - t0) * 1000

        if i >= warmup:
            times.append(elapsed)

    return {
        "mean_ms": float(np.mean(times)),
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
    }


# ----------------------------------------------------------------------------
# 4. Figure
# ----------------------------------------------------------------------------

def plot_depth_study(results, eval_name, depths, split, first_stage):

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [0] + list(depths)
    labels = ["No rerank"] + [f"Top {d}" for d in depths]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for name, pretty in [("mrr@10", "MRR@10"), ("ndcg@10", "nDCG@10"),
                         ("recall@10", "Recall@10")]:
        values = [results["first_stage"][eval_name][name]] + \
                 [results["rerank"][f"depth{d}"][eval_name][name] for d in depths]
        axes[0].plot(range(len(xs)), values, marker="o", label=pretty)

    axes[0].set_xticks(range(len(xs)))
    axes[0].set_xticklabels(labels)
    axes[0].set_xlabel("Candidates reranked")
    axes[0].set_title(f"Reranking depth ({first_stage} first stage, {eval_name} queries)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    latencies = [results["latency"][f"depth{d}"]["mean_ms"] for d in depths]
    axes[1].bar([f"Top {d}" for d in depths], latencies)
    axes[1].set_ylabel("Mean reranking latency per query (ms)")
    axes[1].set_title("Reranking cost")

    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"rerank_depth_{first_stage}_{split}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved figure:", path)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test"], default="dev",
                        help="'test' is for the final evaluation only")
    parser.add_argument("--first-stage", choices=["auto", *DENSE_MODELS], default="auto",
                        help="auto = higher dev Recall@20")
    parser.add_argument("--cross-encoder", choices=list(CROSS_ENCODERS), default="minilm-l6")
    parser.add_argument("--depths", nargs="+", type=int, default=[20, 50])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-queries", type=int, default=None,
                        help="use only the first N queries of each set (smoke test)")
    parser.add_argument("--cache-dir", default=os.environ.get(
        "RERANK_CACHE_DIR", str(PROJECT_DIR / "cache" / "rerank")))
    args = parser.parse_args()

    depths = sorted(set(args.depths))
    if depths[0] < 1 or depths[-1] > MAX_FIRST_STAGE_DEPTH:
        raise SystemExit(f"--depths must be between 1 and {MAX_FIRST_STAGE_DEPTH}")
    max_depth = depths[-1]

    import torch
    from sentence_transformers import CrossEncoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = args.batch_size or (256 if device == "cuda" else 32)
    ce_name = CROSS_ENCODERS[args.cross_encoder]

    # 1. Dataset and evaluation sets (same sets as dense_retrieval.py)
    corpus, queries, qrels = load_dataset(args.split)

    if args.split == "dev":
        dev_q, val_q, dev_qrels, val_qrels = split_dataset(queries, qrels)
        eval_sets = {"dev": (dev_q, dev_qrels), "val": (val_q, val_qrels)}
    else:
        eval_sets = {"test": (queries, qrels)}

    # 2. First stage
    if args.first_stage == "auto":
        print("Choosing first stage on dev Recall@20:")
        first_stage = choose_first_stage()
    else:
        first_stage = args.first_stage
    print(f"First stage: {first_stage}   Cross-encoder: {ce_name}   Device: {device}")

    cross_encoder = CrossEncoder(ce_name, device=device, max_length=128)
    num_parameters = sum(p.numel() for p in cross_encoder.model.parameters())

    results = {
        "info": {
            "first_stage": first_stage,
            "cross_encoder": ce_name,
            "cross_encoder_parameters_millions": round(num_parameters / 1e6, 2),
            "device": device,
            "depths": depths,
            "split": args.split,
        },
        "first_stage": {},
        "rerank": {f"depth{d}": {} for d in depths},
        "latency": {},
    }

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    ce_tag = ce_name.split("/")[-1]

    last_run = None

    for set_name, (query_set, qrel_set) in eval_sets.items():

        print(f"\n=== {set_name} queries ===")

        run = load_first_stage_run(first_stage, set_name)

        # Keep only queries that have relevance judgments (as in every other script)
        run = {qid: ranked for qid, ranked in run.items() if qrel_set.get(qid)}
        if args.max_queries:
            run = dict(list(run.items())[:args.max_queries])

        # Score once at the maximum depth
        scores = score_run(
            cross_encoder, ce_name, run, query_set, corpus, max_depth,
            cache_dir / f"ce_scores_{ce_tag}_{first_stage}_{set_name}.json", batch_size,
        )

        # First stage as the "no rerank" reference on exactly the same queries
        results["first_stage"][set_name] = compute_metrics(run, qrel_set)

        print(f"{'':<14}{'R@10':>8}{'R@20':>8}{'R@50':>8}{'MRR@10':>9}{'nDCG@10':>9}")

        def show(label, m):
            print(f"{label:<14}{m['recall@10']:>8.4f}{m['recall@20']:>8.4f}"
                  f"{m['recall@50']:>8.4f}{m['mrr@10']:>9.4f}{m['ndcg@10']:>9.4f}")

        show("first stage", results["first_stage"][set_name])

        for d in depths:
            reranked = rerank_run(run, scores, d)
            metrics = compute_metrics(reranked, qrel_set)
            results["rerank"][f"depth{d}"][set_name] = metrics
            show(f"rerank top{d}", metrics)

            suffix = "" if not args.max_queries else "_smoke"
            (RUNS_DIR / f"rerank_{first_stage}_{set_name}_depth{d}{suffix}.json").write_text(
                json.dumps(reranked))

        last_run = (run, query_set)

    # 3. Latency (measured once, on the last evaluation set)
    run, query_set = last_run
    print("\nMeasuring single-query reranking latency...")
    for d in depths:
        results["latency"][f"depth{d}"] = measure_latency(
            cross_encoder, run, query_set, corpus, d)
        print(f"  depth {d}: {results['latency'][f'depth{d}']['mean_ms']:.1f} ms mean")

    # 4. Save results and figure (val for the dev split, test for the test split)
    suffix = "" if not args.max_queries else "_smoke"
    out_path = RESULTS_DIR / f"rerank_results_{first_stage}_{args.split}{suffix}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print("Saved:", out_path)

    plot_depth_study(results, "val" if args.split == "dev" else "test",
                     depths, args.split + suffix, first_stage)


if __name__ == "__main__":
    main()
