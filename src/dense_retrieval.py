"""
Dense retrieval on BEIR Quora with all-MiniLM-L6-v2 and all-mpnet-base-v2.

Pipeline
    1. Load the dataset and reuse the same 80/20 dev/val query split as the baselines
    2. Encode the whole corpus ONCE per model (chunked, resumable, cached to disk)
    3. Encode queries, rank the corpus by cosine similarity, never letting a query retrieve itself
    4. Report Recall@K, MRR@10, nDCG@10, timing, and model-size statistics
    5. Save the top-50 run files (for the cross-encoder reranker) and a comparison figure

Usage
    python dense_retrieval.py                          # both models, dev split (dev + val queries)
    python dense_retrieval.py --models minilm          # one model only
    python dense_retrieval.py --split test             # FINAL evaluation only, run once at the end
    python dense_retrieval.py --cache-dir /content/drive/MyDrive/quora_cache   # Colab: cache on Drive
"""

import argparse
import gc
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from data_loader import load_dataset
from data_split import split_dataset


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_DIR / "results"


def results_dir(split):
    """Outputs of each split live in results/dev/ or results/test/."""
    return RESULTS_ROOT / split


def runs_dir(split):
    return results_dir(split) / "runs"


def figures_dir(split):
    return results_dir(split) / "figures"

MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
}

KS = (1, 5, 10, 20, 50)   # Recall cutoffs; 20 and 50 are the reranking depths
MAX_K = max(KS)


def sync(device):
    # GPU work is asynchronous, so wait for it before reading the clock
    if device == "cuda":
        torch.cuda.synchronize()


# ----------------------------------------------------------------------------
# 1. Corpus embeddings (computed once, cached to disk, resumable)
# ----------------------------------------------------------------------------

def get_corpus_embeddings(model, key, corpus_ids, corpus_texts,
                          cache_dir, chunk_size, batch_size, device):
    """
    Returns (embeddings as float32 numpy [N, dim], total encoding seconds).

    Embeddings are L2-normalized, so a dot product equals cosine similarity.
    Encoding is done in chunks and progress is saved after each chunk, so a
    Colab disconnect only loses the current chunk.
    """

    model_dir = cache_dir / key
    chunk_dir = model_dir / "chunks"
    model_dir.mkdir(parents=True, exist_ok=True)

    final_path = model_dir / "corpus_embeddings.npy"
    meta_path = model_dir / "meta.json"
    progress_path = model_dir / "progress.json"

    # Fingerprint of the corpus order, so a stale cache is never reused
    ids_hash = hashlib.md5("\n".join(corpus_ids).encode()).hexdigest()

    # Use the finished cache if it matches this corpus
    if final_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta["ids_hash"] == ids_hash and meta["num_docs"] == len(corpus_ids):
            print(f"[{key}] Loaded cached corpus embeddings from {final_path}")
            return np.load(final_path), meta["encode_seconds"]
        print(f"[{key}] Cache does not match this corpus, re-encoding")

    # Resume partial progress if it matches this corpus and chunk size
    progress = {"ids_hash": ids_hash, "chunk_size": chunk_size, "chunk_seconds": {}}
    if progress_path.exists():
        old = json.loads(progress_path.read_text())
        if old.get("ids_hash") == ids_hash and old.get("chunk_size") == chunk_size:
            progress = old
        else:
            shutil.rmtree(chunk_dir, ignore_errors=True)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    num_chunks = (len(corpus_texts) + chunk_size - 1) // chunk_size

    for chunk_index in range(num_chunks):

        chunk_path = chunk_dir / f"chunk_{chunk_index:04d}.npy"

        if chunk_path.exists() and str(chunk_index) in progress["chunk_seconds"]:
            continue

        start = chunk_index * chunk_size
        texts = corpus_texts[start:start + chunk_size]

        sync(device)
        t0 = time.perf_counter()

        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

        sync(device)
        elapsed = time.perf_counter() - t0

        np.save(chunk_path, embeddings)
        progress["chunk_seconds"][str(chunk_index)] = elapsed
        progress_path.write_text(json.dumps(progress))

        done = start + len(texts)
        print(f"[{key}] Encoded {done:,}/{len(corpus_texts):,} questions "
              f"(chunk {chunk_index + 1}/{num_chunks}, {elapsed:.1f}s)")

    # Merge chunks into one file
    all_embeddings = np.concatenate(
        [np.load(chunk_dir / f"chunk_{i:04d}.npy") for i in range(num_chunks)]
    )
    assert all_embeddings.shape[0] == len(corpus_ids)

    encode_seconds = float(sum(progress["chunk_seconds"].values()))

    np.save(final_path, all_embeddings)
    meta_path.write_text(json.dumps({
        "model": MODELS[key],
        "ids_hash": ids_hash,
        "num_docs": len(corpus_ids),
        "dim": int(all_embeddings.shape[1]),
        "encode_seconds": encode_seconds,
        "encode_device": device,
    }))

    # Remove the temporary chunk files
    shutil.rmtree(chunk_dir, ignore_errors=True)
    progress_path.unlink(missing_ok=True)

    return all_embeddings, encode_seconds


# ----------------------------------------------------------------------------
# 2. Retrieval
# ----------------------------------------------------------------------------

def search(query_embeddings, corpus_tensor, exclude_index, k=MAX_K, batch_size=256):
    """
    Cosine-similarity search (embeddings are normalized, so dot product = cosine).

    exclude_index[i] is the corpus position of query i itself (or -1).
    That score is set to -inf so a query can never retrieve itself.
    """

    all_indices = []

    for start in range(0, len(query_embeddings), batch_size):

        scores = query_embeddings[start:start + batch_size] @ corpus_tensor.T

        # Exclude the query itself
        exclude = exclude_index[start:start + batch_size]
        rows = (exclude >= 0).nonzero(as_tuple=True)[0]
        scores[rows, exclude[rows]] = float("-inf")

        top = torch.topk(scores, k, dim=1).indices
        all_indices.append(top.cpu())

    return torch.cat(all_indices).numpy()


# ----------------------------------------------------------------------------
# 3. Evaluation (same definitions as the baselines, plus nDCG@10)
# ----------------------------------------------------------------------------

def compute_metrics(run, qrel_set):

    recalls = {k: [] for k in KS}
    reciprocal_ranks = []
    ndcgs = []

    for query_id, ranked in run.items():

        relevant_docs = set(qrel_set[query_id].keys())

        for k in KS:
            found = len(set(ranked[:k]) & relevant_docs)
            recalls[k].append(found / len(relevant_docs))

        # MRR@10
        reciprocal_rank = 0
        for rank, doc_id in enumerate(ranked[:10], start=1):
            if doc_id in relevant_docs:
                reciprocal_rank = 1 / rank
                break
        reciprocal_ranks.append(reciprocal_rank)

        # nDCG@10 with binary relevance
        dcg = sum(1 / np.log2(rank + 1)
                  for rank, doc_id in enumerate(ranked[:10], start=1)
                  if doc_id in relevant_docs)
        ideal = sum(1 / np.log2(rank + 1)
                    for rank in range(1, min(len(relevant_docs), 10) + 1))
        ndcgs.append(dcg / ideal)

    metrics = {f"recall@{k}": float(np.mean(recalls[k])) for k in KS}
    metrics["mrr@10"] = float(np.mean(reciprocal_ranks))
    metrics["ndcg@10"] = float(np.mean(ndcgs))
    metrics["num_queries"] = len(run)

    return metrics


def evaluate_set(model, corpus_tensor, corpus_ids, id_to_index,
                 query_set, qrel_set, device, dtype, batch_size):

    # Same rule as the baselines: skip queries with no relevant documents
    query_ids = [qid for qid in query_set if qrel_set.get(qid)]
    query_texts = [query_set[qid] for qid in query_ids]

    # Encode queries
    sync(device)
    t0 = time.perf_counter()
    query_embeddings = model.encode(
        query_texts,
        batch_size=batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).to(dtype)
    sync(device)
    query_encoding_time = time.perf_counter() - t0

    # Search the corpus
    exclude_index = torch.tensor(
        [id_to_index.get(qid, -1) for qid in query_ids], device=device
    )

    t0 = time.perf_counter()
    top_indices = search(query_embeddings, corpus_tensor, exclude_index)
    sync(device)
    search_time = time.perf_counter() - t0

    run = {
        qid: [corpus_ids[i] for i in row]
        for qid, row in zip(query_ids, top_indices)
    }

    metrics = compute_metrics(run, qrel_set)
    metrics["query_encoding_seconds"] = query_encoding_time
    metrics["search_seconds"] = search_time
    metrics["total_retrieval_seconds"] = query_encoding_time + search_time

    return metrics, run


def measure_latency(model, corpus_tensor, query_texts, device, dtype,
                    k=10, num_queries=200, warmup=10):
    """Single-query latency (encode + search, batch size 1), like a live system."""

    times = []

    for i, text in enumerate(query_texts[:num_queries + warmup]):

        sync(device)
        t0 = time.perf_counter()

        q = model.encode(
            [text], convert_to_tensor=True,
            normalize_embeddings=True, show_progress_bar=False,
        ).to(dtype)
        torch.topk(q @ corpus_tensor.T, k, dim=1)

        sync(device)
        elapsed = time.perf_counter() - t0

        if i >= warmup:
            times.append(elapsed * 1000)

    return {
        "mean_ms": float(np.mean(times)),
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
    }


# ----------------------------------------------------------------------------
# 4. Run one model end to end
# ----------------------------------------------------------------------------

def run_model(key, args, corpus_ids, corpus_texts, id_to_index, eval_sets):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    batch_size = args.batch_size or (256 if device == "cuda" else 64)

    print("\n" + "=" * 80)
    print(f"MODEL: {MODELS[key]}   (device: {device})")
    print("=" * 80)

    model = SentenceTransformer(MODELS[key], device=device)

    num_parameters = sum(p.numel() for p in model.parameters())

    # Corpus embeddings: encoded once, then loaded from disk
    corpus_embeddings, encode_seconds = get_corpus_embeddings(
        model, key, corpus_ids, corpus_texts,
        Path(args.cache_dir), args.chunk_size, batch_size, device,
    )

    corpus_tensor = torch.from_numpy(corpus_embeddings).to(device=device, dtype=dtype)

    info = {
        "model": MODELS[key],
        "device": device,
        "parameters_millions": round(num_parameters / 1e6, 2),
        "embedding_dim": int(corpus_embeddings.shape[1]),
        "index_size_mb_float32": round(corpus_embeddings.nbytes / 1e6, 1),
        "corpus_size": len(corpus_ids),
        "corpus_encode_seconds": round(encode_seconds, 1),
        "corpus_encode_questions_per_second": round(len(corpus_ids) / encode_seconds, 1),
        "max_seq_length": model.max_seq_length,
        "search_dtype": str(dtype).replace("torch.", ""),
    }

    results = {"info": info, "eval": {}}

    for set_name, (query_set, qrel_set) in eval_sets.items():

        metrics, run = evaluate_set(
            model, corpus_tensor, corpus_ids, id_to_index,
            query_set, qrel_set, device, dtype, batch_size,
        )
        results["eval"][set_name] = metrics

        # Save the top-50 run so the reranker can reuse it without re-searching
        runs_dir(args.split).mkdir(parents=True, exist_ok=True)
        run_path = runs_dir(args.split) / f"dense_{key}_{set_name}_top{MAX_K}.json"
        run_path.write_text(json.dumps(run))

        print(f"\n--- {key} {set_name} results ---")
        print("Queries evaluated:", metrics["num_queries"])
        for k in KS:
            print(f"Recall@{k}:", round(metrics[f"recall@{k}"], 4))
        print("MRR@10:", round(metrics["mrr@10"], 4))
        print("nDCG@10:", round(metrics["ndcg@10"], 4))
        print("Query encoding time:", round(metrics["query_encoding_seconds"], 2), "seconds")
        print("Search time:", round(metrics["search_seconds"], 2), "seconds")

    # Single-query latency, measured on the first evaluation set
    first_set = next(iter(eval_sets.values()))[0]
    results["latency"] = measure_latency(
        model, corpus_tensor, list(first_set.values()), device, dtype
    )

    print("\nCorpus encoding time:", info["corpus_encode_seconds"], "seconds",
          f"({info['corpus_encode_questions_per_second']} questions/s)")
    print("Single-query latency (mean / p95):",
          round(results["latency"]["mean_ms"], 2), "/",
          round(results["latency"]["p95_ms"], 2), "ms")

    # Save results
    results_dir(args.split).mkdir(parents=True, exist_ok=True)
    out_path = results_dir(args.split) / f"dense_results_{key}_{args.split}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print("Saved:", out_path)

    # Free memory before the next model
    del model, corpus_tensor, corpus_embeddings
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return results


# ----------------------------------------------------------------------------
# 5. Comparison table and figure
# ----------------------------------------------------------------------------

def summarize(split):

    import matplotlib.pyplot as plt

    # Compare on the validation queries (dev split) or the test queries
    eval_name = "val" if split == "dev" else "test"

    all_results = {}
    for key in MODELS:
        path = results_dir(split) / f"dense_results_{key}_{split}.json"
        if path.exists():
            all_results[key] = json.loads(path.read_text())

    if not all_results:
        return

    print("\n" + "=" * 80)
    print(f"MODEL COMPARISON ({eval_name} queries)")
    print("=" * 80)

    header = f"{'Model':<10}{'Params(M)':>10}{'Dim':>6}{'R@10':>8}{'R@50':>8}" \
             f"{'MRR@10':>9}{'nDCG@10':>9}{'Enc(min)':>10}{'Lat(ms)':>9}"
    print(header)

    for key, res in all_results.items():
        m = res["eval"][eval_name]
        print(f"{key:<10}{res['info']['parameters_millions']:>10}"
              f"{res['info']['embedding_dim']:>6}"
              f"{m['recall@10']:>8.4f}{m['recall@50']:>8.4f}"
              f"{m['mrr@10']:>9.4f}{m['ndcg@10']:>9.4f}"
              f"{res['info']['corpus_encode_seconds'] / 60:>10.1f}"
              f"{res['latency']['mean_ms']:>9.1f}")

    # Figure: quality, speed, size
    names = list(all_results.keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Quality
    metric_names = ["recall@10", "recall@50", "mrr@10", "ndcg@10"]
    width = 0.8 / len(names)
    for i, key in enumerate(names):
        values = [all_results[key]["eval"][eval_name][m] for m in metric_names]
        axes[0].bar(np.arange(len(metric_names)) + i * width, values, width, label=key)
    axes[0].set_xticks(np.arange(len(metric_names)) + width * (len(names) - 1) / 2)
    axes[0].set_xticklabels(["Recall@10", "Recall@50", "MRR@10", "nDCG@10"])
    axes[0].set_title("Retrieval quality")
    axes[0].legend()

    # Speed
    axes[1].bar(names, [all_results[k]["latency"]["mean_ms"] for k in names])
    axes[1].set_ylabel("Mean single-query latency (ms)")
    axes[1].set_title("Query speed")

    # Size
    axes[2].bar(names, [all_results[k]["info"]["parameters_millions"] for k in names])
    axes[2].set_ylabel("Parameters (millions)")
    axes[2].set_title("Model size")

    plt.tight_layout()
    figures_dir(split).mkdir(parents=True, exist_ok=True)
    fig_path = figures_dir(split) / f"dense_comparison_{split}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("\nSaved figure:", fig_path)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--split", choices=["dev", "test"], default="dev",
                        help="'test' is for the final evaluation only")
    parser.add_argument("--cache-dir", default=os.environ.get(
        "DENSE_CACHE_DIR", str(PROJECT_DIR / "cache" / "embeddings")))
    parser.add_argument("--chunk-size", type=int, default=50_000,
                        help="questions encoded per saved chunk")
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    # 1. Load the dataset
    corpus, queries, qrels = load_dataset(args.split)

    # 2. Evaluation sets
    if args.split == "dev":
        # Same split as the baselines (random_state=42)
        dev_queries, val_queries, dev_qrels, val_qrels = split_dataset(queries, qrels)
        eval_sets = {
            "dev": (dev_queries, dev_qrels),
            "val": (val_queries, val_qrels),
        }
    else:
        eval_sets = {"test": (queries, qrels)}

    # 3. Prepare corpus questions
    corpus_ids = list(corpus.keys())
    corpus_texts = [corpus[doc_id]["text"] for doc_id in corpus_ids]
    id_to_index = {doc_id: index for index, doc_id in enumerate(corpus_ids)}

    # 4. Run each model
    for key in args.models:
        run_model(key, args, corpus_ids, corpus_texts, id_to_index, eval_sets)

    # 5. Compare models
    summarize(args.split)


if __name__ == "__main__":
    main()
