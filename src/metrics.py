"""
Shared evaluation metrics for the reranking and final-analysis scripts.

The definitions are identical to dense_retrieval.compute_metrics, so numbers are
directly comparable across methods:
    - Recall@K    : fraction of a query's relevant questions found in the top K
    - MRR@10      : reciprocal rank of the first relevant question in the top 10
    - nDCG@10     : binary relevance
"""

import numpy as np

KS = (1, 5, 10, 20, 50)


def first_relevant_rank(ranked, relevant_docs):
    """1-based rank of the first relevant question, or None if it is not in the list."""
    for rank, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant_docs:
            return rank
    return None


def per_query_metrics(run, qrels, ks=KS):
    """
    run   : {query_id: [doc_id, ...]} ranked best first
    qrels : {query_id: {doc_id: relevance}}
    Returns {query_id: {...}} with one row of metrics for every query in the run.
    """

    rows = {}

    for query_id, ranked in run.items():

        relevant_docs = set(qrels[query_id].keys())
        row = {}

        for k in ks:
            row[f"recall@{k}"] = len(set(ranked[:k]) & relevant_docs) / len(relevant_docs)

        rank = first_relevant_rank(ranked[:10], relevant_docs)
        row["mrr@10"] = 1 / rank if rank else 0.0

        dcg = sum(1 / np.log2(r + 1)
                  for r, doc_id in enumerate(ranked[:10], start=1)
                  if doc_id in relevant_docs)
        ideal = sum(1 / np.log2(r + 1)
                    for r in range(1, min(len(relevant_docs), 10) + 1))
        row["ndcg@10"] = dcg / ideal

        # Rank of the first relevant question anywhere in the list (None = not retrieved)
        row["first_rank"] = first_relevant_rank(ranked, relevant_docs)

        rows[query_id] = row

    return rows


def compute_metrics(run, qrels, ks=KS):
    """Mean metrics over all queries in the run."""

    rows = per_query_metrics(run, qrels, ks)
    names = [f"recall@{k}" for k in ks] + ["mrr@10", "ndcg@10"]

    metrics = {name: float(np.mean([r[name] for r in rows.values()])) for name in names}
    metrics["num_queries"] = len(rows)

    return metrics
