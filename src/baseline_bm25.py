from data_loader import load_dataset
from data_split import split_dataset

import bm25s

import numpy as np
import time


# 1. Load the dataset
corpus, queries, qrels = load_dataset("dev")

# 2. Split queries into development and validation sets
dev_queries, val_queries, dev_qrels, val_qrels = split_dataset(
    queries, qrels
)

# 3. Prepare corpus questions
corpus_ids = list(corpus.keys())
corpus_texts = [corpus[doc_id]["text"] for doc_id in corpus_ids]

# 4. Tokenize and build the BM25 index
start_time = time.perf_counter()

corpus_tokens = bm25s.tokenize(corpus_texts)

bm25 = bm25s.BM25(method="lucene")
bm25.index(corpus_tokens)

indexing_time = time.perf_counter() - start_time

print("BM25 indexing time:", round(indexing_time, 2), "seconds")


# 5. Retrieve the top K most relevant questions
def retrieve(query, k=10):

    # Tokenize the query
    query_tokens = bm25s.tokenize([query])

    # Retrieve the top K results
    results, scores = bm25.retrieve(query_tokens, k=k)

    # Convert corpus positions into document IDs
    result_ids = [corpus_ids[index] for index in results[0]]

    return result_ids, scores[0]


# 6. Evaluate retrieval performance
def evaluate(query_set, qrel_set, k=10):

    recalls = []
    reciprocal_ranks = []

    start_time = time.perf_counter()

    for query_id, query_text in query_set.items():

        relevant_docs = set(qrel_set.get(query_id, {}).keys())

        if not relevant_docs:
            continue

        # Retrieve similar questions
        results, scores = retrieve(query_text, k)

        # Calculate Recall@K
        relevant_found = len(set(results) & relevant_docs)
        recall = relevant_found / len(relevant_docs)
        recalls.append(recall)

        # Calculate Reciprocal Rank
        reciprocal_rank = 0

        for rank, doc_id in enumerate(results, start=1):
            if doc_id in relevant_docs:
                reciprocal_rank = 1 / rank
                break

        reciprocal_ranks.append(reciprocal_rank)

    retrieval_time = time.perf_counter() - start_time

    print("Queries evaluated:", len(recalls))
    print("Recall@" + str(k) + ":", round(np.mean(recalls), 4))
    print("MRR@" + str(k) + ":", round(np.mean(reciprocal_ranks), 4))
    print("Retrieval time:", round(retrieval_time, 2), "seconds")


# 7. Evaluate development queries
print("\n--- BM25 Development Results ---")
evaluate(dev_queries, dev_qrels)


# 8. Evaluate validation queries
print("\n--- BM25 Validation Results ---")
evaluate(val_queries, val_qrels)


# 9. Inspect an example query
def inspect_query(query_id, k=10):

    query_text = queries[query_id]

    results, scores = retrieve(query_text, k)

    print("\n" + "=" * 80)
    print("ORIGINAL QUERY:")
    print(query_text)

    print("\nTOP", k, "RETRIEVED QUESTIONS:")

    for rank, (doc_id, score) in enumerate(zip(results, scores), start=1):

        print("\nRank:", rank)
        print("BM25 Score:", round(float(score), 4))
        print("Question:", corpus[doc_id]["text"])

        if doc_id in qrels.get(query_id, {}):
            print("Known relevant match: YES")
        else:
            print("Known relevant match: NO")


# Inspect one development query
example_query_id = next(iter(dev_queries))
inspect_query(example_query_id)