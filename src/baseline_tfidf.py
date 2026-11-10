from data_loader import load_dataset
from data_split import split_dataset

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

import numpy as np
import time

# Term Frequency–Inverse Document Frequency

# 1. Load dataset
corpus, queries, qrels = load_dataset("dev")

# 2. Split queries into development and validation sets
dev_queries, val_queries, dev_qrels, val_qrels = split_dataset(
    queries, qrels
)

# 3. Prepare corpus questions
corpus_ids = list(corpus.keys())
corpus_texts = [corpus[doc_id]["text"] for doc_id in corpus_ids]

# Map document IDs to their positions in the corpus
corpus_id_to_index = {
    doc_id: index for index, doc_id in enumerate(corpus_ids)
}

# 4. Convert corpus questions into TF-IDF vectors
vectorizer = TfidfVectorizer()

start_time = time.perf_counter()

corpus_vectors = vectorizer.fit_transform(corpus_texts)

indexing_time = time.perf_counter() - start_time

print("TF-IDF indexing time:", round(indexing_time, 2), "seconds")

# 5. Retrieve the top K most similar questions
def retrieve(query, query_id, k=10):

    # Convert query into a TF-IDF vector
    query_vector = vectorizer.transform([query])

    # Calculate cosine similarity against the corpus
    scores = linear_kernel(query_vector, corpus_vectors)[0]

    # Exclude the query itself if it exists in the corpus
    if query_id in corpus_id_to_index:
        scores[corpus_id_to_index[query_id]] = -1

    # Find the top K results
    top_indices = np.argpartition(scores, -k)[-k:]

    # Sort results by similarity score
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

    return [corpus_ids[index] for index in top_indices]


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
        results = retrieve(query_text, query_id, k)

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


# 7. Evaluate on development queries
print("\n--- TF-IDF Development Results ---")
evaluate(dev_queries, dev_qrels)


# 8. Evaluate on validation queries
print("\n--- TF-IDF Validation Results ---")
evaluate(val_queries, val_qrels)


def inspect_query(query_id, k=10):

    query_text = queries[query_id]

    # Get the top K retrieved questions
    results = retrieve(query_text, query_id, k)

    # Display the original query
    print("\n" + "=" * 80)
    print("ORIGINAL QUERY:")
    print(query_text)

    # Display the retrieved questions
    print("\nTOP", k, "RETRIEVED QUESTIONS:")

    query_vector = vectorizer.transform([query_text])
    scores = linear_kernel(query_vector, corpus_vectors)[0]

    for rank, doc_id in enumerate(results, start=1):

        index = corpus_id_to_index[doc_id]
        score = scores[index]

        print("\nRank:", rank)
        print("Similarity:", round(score, 4))
        print("Question:", corpus[doc_id]["text"])

        # Show whether this question is a known relevant match
        if doc_id in qrels.get(query_id, {}):
            print("Known relevant match: YES")
        else:
            print("Known relevant match: NO")

# Inspect one development query
example_query_id = next(iter(dev_queries))
inspect_query(example_query_id)