from data_loader import load_dataset
from data_split import split_dataset

import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Load dataset
corpus, queries, qrels = load_dataset("dev")

# Split queries
dev_queries, val_queries, dev_qrels, val_qrels = split_dataset(
    queries, qrels
)

# Basic statistics
query_lengths = [len(q.split()) for q in queries.values()]
corpus_lengths = [len(doc["text"].split()) for doc in corpus.values()]

print("Corpus questions:", len(corpus))
print("Queries:", len(queries))
print("Average query length:", round(np.mean(query_lengths), 2))
print("Average corpus question length:", round(np.mean(corpus_lengths), 2))

# Graph 1: Query lengths
plt.figure(figsize=(9, 5))
plt.hist(query_lengths, bins=30)
plt.xlabel("Number of words")
plt.ylabel("Number of queries")
plt.title("Query Length Distribution")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "query_length.png", dpi=300, bbox_inches="tight")
plt.show()


# Graph 2: Most frequent words
words = []

for question in queries.values():
    for word in question.lower().split():
        word = word.strip(".,!?;:'\"()[]{}")

        if word and word not in ENGLISH_STOP_WORDS:
            words.append(word)

top_words = Counter(words).most_common(15)

word_labels = [word[0] for word in top_words]
word_counts = [word[1] for word in top_words]

plt.figure(figsize=(9, 6))
plt.barh(word_labels[::-1], word_counts[::-1])
plt.xlabel("Frequency")
plt.title("15 Most Frequent Words in Queries")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "frequent_words.png", dpi=300, bbox_inches="tight")
plt.show()