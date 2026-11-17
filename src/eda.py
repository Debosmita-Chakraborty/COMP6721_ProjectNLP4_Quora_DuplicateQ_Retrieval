from data_loader import load_dataset
from data_split import split_dataset
from scipy.stats import spearmanr
from matplotlib.colors import LogNorm

import matplotlib.pyplot as plt
import numpy as np
import re
import json

from collections import Counter
from pathlib import Path

# Output directories

BASE_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = BASE_DIR / "results" / "figures"
RESULTS_DIR = BASE_DIR / "results"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Load dataset
corpus, queries, qrels = load_dataset("dev")

dev_queries, val_queries, dev_qrels, val_qrels = split_dataset(
    queries, qrels
)

# Helpers

TOKEN_RE = re.compile(r"\w+(?:'\w+)?", re.UNICODE)


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def normalize(text):
    """Lowercase and collapse whitespace."""
    return " ".join(text.lower().split())


def normalize_loose(text):
    """Also ignores punctuation."""
    return " ".join(tokenize(text))


def percentage(count, total):
    return count / total * 100 if total else 0.0


def describe(values):
    values = np.asarray(values)

    if len(values) == 0:
        return {}

    p25, p50, p75, p90, p95, p99 = np.percentile(
        values, [25, 50, 75, 90, 95, 99]
    )

    return {
        "mean": values.mean(),
        "std": values.std(ddof=1) if len(values) > 1 else 0.0,
        "min": values.min(),
        "p25": p25,
        "median": p50,
        "p75": p75,
        "p90": p90,
        "p95": p95,
        "p99": p99,
        "max": values.max()
    }


def relevant_ids(judgments):
    if isinstance(judgments, dict):
        return [
            doc_id for doc_id, score in judgments.items()
            if float(score) > 0
        ]

    return list(judgments)


def md_table(title, rows, headers=("Statistic", "Value"), note=None):
    lines = [f"## {title}", ""]

    if note:
        lines += [f"*{note}*", ""]

    lines += [
        f"| {headers[0]} | {headers[1]} |",
        "|---|---:|"
    ]

    lines += [f"| {key} | {value} |" for key, value in rows]

    return "\n".join(lines) + "\n"


def describe_rows(values, unit="words"):
    return [
        (key.capitalize(), f"{value:.2f} {unit}")
        for key, value in values.items()
    ]


# Data preparation

# Keep query IDs in a consistent order.
dev_query_ids = list(
    dev_queries.keys() if isinstance(dev_queries, dict) else dev_queries
)

query_texts = list(queries.values())
corpus_texts = [doc["text"] for doc in corpus.values()]

query_lengths = [len(text.split()) for text in query_texts]
corpus_lengths = [len(text.split()) for text in corpus_texts]

dev_query_lengths = [
    len(queries[q].split()) for q in dev_query_ids
]

dev_rel = {
    q: relevant_ids(dev_qrels.get(q, {}))
    for q in dev_query_ids
}

relevant_counts = [
    len(dev_rel[q]) for q in dev_query_ids
]


# 1. Length statistics

query_stats = describe(query_lengths)
corpus_stats = describe(corpus_lengths)

total_queries = len(queries)
total_documents = len(corpus)

short_queries = sum(1 for n in query_lengths if n <= 5)
medium_queries = sum(1 for n in query_lengths if 6 <= n <= 15)
long_queries = sum(1 for n in query_lengths if n > 15)


# 2. Relevance statistics

queries_with_relevance = sum(
    1 for count in relevant_counts if count > 0
)

queries_with_one_relevant = sum(
    1 for count in relevant_counts if count == 1
)

relevant_ids_missing = sum(
    1 for ids in dev_rel.values()
    for doc_id in ids if doc_id not in corpus
)

if len(dev_query_lengths) > 1:
    rho, rho_p = spearmanr(dev_query_lengths, relevant_counts)
else:
    rho = np.nan
    rho_p = np.nan


# 3. Duplicate questions

corpus_exact = Counter(normalize(text) for text in corpus_texts)
corpus_loose = Counter(normalize_loose(text) for text in corpus_texts)

query_exact = Counter(normalize(text) for text in query_texts)
query_loose = Counter(normalize_loose(text) for text in query_texts)

duplicate_documents = total_documents - len(corpus_exact)
duplicate_documents_loose = total_documents - len(corpus_loose)

duplicate_queries = total_queries - len(query_exact)

repeated_query_groups = sum(
    1 for count in query_exact.values() if count > 1
)

# Query text that also appears in the corpus.
queries_in_corpus = sum(
    1 for text in query_texts
    if normalize_loose(text) in corpus_loose
)


# 4. Vocabulary statistics

query_vocab = set()
corpus_vocab = set()

for text in query_texts:
    query_vocab.update(tokenize(text))

for text in corpus_texts:
    corpus_vocab.update(tokenize(text))

shared_vocab = query_vocab & corpus_vocab

missing_percentages = []
empty_token_queries = 0

for text in query_texts:
    words = set(tokenize(text))

    if not words:
        empty_token_queries += 1
        continue

    missing = words - corpus_vocab

    missing_percentages.append(
        len(missing) / len(words) * 100
    )


# Query vs. its relevant documents.

best_overlap = []

for query_id in dev_query_ids:
    query_words = set(tokenize(queries[query_id]))

    relevant_documents = [
        doc_id for doc_id in dev_rel[query_id]
        if doc_id in corpus
    ]

    if not query_words or not relevant_documents:
        continue

    overlaps = []

    for doc_id in relevant_documents:
        document_words = set(tokenize(corpus[doc_id]["text"]))

        overlap = len(query_words & document_words) / len(query_words)
        overlaps.append(overlap)

    best_overlap.append(max(overlaps))

zero_overlap = sum(
    1 for overlap in best_overlap if overlap == 0
)


# 5. Organize statistics

stats = {
    "overview": {
        "corpus_questions": total_documents,
        "queries": total_queries,
        "dev_queries": len(dev_query_ids),
        "val_queries": len(val_queries)
    },

    "query_length": query_stats,

    "corpus_length": corpus_stats,

    "relevance_dev": {
        "with_relevant": queries_with_relevance,
        "without_relevant": len(relevant_counts) - queries_with_relevance,
        "exactly_one_relevant": queries_with_one_relevant,
        "mean_relevant": float(np.mean(relevant_counts)),
        "median_relevant": float(np.median(relevant_counts)),
        "max_relevant": max(relevant_counts, default=0),
        "relevant_ids_missing_from_corpus": relevant_ids_missing,
        "spearman_len_vs_relevant": float(rho)
    },

    "duplicates": {
        "duplicate_corpus_exact": duplicate_documents,
        "duplicate_corpus_loose": duplicate_documents_loose,
        "duplicate_queries": duplicate_queries,
        "repeated_query_groups": repeated_query_groups,
        "queries_also_in_corpus": queries_in_corpus
    },

    "vocabulary": {
        "query_types": len(query_vocab),
        "corpus_types": len(corpus_vocab),
        "shared_types": len(shared_vocab),
        "mean_pct_query_words_missing": (
            float(np.mean(missing_percentages))
            if missing_percentages else 0.0
        ),
        "queries_with_any_missing_word": sum(
            1 for value in missing_percentages if value > 0
        ),
        "queries_with_no_tokens": empty_token_queries,
        "mean_best_overlap_with_relevant": (
            float(np.mean(best_overlap)) if best_overlap else 0.0
        ),
        "queries_zero_overlap_with_relevant": zero_overlap
    }
}


# 6. Save JSON report

# Save the reports in results/.
REPORT_DIR = RESULTS_DIR

REPORT_DIR.mkdir(parents=True, exist_ok=True)

(REPORT_DIR / "eda_statistics.json").write_text(
    json.dumps(stats, indent=2, default=lambda x: x.item()),
    encoding="utf-8"
)


# 7. Create Markdown report

overview = stats["overview"]
relevance = stats["relevance_dev"]
duplicates = stats["duplicates"]
vocabulary = stats["vocabulary"]

n_dev = len(relevant_counts)

report = "# Dataset Exploratory Data Analysis\n\n"

report += md_table("1. Dataset Overview", [
    ("Corpus questions", f"{overview['corpus_questions']:,}"),
    ("Queries", f"{overview['queries']:,}"),
    ("Development queries", f"{overview['dev_queries']:,}"),
    ("Validation queries", f"{overview['val_queries']:,}")
])

report += md_table(
    "2. Query Length",
    describe_rows(query_stats)
)

report += md_table(
    "3. Corpus Question Length",
    describe_rows(corpus_stats)
)

report += md_table("4. Query Length Categories", [
    ("Short (1-5 words)",
     f"{short_queries:,} ({percentage(short_queries, total_queries):.2f}%)"),
    ("Medium (6-15 words)",
     f"{medium_queries:,} ({percentage(medium_queries, total_queries):.2f}%)"),
    ("Long (16+ words)",
     f"{long_queries:,} ({percentage(long_queries, total_queries):.2f}%)")
], headers=("Category", "Queries"))

report += md_table("5. Relevance (Development Split)", [
    ("Queries with relevant documents",
     f"{relevance['with_relevant']:,} "
     f"({percentage(relevance['with_relevant'], n_dev):.2f}%)"),
    ("Queries with exactly one relevant document",
     f"{relevance['exactly_one_relevant']:,} "
     f"({percentage(relevance['exactly_one_relevant'], n_dev):.2f}%)"),
    ("Mean / median relevant documents",
     f"{relevance['mean_relevant']:.2f} / "
     f"{relevance['median_relevant']:.0f}"),
    ("Maximum relevant documents for one query",
     relevance["max_relevant"]),
    ("Relevant IDs missing from corpus",
     relevance["relevant_ids_missing_from_corpus"]),
    ("Spearman: query length vs. relevant count",
     f"{relevance['spearman_len_vs_relevant']:.3f}")
])

report += md_table("6. Vocabulary", [
    ("Unique query words", f"{vocabulary['query_types']:,}"),
    ("Unique corpus words", f"{vocabulary['corpus_types']:,}"),
    ("Query vocabulary found in corpus",
     f"{percentage(vocabulary['shared_types'], vocabulary['query_types']):.2f}%"),
    ("Queries with missing words",
     f"{vocabulary['queries_with_any_missing_word']:,}"),
    ("Queries with no tokens",
     vocabulary["queries_with_no_tokens"]),
    ("Mean best overlap with a relevant document",
     f"{vocabulary['mean_best_overlap_with_relevant']:.2%}"),
    ("Queries with zero overlap with relevant documents",
     f"{vocabulary['queries_zero_overlap_with_relevant']:,}")
], note="Tokenization uses lowercase words and preserves internal apostrophes. Overlap is the fraction of distinct query words found in a relevant document.")

report += md_table("7. Duplicate Questions", [
    ("Duplicate corpus entries (exact)",
     f"{duplicates['duplicate_corpus_exact']:,} "
     f"({percentage(duplicates['duplicate_corpus_exact'], total_documents):.2f}%)"),
    ("Duplicate corpus entries (ignoring punctuation)",
     f"{duplicates['duplicate_corpus_loose']:,} "
     f"({percentage(duplicates['duplicate_corpus_loose'], total_documents):.2f}%)"),
    ("Repeated query entries",
     f"{duplicates['duplicate_queries']:,} "
     f"({percentage(duplicates['duplicate_queries'], total_queries):.2f}%)"),
    ("Distinct repeated query groups",
     f"{duplicates['repeated_query_groups']:,}"),
    ("Queries also appearing in corpus",
     f"{duplicates['queries_also_in_corpus']:,} "
     f"({percentage(duplicates['queries_also_in_corpus'], total_queries):.2f}%)")
])

(REPORT_DIR / "eda_statistics.md").write_text(
    report,
    encoding="utf-8"
)


# Graph settings

CAP_LEN = 40
CAP_REL = 20


def save(name):
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / name, dpi=300)
    plt.show()
    plt.close()


def length_hist(lengths, ylabel, title, fname):
    """Integer-aligned word-count histogram, clipped at CAP_LEN, with median and p90."""

    clipped = np.minimum(lengths, CAP_LEN)
    pct_capped = 100 * np.mean(np.array(lengths) >= CAP_LEN)

    plt.figure(figsize=(9, 5))

    plt.hist(
        clipped,
        bins=np.arange(0.5, CAP_LEN + 1.5, 1),
        edgecolor="black"
    )

    plt.axvline(
        np.median(lengths),
        linestyle="--",
        color="tab:orange",
        label=f"Median: {np.median(lengths):.0f} words"
    )

    plt.axvline(
        np.percentile(lengths, 90),
        linestyle=":",
        color="tab:red",
        label=f"90th pct: {np.percentile(lengths, 90):.0f} words"
    )

    plt.xlabel(
        f"Number of words ({CAP_LEN} = {CAP_LEN}+, {pct_capped:.2f}% of items)"
    )
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xlim(0, CAP_LEN)
    plt.legend()

    save(fname)


def annotated_bars(labels, values, ylabel, title, fname):
    """Bar chart on a log axis with exact counts and percentages."""

    total = sum(values)

    plt.figure(figsize=(8, 5))

    bars = plt.bar(labels, values, edgecolor="black")
    plt.yscale("log")

    for bar, v in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            max(v, 1) * 1.15,
            f"{v:,}\n({100 * v / total:.2f}%)",
            ha="center",
            va="bottom"
        )

    plt.ylim(0.5, max(values) * 8)
    plt.ylabel(ylabel + " (log scale)")
    plt.title(title)

    save(fname)


# Graph 1: query length

length_hist(
    query_lengths,
    "Number of queries",
    "Query Length Distribution",
    "query_length.png"
)


# Graph 2: relevant documents per query

rel = np.minimum(relevant_counts, CAP_REL)
pct_one = 100 * np.mean(np.array(relevant_counts) == 1)

plt.figure(figsize=(9, 5))

plt.hist(
    rel,
    bins=np.arange(-0.5, CAP_REL + 1.5, 1),
    edgecolor="black"
)

plt.yscale("log")

plt.xlabel(f"Relevant documents ({CAP_REL} = {CAP_REL}+)")
plt.ylabel("Number of queries (log scale)")
plt.title(f"Relevant Documents per Query ({pct_one:.1f}% have exactly 1)")
plt.xticks(range(0, CAP_REL + 1, 2))
plt.xlim(-0.5, CAP_REL + 0.5)

save("relevant_documents.png")


# Graph 3: query length vs. relevant docs

# Use the same query order for both variables.
query_lengths_dev = [
    len(queries[q].split()) for q in dev_query_ids
]

x = np.minimum(query_lengths_dev, CAP_LEN)
y = np.minimum(relevant_counts, CAP_REL)

rho, p = spearmanr(query_lengths_dev, relevant_counts)

plt.figure(figsize=(9, 5))

h = plt.hist2d(
    x,
    y,
    bins=[
        np.arange(-0.5, CAP_LEN + 1.5, 1),
        np.arange(-0.5, CAP_REL + 1.5, 1)
    ],
    norm=LogNorm(),
    cmap="viridis"
)

plt.colorbar(h[3], label="Number of queries (log scale)")

xs = np.arange(1, CAP_LEN + 1)

means = [
    y[x == v].mean() if (x == v).sum() >= 5 else np.nan
    for v in xs
]

plt.plot(
    xs,
    means,
    color="red",
    marker="o",
    markersize=3,
    label="Mean (n >= 5)"
)

plt.xlabel(f"Query length (words, {CAP_LEN} = {CAP_LEN}+)")
plt.ylabel(f"Relevant documents ({CAP_REL} = {CAP_REL}+)")
plt.title(f"Query Length vs. Relevant Documents (Spearman rho = {rho:.2f})")
plt.legend()

save("query_length_vs_relevance.png")


# Graph 4: corpus length

length_hist(
    corpus_lengths,
    "Number of documents",
    "Corpus Question Length Distribution",
    "corpus_length.png"
)


# Graph 5: query vs. relevant doc length

# Calculate one average relevant-document length per query.
per_query_q = []
per_query_d = []

for query_id in dev_query_ids:
    judgments = dev_qrels.get(query_id, {})

    if isinstance(judgments, dict):
        rel_ids = [
            d for d, s in judgments.items()
            if float(s) > 0
        ]
    else:
        rel_ids = judgments

    doc_lens = [
        len(corpus[d]["text"].split())
        for d in rel_ids if d in corpus
    ]

    if doc_lens:
        per_query_q.append(len(queries[query_id].split()))
        per_query_d.append(np.mean(doc_lens))

plt.figure(figsize=(9, 5))

plt.boxplot(
    [per_query_q, per_query_d],
    tick_labels=[
        "Queries",
        "Relevant documents\n(mean per query)"
    ],
    showfliers=False
)

plt.ylabel("Number of words")
plt.title(
    "Query vs. Relevant Document Lengths "
    "(one value per query, outliers hidden)"
)

save("query_vs_relevant_length.png")


# Graph 6: duplicate vs. unique corpus questions

unique_documents = len(corpus_exact)
duplicate_entries = total_documents - unique_documents

annotated_bars(
    ["Unique Questions", "Duplicate Entries"],
    [unique_documents, duplicate_entries],
    "Number of questions",
    "Duplicate vs. Unique Corpus Questions",
    "corpus_duplicates.png"
)


# Graph 7: missing query vocabulary

missing = np.array(missing_percentages)

pct_zero = 100 * np.mean(missing == 0) if len(missing) else 0

plt.figure(figsize=(9, 5))

plt.hist(
    missing,
    bins=np.arange(0, 105, 5),
    edgecolor="black"
)

plt.yscale("log")

plt.xlabel("Percentage of query words missing from corpus")
plt.ylabel("Number of queries (log scale)")
plt.title(
    f"Query Vocabulary Missing from Corpus "
    f"({pct_zero:.1f}% of queries have none missing)"
)

plt.xlim(0, 100)

save("missing_query_words.png")


# Graph 8: unique vs. repeated queries

single_queries = sum(
    1 for count in query_exact.values() if count == 1
)

repeated_query_entries = total_queries - single_queries

annotated_bars(
    ["Appearing Once", "Repeated Entries"],
    [single_queries, repeated_query_entries],
    "Number of query entries",
    "Unique vs. Repeated Queries",
    "query_duplicates.png"
)