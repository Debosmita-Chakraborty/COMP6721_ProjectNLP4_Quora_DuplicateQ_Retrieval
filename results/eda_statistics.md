# Dataset Exploratory Data Analysis

## 1. Dataset Overview

| Statistic | Value |
|---|---:|
| Corpus questions | 522,931 |
| Queries | 5,000 |
| Development queries | 4,000 |
| Validation queries | 1,000 |
## 2. Query Length

| Statistic | Value |
|---|---:|
| Mean | 9.56 words |
| Std | 4.19 words |
| Min | 3.00 words |
| P25 | 7.00 words |
| Median | 9.00 words |
| P75 | 11.00 words |
| P90 | 15.00 words |
| P95 | 18.00 words |
| P99 | 24.00 words |
| Max | 58.00 words |
## 3. Corpus Question Length

| Statistic | Value |
|---|---:|
| Mean | 11.46 words |
| Std | 6.28 words |
| Min | 1.00 words |
| P25 | 7.00 words |
| Median | 10.00 words |
| P75 | 14.00 words |
| P90 | 19.00 words |
| P95 | 24.00 words |
| P99 | 33.00 words |
| Max | 237.00 words |
## 4. Query Length Categories

| Category | Queries |
|---|---:|
| Short (1-5 words) | 488 (9.76%) |
| Medium (6-15 words) | 4,085 (81.70%) |
| Long (16+ words) | 427 (8.54%) |
## 5. Relevance (Development Split)

| Statistic | Value |
|---|---:|
| Queries with relevant documents | 4,000 (100.00%) |
| Queries with exactly one relevant document | 3,083 (77.08%) |
| Mean / median relevant documents | 1.54 / 1 |
| Maximum relevant documents for one query | 84 |
| Relevant IDs missing from corpus | 0 |
| Spearman: query length vs. relevant count | 0.007 |
## 6. Vocabulary

*Tokenization uses lowercase words and preserves internal apostrophes. Overlap is the fraction of distinct query words found in a relevant document.*

| Statistic | Value |
|---|---:|
| Unique query words | 7,158 |
| Unique corpus words | 89,574 |
| Query vocabulary found in corpus | 98.59% |
| Queries with missing words | 95 |
| Queries with no tokens | 0 |
| Mean best overlap with a relevant document | 69.66% |
| Queries with zero overlap with relevant documents | 0 |
## 7. Duplicate Questions

| Statistic | Value |
|---|---:|
| Duplicate corpus entries (exact) | 730 (0.14%) |
| Duplicate corpus entries (ignoring punctuation) | 1,610 (0.31%) |
| Repeated query entries | 0 (0.00%) |
| Distinct repeated query groups | 0 |
| Queries also appearing in corpus | 78 (1.56%) |
