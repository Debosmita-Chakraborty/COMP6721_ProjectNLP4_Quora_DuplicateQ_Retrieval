# MRR@10 by query characteristics (test queries)

## By query length

| Group | Queries | TF-IDF | BM25 | MiniLM | MPNet | MPNet+CE@20 | MPNet+CE@50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| short (<=7 words) | 3188 | 0.6992 | 0.7143 | 0.8738 | 0.8686 | 0.8375 | 0.8242 |
| medium (8-11) | 4494 | 0.7285 | 0.7436 | 0.8711 | 0.8681 | 0.8258 | 0.8126 |
| long (12+) | 2318 | 0.7423 | 0.7817 | 0.8527 | 0.8558 | 0.8507 | 0.8403 |

## By word overlap with relevant question

| Group | Queries | TF-IDF | BM25 | MiniLM | MPNet | MPNet+CE@20 | MPNet+CE@50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| low overlap (<0.30) | 1489 | 0.4316 | 0.4610 | 0.7467 | 0.7485 | 0.7004 | 0.6878 |
| medium (0.30-0.60) | 4707 | 0.6645 | 0.6984 | 0.8405 | 0.8376 | 0.8147 | 0.7999 |
| high (>=0.60) | 3804 | 0.9077 | 0.9088 | 0.9487 | 0.9455 | 0.9135 | 0.9038 |

Word overlap = mean Jaccard similarity between the query's words and each of its relevant questions (lower-cased words).
