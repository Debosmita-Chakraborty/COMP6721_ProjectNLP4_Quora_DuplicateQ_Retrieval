# MRR@10 by query characteristics (val queries)

## By query length

| Group | Queries | TF-IDF | BM25 | MiniLM | MPNet | MPNet+CE@20 | MPNet+CE@50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| short (<=7 words) | 320 | 0.6820 | 0.6909 | 0.8509 | 0.8453 | 0.8070 | 0.7941 |
| medium (8-11) | 455 | 0.7101 | 0.7474 | 0.8519 | 0.8491 | 0.8311 | 0.8163 |
| long (12+) | 225 | 0.7183 | 0.7653 | 0.8482 | 0.8371 | 0.8307 | 0.8212 |

## By word overlap with relevant question

| Group | Queries | TF-IDF | BM25 | MiniLM | MPNet | MPNet+CE@20 | MPNet+CE@50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| low overlap (<0.30) | 155 | 0.4228 | 0.4927 | 0.7540 | 0.7442 | 0.6910 | 0.6687 |
| medium (0.30-0.60) | 470 | 0.6326 | 0.6607 | 0.8131 | 0.8011 | 0.7872 | 0.7739 |
| high (>=0.60) | 375 | 0.9069 | 0.9239 | 0.9379 | 0.9422 | 0.9232 | 0.9145 |

Word overlap = mean Jaccard similarity between the query's words and each of its relevant questions (lower-cased words).
