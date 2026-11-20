# Final results (val queries, n = 1000)

| Method | R@1 | R@5 | R@10 | R@20 | R@50 | MRR@10 | nDCG@10 | Latency (ms/query) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF | 0.5567 | 0.7529 | 0.8172 | 0.8696 | 0.9188 | 0.7029 | 0.7112 | 12.8 |
| BM25 | 0.5761 | 0.7856 | 0.8428 | 0.8834 | 0.9246 | 0.7333 | 0.7390 | 2.7 |
| MiniLM | **0.6834** | **0.9052** | **0.9450** | 0.9722 | **0.9888** | **0.8507** | **0.8620** | 8.0 |
| MPNet | 0.6773 | 0.8983 | 0.9436 | **0.9745** | 0.9887 | 0.8452 | 0.8564 | 22.2 |
| MPNet+CE@20 | 0.6455 | 0.8929 | 0.9437 | **0.9745** | 0.9887 | 0.8233 | 0.8399 | 33.4 |
| MPNet+CE@50 | 0.6324 | 0.8822 | 0.9319 | 0.9689 | 0.9887 | 0.8103 | 0.8263 | 50.6 |

Bold = best in column. TF-IDF and BM25 latency is the average over a batch of queries; dense latency is single-query (encode + search); reranked latency adds the cross-encoder time for one query to the first-stage latency.
All methods use the same corpus and queries, the same K values and the same metric code, and a query never retrieves itself.
