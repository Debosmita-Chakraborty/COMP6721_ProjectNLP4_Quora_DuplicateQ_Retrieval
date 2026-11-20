# Final results (test queries, n = 10000)

| Method | R@1 | R@5 | R@10 | R@20 | R@50 | MRR@10 | nDCG@10 | Latency (ms/query) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF | 0.5571 | 0.7731 | 0.8324 | 0.8812 | 0.9269 | 0.7223 | 0.7283 | 12.7 |
| BM25 | 0.5758 | 0.7920 | 0.8502 | 0.8941 | 0.9322 | 0.7431 | 0.7491 | 2.7 |
| MiniLM | **0.6979** | 0.9103 | 0.9503 | **0.9741** | 0.9889 | **0.8677** | **0.8757** | 9.0 |
| MPNet | 0.6932 | **0.9112** | 0.9503 | 0.9741 | **0.9903** | 0.8654 | 0.8746 | 17.7 |
| MPNet+CE@20 | 0.6569 | 0.9055 | **0.9533** | 0.9741 | **0.9903** | 0.8353 | 0.8547 | 30.6 |
| MPNet+CE@50 | 0.6457 | 0.8933 | 0.9461 | 0.9731 | **0.9903** | 0.8227 | 0.8426 | 41.4 |

Bold = best in column. TF-IDF and BM25 latency is the average over a batch of queries; dense latency is single-query (encode + search); reranked latency adds the cross-encoder time for one query to the first-stage latency.
All methods use the same corpus and queries, the same K values and the same metric code, and a query never retrieves itself.
