# COMP6721_ProjectNLP4_Quora_DuplicateQ_Retrieval
COMP6721 F26 Applied AI Group Project on NLP4- Duplicate Question Retrieval on Quora, by Blen Abate (40193898), Anusha Gairola (40347249), Debosmita Chakraborty (40328670) 

## Setup
```bash
pip install -r requirements.txt
cd src
```
The BEIR Quora dataset is downloaded automatically to `data/quora` on first run.
A GPU (e.g. Colab) is strongly recommended for `dense_retrieval.py` and `rerank.py`.

## Pipeline (run from `src/`, in this order)
| Step | Command | Output |
|---|---|---|
| EDA | `python eda.py` | `results/eda_statistics.*`, figures |
| TF-IDF / BM25 baselines | `python baseline_tfidf.py`, `python baseline_bm25.py` | `results/tfidf_results.txt`, `results/bm25_results.txt` |
| Dense retrieval (MiniLM, MPNet) | `python dense_retrieval.py` | `results/dense_results_*.json`, `results/runs/dense_*_top50.json` |
| Cross-encoder reranking + depth study | `python rerank.py` | `results/rerank_results_*.json`, `results/figures/rerank_depth_*.png` |
| Final tables, figures, failure analysis | `python final_analysis.py` | `results/results_table_*.md/.csv`, `breakdown_*.md`, `failure_analysis_*.md`, `qualitative_examples_*.md` |

## Evaluation protocol
- Development uses the BEIR `dev` queries, split 80/20 into dev (4000, tuning and model selection) and val (1000, reporting) with `random_state=42`.
- The BEIR `test` queries are used once at the end: run `dense_retrieval.py --split test`, then `rerank.py --split test`, then `final_analysis.py --split test`.
- A query never retrieves itself.

## Limitations
No model is fine-tuned. The pretrained models may have seen Quora duplicate pairs during their own training, so dense and reranked scores may be optimistic.