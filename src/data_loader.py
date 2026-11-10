from pathlib import Path

from beir import util
from beir.datasets.data_loader import GenericDataLoader


# Dataset location
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "data" / "quora"

# Dataset download link
URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/quora.zip"


def load_dataset(split="dev"):

    # Download the dataset if it does not exist
    if not DATASET_DIR.exists():
        util.download_and_unzip(URL, str(PROJECT_DIR))

    # Load the requested split
    corpus, queries, qrels = GenericDataLoader(
        data_folder=str(DATASET_DIR)
    ).load(split=split)

    return corpus, queries, qrels


if __name__ == "__main__":

    corpus, queries, qrels = load_dataset("dev")

    print("Corpus:", len(corpus))
    print("Queries:", len(queries))
    print("Qrels:", len(qrels))

    print("\nExample question:")
    print(next(iter(corpus.values()))["text"])