from sklearn.model_selection import train_test_split


def split_dataset(queries, qrels):

    # Get all query IDs
    query_ids = list(queries.keys())

    # Split query IDs: 80% development, 20% validation
    dev_ids, val_ids = train_test_split(
        query_ids,
        test_size=0.2,
        random_state=42
    )

    # Create development and validation dictionaries
    dev_queries = {qid: queries[qid] for qid in dev_ids}
    val_queries = {qid: queries[qid] for qid in val_ids}

    dev_qrels = {qid: qrels[qid] for qid in dev_ids if qid in qrels}
    val_qrels = {qid: qrels[qid] for qid in val_ids if qid in qrels}

    return dev_queries, val_queries, dev_qrels, val_qrels