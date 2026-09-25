"""Reading the ground truth and writing the two required submission files."""
import os

import pandas as pd


def load_ground_truth(path):
    """
    train_ground_truth.tsv -> dict s1_entity_id -> set of matched ids.
    An empty matched_entity_ids cell means a singleton, which is a real label,
    not a missing value.
    """
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    ground_truth = {}
    for s1_id, matched in zip(df["source1_entity_id"], df["matched_entity_ids"]):
        ids = {part.strip() for part in str(matched).split(",") if part.strip()}
        ground_truth[str(s1_id).strip()] = ids
    return ground_truth


def _write_id_lists(path, s1_ids, id_lists, id_column):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    rows = [
        {"source1_entity_id": s1_id, id_column: ",".join(sorted(id_lists.get(s1_id, ())))}
        for s1_id in s1_ids
    ]
    pd.DataFrame(rows, columns=["source1_entity_id", id_column]).to_csv(
        path, sep="\t", index=False
    )


def write_matching_results(path, s1_ids, predictions):
    """One row per Source 1 entity, in the order given by `s1_ids`."""
    _write_id_lists(path, s1_ids, predictions, "matched_entity_ids")


def write_candidate_pairs(path, s1_ids, candidates):
    """The final candidate set the model scored, as the challenge requires."""
    lists = {s1_id: set(matches.keys()) if isinstance(matches, dict) else set(matches)
             for s1_id, matches in candidates.items()}
    _write_id_lists(path, s1_ids, lists, "candidate_entity_ids")
