"""
Official challenge metric plus the per-stage diagnostics used to debug the pipeline.

The scored metric is a macro-average of a per-Source-1-entity F-beta (beta=0.5).
Expanding the F-beta definition for a predicted set of size m, a true set of
size n and c correct predictions gives a closed form that the whole pipeline
relies on:

    F0.5 = 1.25 * p * r / (0.25 * p + r) = 1.25 * c / (m + 0.25 * n)

Edge cases (singletons) are defined by the problem statement rather than the
formula: an entity with no true matches scores 1.0 for an empty prediction and
0.0 for any non-empty one.
"""
import numpy as np


def f05_single(predicted, truth):
    """F0.5 for one Source 1 entity. `predicted` and `truth` are sets of ids."""
    m, n = len(predicted), len(truth)

    if n == 0:
        return 1.0 if m == 0 else 0.0
    if m == 0:
        return 0.0

    c = len(predicted & truth)
    if c == 0:
        return 0.0
    return 1.25 * c / (m + 0.25 * n)


def macro_f05(predictions, ground_truth):
    """
    Macro-average of f05_single over every Source 1 entity in `ground_truth`.

    Args:
        predictions: dict s1_id -> set of predicted entity_ids.
        ground_truth: dict s1_id -> set of true entity_ids. Defines the
            evaluation set: entities missing from `predictions` score as empty.

    Returns:
        (score, per_entity) where per_entity is dict s1_id -> float.
    """
    per_entity = {
        s1_id: f05_single(predictions.get(s1_id, set()), truth)
        for s1_id, truth in ground_truth.items()
    }
    score = float(np.mean(list(per_entity.values()))) if per_entity else 0.0
    return score, per_entity


def all_empty_baseline(ground_truth):
    """
    Score of predicting an empty list for every entity, i.e. the singleton
    fraction. This is the floor any real model has to beat.
    """
    if not ground_truth:
        return 0.0
    return float(np.mean([1.0 if not t else 0.0 for t in ground_truth.values()]))


def blocking_report(candidates, ground_truth, n_pool):
    """
    Quality of the candidate-generation stage, which caps recall for everything
    downstream. The challenge organisers audit exactly these numbers.

    Args:
        candidates: dict s1_id -> set/list of candidate entity_ids.
        ground_truth: dict s1_id -> set of true entity_ids.
        n_pool: size of the Source 2 + Source 3 pool (for the reduction ratio).
    """
    n_true = n_found = 0
    sizes = []
    entities_fully_covered = 0
    n_with_truth = 0

    for s1_id, truth in ground_truth.items():
        cand = set(candidates.get(s1_id, ()))
        sizes.append(len(cand))
        n_true += len(truth)
        found = len(cand & truth)
        n_found += found
        if truth:
            n_with_truth += 1
            if found == len(truth):
                entities_fully_covered += 1

    sizes = np.array(sizes) if sizes else np.array([0])
    total_pairs = len(ground_truth) * max(n_pool, 1)

    return {
        "pair_recall": n_found / n_true if n_true else 1.0,
        "entities_fully_covered": entities_fully_covered / n_with_truth if n_with_truth else 1.0,
        "candidates_per_entity_mean": float(sizes.mean()),
        "candidates_per_entity_p50": float(np.percentile(sizes, 50)),
        "candidates_per_entity_p95": float(np.percentile(sizes, 95)),
        "candidates_per_entity_max": int(sizes.max()),
        "reduction_ratio": 1.0 - (float(sizes.sum()) / total_pairs),
        "n_candidate_pairs": int(sizes.sum()),
    }


def error_attribution(predictions, candidates, ground_truth):
    """
    Assign every lost point to exactly one stage so it is obvious where to work
    next. Counted over true matches (recall side) plus false merges.
    """
    buckets = {
        "recovered": 0,          # true match predicted
        "lost_in_blocking": 0,   # true match never became a candidate
        "lost_in_decision": 0,   # was a candidate, model/decision dropped it
        "false_merge": 0,        # predicted id that is not a true match
        "singleton_correct": 0,
        "singleton_broken": 0,   # true singleton given a non-empty prediction
    }

    for s1_id, truth in ground_truth.items():
        pred = predictions.get(s1_id, set())
        cand = set(candidates.get(s1_id, ()))

        if not truth:
            buckets["singleton_correct" if not pred else "singleton_broken"] += 1
            buckets["false_merge"] += len(pred)
            continue

        for t in truth:
            if t in pred:
                buckets["recovered"] += 1
            elif t in cand:
                buckets["lost_in_decision"] += 1
            else:
                buckets["lost_in_blocking"] += 1
        buckets["false_merge"] += len(pred - truth)

    return buckets


def slice_report(per_entity, ground_truth, country_of):
    """
    Macro F0.5 sliced by country and by true-match count, which is where
    systematic weaknesses (e.g. an unseen country) actually show up.
    """
    by_country, by_size = {}, {}

    for s1_id, score in per_entity.items():
        country = country_of.get(s1_id) or "UNKNOWN"
        by_country.setdefault(country, []).append(score)

        n = len(ground_truth.get(s1_id, ()))
        bucket = "0 (singleton)" if n == 0 else ("1" if n == 1 else ("2" if n == 2 else "3+"))
        by_size.setdefault(bucket, []).append(score)

    summarise = lambda d: {
        k: {"macro_f05": float(np.mean(v)), "n_entities": len(v)}
        for k, v in sorted(d.items())
    }
    return {"by_country": summarise(by_country), "by_true_match_count": summarise(by_size)}
