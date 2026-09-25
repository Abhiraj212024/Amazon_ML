"""
Stage C: global conflict resolution.

Source 1 is the deduplicated reference source, which implies each Source 2 /
Source 3 record describes at most one Source 1 entity. Note the direction: a
single Source 1 entity may legitimately collect many Source 2/3 records that
word the same business differently - that is the one-to-many the task asks for.
The constraint is the other way round, and only fires when two *different*
Source 1 entities both claim the same record. There, at most one can be right,
so keeping only the higher-scoring claim is a free precision gain.

`check_one_to_one` verifies the assumption against the training ground truth
before any of it is applied.
"""
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


def check_one_to_one(ground_truth):
    """
    Does each Source 2/3 record appear under at most one Source 1 entity?

    Returns a report with the violation rate and a few examples. Resolution
    should only be enabled when `violation_rate` is ~0.
    """
    owners = defaultdict(list)
    for s1_id, matches in ground_truth.items():
        for cand_id in matches:
            owners[cand_id].append(s1_id)

    shared = {cand: s1s for cand, s1s in owners.items() if len(s1s) > 1}
    return {
        "n_matched_records": len(owners),
        "n_shared_records": len(shared),
        "violation_rate": len(shared) / len(owners) if owners else 0.0,
        "examples": [
            {"candidate_entity_id": cand, "claimed_by": s1s}
            for cand, s1s in list(shared.items())[:5]
        ],
        "holds": len(shared) == 0,
    }


def resolve_scored_pairs(scored, margin=0.0):
    """
    Pre-decision variant: drop a candidate from every entity except its
    highest-scoring claimant, before the decision layer runs.

    Aggressive - the losing entity loses the candidate even if the winner later
    declines it. `margin` keeps a contested claim when the scores are close
    enough that the ordering is not trustworthy.

    Args:
        scored: dict s1_id -> list of (candidate_id, probability).

    Returns:
        (filtered_scored, report)
    """
    best = {}
    for s1_id, pairs in scored.items():
        for cand_id, prob in pairs:
            if cand_id not in best or prob > best[cand_id][1]:
                best[cand_id] = (s1_id, prob)

    filtered, n_dropped = {}, 0
    for s1_id, pairs in scored.items():
        kept = []
        for cand_id, prob in pairs:
            winner_id, winner_prob = best[cand_id]
            if winner_id == s1_id or prob >= winner_prob - margin:
                kept.append((cand_id, prob))
            else:
                n_dropped += 1
        filtered[s1_id] = kept

    total = sum(len(p) for p in scored.values())
    return filtered, {
        "stage": "pre",
        "pairs_in": total,
        "pairs_dropped": n_dropped,
        "drop_rate": n_dropped / total if total else 0.0,
    }


def resolve_predictions(predictions, scored):
    """
    Post-decision variant (the default): only intervene where two entities have
    actually *predicted* the same record, and give it to the higher-scoring one.

    Safer than the pre-decision variant, because it never removes a candidate on
    behalf of a winner that ends up not wanting it, and it leaves each entity's
    calibrated decision otherwise intact.

    Returns:
        (resolved_predictions, report)
    """
    prob_of = {
        (s1_id, cand_id): prob
        for s1_id, pairs in scored.items()
        for cand_id, prob in pairs
    }

    claimants = defaultdict(list)
    for s1_id, preds in predictions.items():
        for cand_id in preds:
            claimants[cand_id].append(s1_id)

    resolved = {s1_id: set(preds) for s1_id, preds in predictions.items()}
    n_conflicts = n_dropped = 0

    for cand_id, s1_ids in claimants.items():
        if len(s1_ids) < 2:
            continue
        n_conflicts += 1
        winner = max(s1_ids, key=lambda s: prob_of.get((s, cand_id), 0.0))
        for s1_id in s1_ids:
            if s1_id != winner:
                resolved[s1_id].discard(cand_id)
                n_dropped += 1

    total = sum(len(p) for p in predictions.values())
    return resolved, {
        "stage": "post",
        "predicted_ids": total,
        "contested_records": n_conflicts,
        "ids_dropped": n_dropped,
        "drop_rate": n_dropped / total if total else 0.0,
    }
