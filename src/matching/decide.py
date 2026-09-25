"""
Stage D: turn calibrated per-candidate probabilities into a predicted set.

This is where the metric is won or lost. Using F0.5 = 1.25c / (m + 0.25n):

    n=1, correct match alone        -> 1.000
    n=1, correct match + one wrong  -> 0.556
    n=0, empty prediction           -> 1.000
    n=0, any prediction             -> 0.000

So a single false positive on a single-match entity costs 0.44, while skipping a
third true match when you already hold one costs only 0.29. Precision on small
entities dominates; recall on large ones barely matters.

`expected_f05` exploits that directly: rather than applying one global threshold
to every entity, it estimates the expected F0.5 of each top-k prefix by Monte
Carlo over the calibrated probabilities and returns the best k. k=0 (predict a
singleton) falls out of the same computation, so no separate abstain rule is
needed. Two simpler strategies are kept for A/B comparison.
"""
import numpy as np

STRATEGIES = ("expected_f05", "threshold", "top1")


def expected_f05_select(probs, n_samples=256, max_k=10, rng=None):
    """
    Choose how many of the top-scoring candidates to predict.

    Args:
        probs: calibrated P(match), already sorted descending.
        n_samples: Monte Carlo draws. 256 is plenty - the decision is a argmax
            over ~10 options, not a precise value estimate.
        max_k: largest prefix considered.

    Returns:
        (k, expected_score_per_k)
    """
    probs = np.asarray(probs, dtype=float)
    m_all = len(probs)
    if m_all == 0:
        return 0, np.array([1.0])

    rng = rng or np.random.default_rng(0)
    k_max = min(max_k, m_all)

    # sample which candidates are genuinely true matches
    truth = rng.random((n_samples, m_all)) < probs[None, :]
    n_true = truth.sum(axis=1)                      # |T| per sample
    correct = np.cumsum(truth[:, :k_max], axis=1)   # c for each prefix size

    scores = np.empty(k_max + 1, dtype=float)
    # k = 0: scores 1.0 exactly when the entity really is a singleton
    scores[0] = float(np.mean(n_true == 0))

    for k in range(1, k_max + 1):
        c = correct[:, k - 1]
        f = np.where(n_true == 0, 0.0, 1.25 * c / (k + 0.25 * n_true))
        scores[k] = float(f.mean())

    return int(np.argmax(scores)), scores


def threshold_select(probs, t_high=0.5, ratio=0.6, max_k=10):
    """
    Simpler baseline: accept the top candidate above `t_high`, then accept
    further candidates only if they are both above `t_high` and within `ratio`
    of the top score. Useful as the control when judging `expected_f05`.
    """
    probs = np.asarray(probs, dtype=float)
    if len(probs) == 0 or probs[0] < t_high:
        return 0
    top = probs[0]
    k = 1
    for p in probs[1:max_k]:
        if p >= t_high and p >= ratio * top:
            k += 1
        else:
            break
    return k


def top1_select(probs, t_high=0.5):
    """Weakest baseline: at most one match per entity."""
    probs = np.asarray(probs, dtype=float)
    return 1 if len(probs) and probs[0] >= t_high else 0


def select_matches(scored, strategy="expected_f05", seed=0, **kwargs):
    """
    Apply a decision strategy to every entity.

    Args:
        scored: dict s1_id -> list of (candidate_id, probability), any order.
        strategy: one of STRATEGIES.

    Returns:
        dict s1_id -> set of predicted candidate ids.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}, expected one of {STRATEGIES}")

    rng = np.random.default_rng(seed)
    predictions = {}

    for s1_id, pairs in scored.items():
        ranked = sorted(pairs, key=lambda item: -item[1])
        probs = [p for _, p in ranked]

        if strategy == "expected_f05":
            k, _ = expected_f05_select(probs, rng=rng, **kwargs)
        elif strategy == "threshold":
            k = threshold_select(probs, **kwargs)
        else:
            k = top1_select(probs, **kwargs)

        predictions[s1_id] = {cand_id for cand_id, _ in ranked[:k]}

    return predictions


def tune_threshold(scored, ground_truth, grid_t=None, grid_ratio=None):
    """
    Grid search the `threshold` strategy directly against macro F0.5.

    Tuning on the real metric rather than AUC or plain F1 matters: those pick a
    balanced operating point, and F0.5 wants a precision-heavy one.
    """
    from .metrics import macro_f05

    grid_t = grid_t if grid_t is not None else np.arange(0.20, 0.96, 0.05)
    grid_ratio = grid_ratio if grid_ratio is not None else np.arange(0.4, 1.01, 0.1)

    best = {"t_high": 0.5, "ratio": 0.6, "macro_f05": -1.0}
    for t in grid_t:
        for ratio in grid_ratio:
            preds = select_matches(scored, "threshold", t_high=float(t), ratio=float(ratio))
            score, _ = macro_f05(preds, ground_truth)
            if score > best["macro_f05"]:
                best = {"t_high": float(t), "ratio": float(ratio), "macro_f05": score}
    return best
