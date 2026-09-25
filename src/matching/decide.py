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

STRATEGIES = ("tiered", "expected_f05", "threshold", "top1")


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


def tiered_select(probs, t_first=0.35, t_rest=0.75, ratio=0.4, max_k=10):
    """
    Separate thresholds for the first accept and for the rest.

    Motivated by the break-even acceptance probability. For an entity holding m
    correct predictions out of n true matches, one more candidate is worth
    adding when its probability exceeds

        q* = m / (m + 0.25n)

    which at n=4 runs 0.00, 0.50, 0.67, 0.75, 0.80 for m = 0..4. A single
    threshold cannot express that: tuned for m=3 it throws away the first
    accept, where the bar should be far lower. Splitting the first accept from
    the rest captures most of the shape with one extra parameter, and unlike
    `expected_f05` it does not depend on the probabilities being well
    calibrated - only on their ordering.
    """
    probs = np.asarray(probs, dtype=float)
    if len(probs) == 0 or probs[0] < t_first:
        return 0
    top = probs[0]
    k = 1
    for p in probs[1:max_k]:
        if p >= t_rest and p >= ratio * top:
            k += 1
        else:
            break
    return k


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
        elif strategy == "tiered":
            k = tiered_select(probs, **kwargs)
        elif strategy == "threshold":
            k = threshold_select(probs, **kwargs)
        else:
            k = top1_select(probs, **kwargs)

        predictions[s1_id] = {cand_id for cand_id, _ in ranked[:k]}

    return predictions


def build_tuning_arrays(scored, ground_truth, max_k=10):
    """
    Pack the scored candidates into padded arrays so a grid point can be
    evaluated with numpy instead of a Python loop per entity.

    Returns (P, T, n_true) where, for each entity, P holds the top `max_k`
    probabilities descending (padded with -1), T marks which of those are true
    matches, and n_true is the entity's full number of true matches -- which
    counts matches beyond max_k and those blocking never retrieved, so the
    tuner optimises the real metric rather than a recall-capped version of it.
    """
    entity_ids = list(scored)
    P = np.full((len(entity_ids), max_k), -1.0, dtype=float)
    T = np.zeros((len(entity_ids), max_k), dtype=bool)
    n_true = np.zeros(len(entity_ids), dtype=float)

    for row, s1_id in enumerate(entity_ids):
        truth = ground_truth.get(s1_id, set())
        n_true[row] = len(truth)
        ranked = sorted(scored[s1_id], key=lambda item: -item[1])[:max_k]
        for col, (cand_id, prob) in enumerate(ranked):
            P[row, col] = prob
            T[row, col] = cand_id in truth

    return P, T, n_true


def _macro_f05_from_k(k, T, n_true):
    """Vectorised macro F0.5 for a per-entity prefix length `k`."""
    correct = np.cumsum(T, axis=1)
    # c for the chosen prefix; k=0 contributes nothing
    rows = np.arange(len(k))
    c = np.where(k > 0, correct[rows, np.clip(k - 1, 0, T.shape[1] - 1)], 0)

    with np.errstate(divide="ignore", invalid="ignore"):
        scores = np.where(
            n_true == 0,
            np.where(k == 0, 1.0, 0.0),
            np.where(k == 0, 0.0, 1.25 * c / np.maximum(k + 0.25 * n_true, 1e-9)),
        )
    return float(scores.mean())


def _tiered_k(P, t_first, t_rest, ratio):
    """Prefix length per entity under the tiered rule, vectorised."""
    has_candidate = P[:, 0] >= 0
    accept_first = has_candidate & (P[:, 0] >= t_first)

    if P.shape[1] == 1:
        return accept_first.astype(int)

    top = P[:, 0:1]
    eligible = (P[:, 1:] >= t_rest) & (P[:, 1:] >= ratio * top) & (P[:, 1:] >= 0)
    # only a leading run of accepts counts, matching the sequential rule
    run = np.cumprod(eligible, axis=1).sum(axis=1)
    return np.where(accept_first, 1 + run, 0).astype(int)


def _threshold_k(P, t_high, ratio):
    return _tiered_k(P, t_high, t_high, ratio)


def tune_tiered(scored, ground_truth, max_k=10, grid_first=None, grid_rest=None,
                grid_ratio=None):
    """
    Grid search the tiered strategy directly against macro F0.5.

    Vectorised, so a three-dimensional grid costs milliseconds per point and
    the ranges can be wide enough that the optimum is interior rather than
    stuck against an edge.
    """
    grid_first = grid_first if grid_first is not None else np.arange(0.05, 0.96, 0.05)
    grid_rest = grid_rest if grid_rest is not None else np.arange(0.05, 0.96, 0.05)
    grid_ratio = grid_ratio if grid_ratio is not None else np.arange(0.05, 1.01, 0.05)

    P, T, n_true = build_tuning_arrays(scored, ground_truth, max_k)

    best = {"t_first": 0.35, "t_rest": 0.75, "ratio": 0.4, "macro_f05": -1.0}
    optima = []
    for t_first in grid_first:
        for t_rest in grid_rest:
            for ratio in grid_ratio:
                score = _macro_f05_from_k(_tiered_k(P, t_first, t_rest, ratio), T, n_true)
                params = {
                    "t_first": float(t_first), "t_rest": float(t_rest),
                    "ratio": float(ratio),
                }
                if score > best["macro_f05"] + 1e-12:
                    best = dict(params, macro_f05=score)
                    optima = [params]
                elif abs(score - best["macro_f05"]) <= 1e-12:
                    optima.append(params)

    best["on_grid_edge"] = _edge_flags(
        optima, {"t_first": grid_first, "t_rest": grid_rest, "ratio": grid_ratio}
    )
    best["n_optimal_settings"] = len(optima)
    return best


def _edge_flags(optima, grids):
    """
    Name the parameters whose optimum is genuinely pinned to a grid edge.

    An optimum at an edge usually means the grid, not the data, chose it: the
    earlier threshold tuning picked ratio=0.4 only because the range stopped
    there. But a parameter can also sit at an edge because it has no effect -
    when `ratio` binds harder than `t_rest`, every value of `t_rest` scores
    identically and the first one wins arbitrarily. Flagging that would send
    you widening a range that changes nothing.

    So a parameter is flagged only when *every* combination achieving the best
    score puts it on an edge. If any equally-good combination places it inside
    the range, the range is not what is limiting it.

    Args:
        optima: parameter dicts that all achieve the best score.
        grids: {parameter name: the values searched}.
    """
    edges = []
    for key, grid in grids.items():
        lo, hi = float(min(grid)), float(max(grid))
        on_edge = [
            abs(o[key] - lo) < 1e-9 or abs(o[key] - hi) < 1e-9
            for o in optima if key in o
        ]
        if on_edge and all(on_edge):
            edges.append(key)
    return edges


def tune_threshold(scored, ground_truth, max_k=10, grid_t=None, grid_ratio=None):
    """
    Grid search the single-threshold strategy against macro F0.5.

    Tuning on the real metric rather than AUC or plain F1 matters: those pick a
    balanced operating point, and F0.5 wants a precision-heavy one.
    """
    grid_t = grid_t if grid_t is not None else np.arange(0.10, 0.96, 0.05)
    # the ratio range starts far lower than before: the previous grid began at
    # 0.4 and the optimum sat exactly there, so the range was the binding limit
    grid_ratio = grid_ratio if grid_ratio is not None else np.arange(0.05, 1.01, 0.05)

    P, T, n_true = build_tuning_arrays(scored, ground_truth, max_k)

    best = {"t_high": 0.5, "ratio": 0.6, "macro_f05": -1.0}
    optima = []
    for t in grid_t:
        for ratio in grid_ratio:
            score = _macro_f05_from_k(_threshold_k(P, t, ratio), T, n_true)
            params = {"t_high": float(t), "ratio": float(ratio)}
            if score > best["macro_f05"] + 1e-12:
                best = dict(params, macro_f05=score)
                optima = [params]
            elif abs(score - best["macro_f05"]) <= 1e-12:
                optima.append(params)

    best["on_grid_edge"] = _edge_flags(
        optima, {"t_high": grid_t, "ratio": grid_ratio}
    )
    best["n_optimal_settings"] = len(optima)
    return best
