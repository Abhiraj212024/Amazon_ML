"""
On-disk cache for the featurise-and-score stage.

Blocking was already cached, but featurising and scoring still cost several
minutes per run, which is the loop you pay every time you tune the decision
layer. Since the decision layer is where most of the loss sits, that loop has
to be cheap: with scores cached, a full threshold sweep is milliseconds.

The key covers the candidate set, the record contents and the model
configuration, so it invalidates itself whenever any of them change.
"""
import hashlib
import json
import logging
import os
import pickle

import numpy as np

logger = logging.getLogger(__name__)


def fingerprint(candidates, extra=None):
    """
    Stable id for a candidate set plus whatever else the scores depend on.

    Hashes the (entity, candidate) pairs rather than object identity, so a
    cache written by one process is reusable by the next.
    """
    hasher = hashlib.sha256()
    for s1_id in sorted(candidates):
        hasher.update(str(s1_id).encode())
        hasher.update(b"\x00")
        for cand_id in sorted(candidates[s1_id]):
            hasher.update(str(cand_id).encode())
            hasher.update(b"\x01")
    hasher.update(json.dumps(extra or {}, sort_keys=True, default=str).encode())
    return hasher.hexdigest()[:16]


def load(cache_dir, tag, key):
    if not cache_dir:
        return None
    path = os.path.join(cache_dir, f"{tag}_scores_{key}.pkl")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
    except (pickle.UnpicklingError, EOFError, AttributeError) as error:
        # a truncated or stale cache must not take the run down with it
        logger.warning("ignoring unreadable score cache %s (%s)", path, error)
        return None
    logger.info("loaded cached scores from %s", path)
    return payload


def save(cache_dir, tag, key, payload):
    if not cache_dir:
        return
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{tag}_scores_{key}.pkl")
    tmp = path + ".tmp"
    with open(tmp, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)  # atomic: an interrupted write leaves no half cache
    logger.info("cached scores to %s", path)


def calibration_report(probs, labels, n_bins=10):
    """
    How far the calibrated probabilities are from observed frequencies.

    `expected_f05` treats a score as a real probability when it weighs adding a
    candidate, so a miscalibrated model silently distorts every accept/reject
    decision. A large gap between `mean_predicted` and `observed_rate` in the
    high bins is the signal that it is abstaining for the wrong reason.
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if len(probs) == 0:
        return {}

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (probs >= lo) & (probs < hi if hi < 1.0 else probs <= hi)
        if not mask.any():
            continue
        bins.append({
            "range": [round(float(lo), 2), round(float(hi), 2)],
            "n_pairs": int(mask.sum()),
            "mean_predicted": float(probs[mask].mean()),
            "observed_rate": float(labels[mask].mean()),
        })

    return {
        "brier_score": float(np.mean((probs - labels) ** 2)),
        "mean_predicted": float(probs.mean()),
        "observed_rate": float(labels.mean()),
        "bins": bins,
    }
