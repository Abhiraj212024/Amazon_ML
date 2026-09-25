"""
Validation splits.

Two rules keep the estimate honest:

  * split by Source 1 entity, never by pair, so no entity's candidates straddle
    train and validation;
  * keep the *entire* Source 2/3 pool available to blocking for validation
    entities. Subsampling the pool thins out the distractors and inflates
    precision, which is exactly the number this metric is most sensitive to.

`leave_one_country_out` is the only available proxy for the unseen country in
the test set (France): train on one country, validate on another, and see how
far the thresholds transfer.
"""
import numpy as np


def _stratum(s1_id, ground_truth, country_of):
    n = len(ground_truth.get(s1_id, ()))
    size_bucket = "0" if n == 0 else ("1" if n == 1 else ("2" if n == 2 else "3+"))
    return f"{country_of.get(s1_id) or 'UNKNOWN'}|{size_bucket}"


def holdout_split(s1_ids, ground_truth, country_of, val_fraction=0.2, seed=42):
    """
    Stratified holdout on (country, number of true matches) so the validation
    slice carries the same singleton rate as the training slice.

    Returns (train_ids, val_ids).
    """
    rng = np.random.default_rng(seed)

    strata = {}
    for s1_id in s1_ids:
        strata.setdefault(_stratum(s1_id, ground_truth, country_of), []).append(s1_id)

    train, val = [], []
    for _, members in sorted(strata.items()):
        members = list(rng.permutation(np.array(members, dtype=object)))
        n_val = int(round(val_fraction * len(members)))
        # never let a stratum contribute to only one side when it can split
        if len(members) > 1:
            n_val = min(max(n_val, 1), len(members) - 1)
        val.extend(members[:n_val])
        train.extend(members[n_val:])

    return sorted(train), sorted(val)


def leave_one_country_out(s1_ids, country_of, holdout_country):
    """Train on every other country, validate on `holdout_country`."""
    train = [s for s in s1_ids if country_of.get(s) != holdout_country]
    val = [s for s in s1_ids if country_of.get(s) == holdout_country]
    return sorted(train), sorted(val)


def split_summary(train_ids, val_ids, ground_truth, country_of):
    def describe(ids):
        singletons = sum(1 for s in ids if not ground_truth.get(s))
        countries = {}
        for s in ids:
            countries[country_of.get(s) or "UNKNOWN"] = countries.get(country_of.get(s) or "UNKNOWN", 0) + 1
        return {
            "n_entities": len(ids),
            "singleton_rate": singletons / len(ids) if ids else 0.0,
            "countries": countries,
        }

    return {"train": describe(train_ids), "val": describe(val_ids)}
