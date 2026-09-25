"""
Stage B scorer: a gradient-boosted pairwise classifier with calibrated output.

Calibration matters more than usual here. The decision layer treats the score as
a real probability when it computes the expected F0.5 of a candidate set, so an
uncalibrated margin would silently distort every accept/reject choice.

Training negatives are the pipeline's own blocking output rather than random
pairs, so the model learns the boundary it will actually be asked about.
"""
import logging

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression

from .pair_features import FEATURE_NAMES

logger = logging.getLogger(__name__)

try:  # optional: faster and usually a little stronger, but not required
    import lightgbm as lgb
    _HAS_LIGHTGBM = True
except ImportError:  # pragma: no cover - depends on the environment
    _HAS_LIGHTGBM = False


class PairwiseMatcher:
    """Fit on candidate pairs, predict a calibrated P(match) per pair."""

    def __init__(self, random_state=42, calibration_fraction=0.25, **model_kwargs):
        self.random_state = random_state
        self.calibration_fraction = calibration_fraction
        self.model_kwargs = model_kwargs
        self.model = None
        self.calibrator = None
        self.feature_names = list(FEATURE_NAMES)

    def _new_model(self):
        if _HAS_LIGHTGBM:
            params = dict(
                n_estimators=400, learning_rate=0.05, num_leaves=31,
                min_child_samples=20, subsample=0.9, subsample_freq=1,
                colsample_bytree=0.9, random_state=self.random_state, verbosity=-1,
            )
            params.update(self.model_kwargs)
            return lgb.LGBMClassifier(**params)

        params = dict(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
            min_samples_leaf=20, l2_regularization=1.0,
            random_state=self.random_state,
        )
        params.update(self.model_kwargs)
        return HistGradientBoostingClassifier(**params)

    def fit(self, X, y, groups):
        """
        Args:
            X: (n_pairs, n_features) float array.
            y: binary labels.
            groups: per-pair Source 1 entity id. Used to split fit/calibration
                by entity so no entity's pairs straddle both sides.
        """
        y = np.asarray(y).astype(int)
        groups = np.asarray(groups, dtype=object)

        unique_groups = np.unique(groups)
        rng = np.random.default_rng(self.random_state)
        shuffled = rng.permutation(unique_groups)
        n_cal = max(1, int(self.calibration_fraction * len(shuffled)))
        cal_groups = set(shuffled[:n_cal].tolist())

        is_cal = np.array([g in cal_groups for g in groups])
        # if the calibration slice has only one class, calibrate on everything
        if is_cal.sum() == 0 or len(np.unique(y[is_cal])) < 2:
            is_cal = np.ones(len(y), dtype=bool)
        is_fit = ~is_cal if (~is_cal).sum() > 0 and len(np.unique(y[~is_cal])) >= 2 else np.ones(len(y), dtype=bool)

        self.model = self._new_model()
        self.model.fit(X[is_fit], y[is_fit])
        logger.info(
            "pairwise model fit on %d pairs (%d positive), calibrated on %d",
            int(is_fit.sum()), int(y[is_fit].sum()), int(is_cal.sum()),
        )

        raw = self.model.predict_proba(X[is_cal])[:, 1]
        self.calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self.calibrator.fit(raw, y[is_cal])
        return self

    def predict_proba(self, X):
        if self.model is None:
            raise RuntimeError("PairwiseMatcher.fit must be called before predict_proba")
        if len(X) == 0:
            return np.zeros(0, dtype=float)
        raw = self.model.predict_proba(X)[:, 1]
        return self.calibrator.predict(raw) if self.calibrator is not None else raw

    def feature_importance(self, X=None, y=None, n_repeats=3, max_rows=20000):
        """
        Feature name -> importance, most important first.

        LightGBM exposes split counts directly. HistGradientBoostingClassifier
        does not, so fall back to permutation importance over a sample, which
        needs X and y. Returns {} when neither route is available.
        """
        importances = getattr(self.model, "feature_importances_", None)
        if importances is None:
            if X is None or y is None or len(X) == 0:
                return {}
            from sklearn.inspection import permutation_importance

            rng = np.random.default_rng(self.random_state)
            if len(X) > max_rows:
                rows = rng.choice(len(X), size=max_rows, replace=False)
                X, y = X[rows], np.asarray(y)[rows]
            result = permutation_importance(
                self.model, X, y, n_repeats=n_repeats,
                random_state=self.random_state, scoring="average_precision",
            )
            importances = result.importances_mean

        pairs = sorted(zip(self.feature_names, importances), key=lambda kv: -kv[1])
        return {name: float(value) for name, value in pairs}


def label_pairs(pair_index, ground_truth):
    """1 when the candidate is a true match for its Source 1 entity, else 0."""
    s1 = pair_index["s1_entity_id"].to_numpy(dtype=object)
    cand = pair_index["candidate_entity_id"].to_numpy(dtype=object)
    return np.array(
        [1 if c in ground_truth.get(s, ()) else 0 for s, c in zip(s1, cand)],
        dtype=int,
    )
