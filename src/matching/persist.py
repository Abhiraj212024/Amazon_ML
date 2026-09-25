"""
Save a trained pipeline so inference can run as a separate, restartable job.

Running training and inference in one process is fine on a slice and
impossible at full scale: blocking the whole test set against the whole pool
takes far longer than a Kaggle session allows, and the feature matrix for
every candidate pair does not fit in memory at once. Splitting them lets
training run once on a sample, and inference run in shards that can be resumed.

The bundle carries everything inference needs that was derived from the
training data - the model, the IDF tables, the tuned decision parameters and
the blocking configuration - so predictions cannot silently drift from the
configuration they were tuned for.
"""
import logging
import os
import pickle

logger = logging.getLogger(__name__)

# Bumped whenever the bundle's contents change in a way that makes an older
# file unusable, so a stale bundle fails loudly instead of predicting nonsense.
BUNDLE_VERSION = 1


def save_bundle(path, matcher, name_idf, addr_idf, tuned, tuned_tiered,
                blocking_config, strategy, conflict_stage, max_k, feature_names,
                extra=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    payload = {
        "version": BUNDLE_VERSION,
        "matcher": matcher,
        "name_idf": name_idf,
        "addr_idf": addr_idf,
        "tuned_threshold": tuned,
        "tuned_tiered": tuned_tiered,
        # the encoder is a live object and is rebuilt at predict time from its
        # name, so it is deliberately not pickled here
        "blocking_config": {k: v for k, v in (blocking_config or {}).items()
                            if k != "embedding_encoder"},
        "strategy": strategy,
        "conflict_stage": conflict_stage,
        "max_k": max_k,
        "feature_names": list(feature_names),
        "extra": extra or {},
    }
    tmp = path + ".tmp"
    with open(tmp, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    logger.info("saved model bundle to %s (%.1f MB)", path, os.path.getsize(path) / 1e6)


def load_bundle(path, feature_names=None):
    with open(path, "rb") as handle:
        payload = pickle.load(handle)

    version = payload.get("version")
    if version != BUNDLE_VERSION:
        raise ValueError(
            f"model bundle {path} is version {version}, this code expects "
            f"{BUNDLE_VERSION}; retrain with the current code"
        )
    # a feature layout mismatch would score the wrong columns, silently
    if feature_names is not None and list(payload["feature_names"]) != list(feature_names):
        raise ValueError(
            f"model bundle {path} was trained on a different feature set "
            f"({len(payload['feature_names'])} features vs {len(feature_names)}); retrain it"
        )
    logger.info("loaded model bundle from %s", path)
    return payload
