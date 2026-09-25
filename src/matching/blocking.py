"""
Stage A: candidate generation.

Recall lost here is unrecoverable, so the strategy is a *union* of several
independent channels rather than a single similarity. Each channel is good at a
different noise pattern, and a pair only has to survive one of them:

    name_char     char 3-5 gram TF-IDF on the name   -> typos, suffix variants
    addr_char     char 3-5 gram TF-IDF on the address -> abbreviations, spacing
    name_word     word-level TF-IDF on the name       -> word-order transposition
    numeric       shared house / postal numbers       -> renamed businesses
    rare_token    shared high-IDF name tokens         -> distinctive long-tail names

Everything is blocked by country first, which is a near-free 100x reduction.
Records with a missing country are pooled into every country block so they are
never silently dropped.
"""
import hashlib
import json
import logging
import os
import pickle
import re
import time
from collections import Counter, defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

try:
    # Computes the top-n of a sparse product without ever densifying it.
    # Roughly 10x faster single-threaded than chunked densification, and it
    # parallelises, which the fallback path does not. Verified to return
    # bit-identical results to the fallback.
    from sparse_dot_topn import sp_matmul_topn
    _HAS_SPARSE_DOT_TOPN = True
except ImportError:  # pragma: no cover - depends on the environment
    _HAS_SPARSE_DOT_TOPN = False

CHANNELS = ("name_char", "addr_char", "name_word", "numeric", "rare_token", "embedding")


def _text(df, column):
    """Column as a list of plain strings, with missing values as ''."""
    if column not in df.columns:
        return [""] * len(df)
    return df[column].fillna("").astype(str).tolist()


def _sparse_topk(query, pool, k, min_score=0.0, max_dense_cells=4_000_000, n_threads=-1):
    """
    Top-k pool rows for each query row by cosine similarity.

    TF-IDF rows are L2-normalised, so the dot product is the cosine.

    Uses sparse_dot_topn when available, which keeps the product sparse and runs
    multi-threaded. The fallback densifies the product in row chunks sized to a
    memory budget, which is correct but an order of magnitude slower on a large
    block. Both paths yield the same pairs.

    Yields (query_index, pool_index, score).
    """
    n_pool = pool.shape[0]
    if n_pool == 0 or query.shape[0] == 0:
        return
    k = min(k, n_pool)

    if _HAS_SPARSE_DOT_TOPN:
        # pushing the threshold down into the kernel prunes work as it goes
        product = sp_matmul_topn(
            query, pool, top_n=k, threshold=max(min_score, 0.0) or None,
            sort=False, n_threads=n_threads,
        )
        indptr, indices, data = product.indptr, product.indices, product.data
        for row in range(product.shape[0]):
            for offset in range(indptr[row], indptr[row + 1]):
                yield row, int(indices[offset]), float(data[offset])
        return

    chunk = max(1, min(query.shape[0], int(max_dense_cells // max(n_pool, 1))))
    for start in range(0, query.shape[0], chunk):
        block = (query[start:start + chunk] @ pool.T).toarray()
        # argpartition gives the k largest per row without a full sort
        part = np.argpartition(-block, k - 1, axis=1)[:, :k]
        for row in range(block.shape[0]):
            cols = part[row]
            scores = block[row, cols]
            keep = scores > 0
            for col, score in zip(cols[keep], scores[keep]):
                yield start + row, int(col), float(score)


def _tfidf_channel(s1_df, pool_df, column, k, analyzer, ngram_range, min_score,
                   n_threads=-1):
    """One TF-IDF channel: fit on both sides together, then top-k per S1 row."""
    s1_text, pool_text = _text(s1_df, column), _text(pool_df, column)
    if not any(s1_text) or not any(pool_text):
        return {}

    vectorizer = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        min_df=1,
        sublinear_tf=True,
        dtype=np.float32,
    )
    try:
        vectorizer.fit(s1_text + pool_text)
        s1_matrix = vectorizer.transform(s1_text)
        pool_matrix = vectorizer.transform(pool_text)
    except ValueError:
        # empty vocabulary (e.g. every value missing in a tiny country block)
        return {}

    hits = defaultdict(dict)
    for qi, pi, score in _sparse_topk(
        s1_matrix, pool_matrix, k, min_score=min_score, n_threads=n_threads
    ):
        if score >= min_score:
            hits[qi][pi] = score
    return hits


def _inverted_index_channel(s1_tokens, pool_tokens, k, max_df_ratio, min_shared,
                            min_df_cap=100):
    """
    Shared-token channel. Tokens present in more than `max_df_ratio` of the pool
    carry no signal and would create huge posting lists, so they are skipped.

    The cap has an absolute floor: a pure ratio filters out every token on a
    small corpus (a 2% cap over 1k records skips anything seen more than 20
    times, which is most name tokens) while behaving sensibly at 100k records.
    Remaining tokens are still IDF-weighted, so common ones contribute little.
    """
    n_pool = len(pool_tokens)
    if n_pool == 0:
        return {}

    postings = defaultdict(list)
    for idx, tokens in enumerate(pool_tokens):
        # sorted, not just deduplicated: set iteration order over strings varies
        # between processes (hash randomisation), which changes the order of the
        # float accumulation below. Float addition is not associative, so the
        # tiny differences flip ties at the k-th position and the candidate set
        # stops being reproducible across runs.
        for token in sorted(set(tokens)):
            postings[token].append(idx)

    df_cap = max(1, int(max_df_ratio * n_pool), min(min_df_cap, n_pool))
    idf = {
        token: np.log(n_pool / len(ids))
        for token, ids in postings.items()
        if len(ids) <= df_cap
    }

    hits = {}
    for qi, tokens in enumerate(s1_tokens):
        weights = Counter()
        counts = Counter()
        for token in sorted(set(tokens)):
            if token not in idf:
                continue
            weight = idf[token]
            for pi in postings[token]:
                weights[pi] += weight
                counts[pi] += 1
        if not weights:
            continue
        scored = [(pi, w) for pi, w in weights.items() if counts[pi] >= min_shared]
        scored.sort(key=lambda item: -item[1])
        top = scored[:k]
        if top:
            # normalise so the score is comparable across entities; guard the
            # degenerate case where every shared token has zero idf, which
            # would otherwise divide by zero and emit NaN scores
            best = top[0][1]
            if best > 0:
                hits[qi] = {pi: float(w / best) for pi, w in top}
    return hits


def _embedding_channel(s1_df, pool_df, encoder, k, min_score, n_threads):
    """
    Dense-similarity channel. Imported lazily so the optional embedding
    dependencies are only required when this channel is actually enabled.
    """
    if encoder is None:
        raise ValueError(
            "the 'embedding' channel needs an encoder; pass one as "
            "config['embedding_encoder']"
        )
    from .embeddings import embedding_topk, serialise_records

    query = encoder.encode(serialise_records(s1_df))
    pool = encoder.encode(serialise_records(pool_df))
    return embedding_topk(query, pool, k=k, min_score=min_score, n_threads=n_threads)


def _name_tokens(df):
    column = "business_name_core" if "business_name_core" in df.columns else "business_name_clean"
    return [t.split() for t in _text(df, column)]


def _numeric_tokens(df):
    if "address_numbers" in df.columns:
        return [t.split() for t in _text(df, "address_numbers")]
    return [re.findall(r"\b\d+\b", t) for t in _text(df, "business_address_clean")]


def _country_key(df):
    if "country_clean" not in df.columns:
        return np.array(["__ALL__"] * len(df), dtype=object)
    return df["country_clean"].fillna("__MISSING__").astype(str).to_numpy(dtype=object)


def generate_candidates(s1_df, pool_df, config=None):
    """
    Build the candidate set for every Source 1 entity.

    Args:
        s1_df: preprocessed Source 1 records.
        pool_df: preprocessed Source 2 + Source 3 records, concatenated.
        config: optional overrides for the per-channel top-k and thresholds.

    Returns:
        dict s1_entity_id -> {candidate_entity_id: {channel: score, ...}}
    """
    cfg = {
        "k_name_char": 25,
        "k_addr_char": 25,
        "k_name_word": 25,
        "k_numeric": 25,
        "k_rare_token": 25,
        "min_score_char": 0.15,
        "min_score_word": 0.15,
        "numeric_max_df_ratio": 0.05,
        "rare_token_max_df_ratio": 0.05,
        "n_threads": -1,
        # Which channels to run. Measured unique recall on the real data was
        # addr_char 12.4%, rare_token 0.29%, numeric 0.24%, name_char 0.20%,
        # name_word 0.04%, while rare_token was the single slowest channel.
        # Dropping the cheap-recall channels trades ~0.5% of pairs for ~40% of
        # blocking time, which is worth it when the loss is in the decision
        # layer rather than in blocking.
        # "embedding" is left out by default: it needs sentence-transformers
        # and hnswlib plus a model download, so it must be opted into.
        "channels": [c for c in CHANNELS if c != "embedding"],
        "k_embedding": 25,
        "min_score_embedding": 0.5,
        "embedding_encoder": None,
    }
    cfg.update(config or {})

    s1_ids = s1_df["entity_id"].to_numpy(dtype=object)
    pool_ids = pool_df["entity_id"].to_numpy(dtype=object)

    s1_country, pool_country = _country_key(s1_df), _country_key(pool_df)
    pool_missing_country = np.flatnonzero(pool_country == "__MISSING__")

    candidates = {sid: defaultdict(dict) for sid in s1_ids}
    timings = {}

    countries = sorted(set(s1_country))
    for country_number, country in enumerate(countries, start=1):
        s1_rows = np.flatnonzero(s1_country == country)
        if country == "__MISSING__":
            # unknown country could belong anywhere -> compare against everything
            pool_rows = np.arange(len(pool_df))
        else:
            pool_rows = np.union1d(
                np.flatnonzero(pool_country == country), pool_missing_country
            )
        if len(pool_rows) == 0:
            continue

        s1_block = s1_df.iloc[s1_rows]
        pool_block = pool_df.iloc[pool_rows]
        logger.info(
            "blocking country %d/%d=%s | s1=%d pool=%d",
            country_number, len(countries), country, len(s1_block), len(pool_block),
        )

        name_column = (
            "business_name_core" if "business_name_core" in s1_df.columns else "business_name_clean"
        )
        channel_specs = {
            "name_char": lambda: _tfidf_channel(
                s1_block, pool_block, name_column, cfg["k_name_char"],
                "char_wb", (3, 5), cfg["min_score_char"], cfg["n_threads"],
            ),
            "addr_char": lambda: _tfidf_channel(
                s1_block, pool_block, "business_address_clean", cfg["k_addr_char"],
                "char_wb", (3, 5), cfg["min_score_char"], cfg["n_threads"],
            ),
            "name_word": lambda: _tfidf_channel(
                s1_block, pool_block, name_column, cfg["k_name_word"],
                "word", (1, 1), cfg["min_score_word"], cfg["n_threads"],
            ),
            "numeric": lambda: _inverted_index_channel(
                _numeric_tokens(s1_block), _numeric_tokens(pool_block),
                cfg["k_numeric"], cfg["numeric_max_df_ratio"], min_shared=1,
            ),
            "rare_token": lambda: _inverted_index_channel(
                _name_tokens(s1_block), _name_tokens(pool_block),
                cfg["k_rare_token"], cfg["rare_token_max_df_ratio"], min_shared=1,
            ),
            "embedding": lambda: _embedding_channel(
                s1_block, pool_block, cfg["embedding_encoder"], cfg["k_embedding"],
                cfg["min_score_embedding"], cfg["n_threads"],
            ),
        }
        enabled = set(cfg["channels"])
        unknown = enabled - set(CHANNELS)
        if unknown:
            raise ValueError(
                f"unknown blocking channels: {sorted(unknown)}; expected {list(CHANNELS)}"
            )

        channel_hits = {}
        for channel, build_channel in channel_specs.items():
            if channel not in enabled:
                continue
            channel_started = time.time()
            channel_hits[channel] = build_channel()
            hit_count = sum(len(matches) for matches in channel_hits[channel].values())
            timings[channel] = timings.get(channel, 0.0) + (time.time() - channel_started)
            logger.info(
                "blocking country %d/%d=%s channel=%s done in %.1fs | hits=%d",
                country_number, len(countries), country, channel,
                time.time() - channel_started, hit_count,
            )

        for channel, hits in channel_hits.items():
            for local_qi, matches in hits.items():
                s1_id = s1_ids[s1_rows[local_qi]]
                for local_pi, score in matches.items():
                    candidates[s1_id][pool_ids[pool_rows[local_pi]]][channel] = score

        logger.info(
            "blocking country %d/%d=%s complete | candidates=%d",
            country_number, len(countries), country,
            sum(len(matches) for matches in candidates.values()),
        )

    if timings:
        logger.info(
            "blocking seconds by channel: %s",
            ", ".join(f"{c}={t:.1f}" for c, t in sorted(timings.items(), key=lambda kv: -kv[1])),
        )
    generate_candidates.last_timings = dict(timings)
    return {sid: dict(matches) for sid, matches in candidates.items()}


def channel_contribution(candidates, ground_truth):
    """
    Per-channel recall, so a channel that earns nothing can be dropped and a
    channel doing unique work is kept. `unique_recall` counts true pairs that
    *only* this channel retrieved.
    """
    report = {}
    for channel in CHANNELS:
        found = unique = total = 0
        for s1_id, truth in ground_truth.items():
            matches = candidates.get(s1_id, {})
            for t in truth:
                total += 1
                channels_hit = matches.get(t)
                if channels_hit and channel in channels_hit:
                    found += 1
                    if len(channels_hit) == 1:
                        unique += 1
        report[channel] = {
            "recall": found / total if total else 0.0,
            "unique_recall": unique / total if total else 0.0,
        }
    return report


def _config_cache_view(config):
    """
    A JSON-serialisable, run-stable view of the blocking config.

    The config can hold live objects - the embedding encoder - which are not
    serialisable, and whose repr carries a memory address that would differ on
    every run and defeat the cache entirely. Objects are therefore reduced to a
    stable identifier: what the candidates depend on is which model produced
    the vectors, not which Python object held it.
    """
    view = {}
    for key, value in (config or {}).items():
        if value is None or isinstance(value, (str, int, float, bool)):
            view[key] = value
        elif isinstance(value, (list, tuple)):
            view[key] = [v if isinstance(v, (str, int, float, bool)) else str(v) for v in value]
        else:
            view[key] = getattr(value, "model_name", None) or type(value).__name__
    return view


def _fingerprint(s1_df, pool_df, config):
    """Stable id for a (data, config) combination, used as the cache key."""
    hasher = hashlib.sha256()
    for df in (s1_df, pool_df):
        ids = df["entity_id"].astype(str).to_numpy()
        hasher.update(str(len(ids)).encode())
        hasher.update(hashlib.sha256("\x00".join(ids).encode()).digest())
    hasher.update(json.dumps(_config_cache_view(config), sort_keys=True).encode())
    return hasher.hexdigest()[:16]


def generate_candidates_cached(s1_df, pool_df, config=None, cache_dir=None, tag="candidates"):
    """
    generate_candidates() with an on-disk cache.

    Blocking dominates the runtime of a full run while every downstream
    experiment (features, model, thresholds, conflict stage) leaves it
    unchanged. Caching on a fingerprint of the entity ids plus the config turns
    the iteration loop from hours into minutes, and invalidates itself
    automatically when either the data or the blocking config changes.
    """
    if not cache_dir:
        return generate_candidates(s1_df, pool_df, config)

    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{tag}_{_fingerprint(s1_df, pool_df, config)}.pkl")

    if os.path.exists(path):
        logger.info("loading cached candidates from %s", path)
        with open(path, "rb") as handle:
            return pickle.load(handle)

    candidates = generate_candidates(s1_df, pool_df, config)
    tmp = path + ".tmp"
    with open(tmp, "wb") as handle:
        pickle.dump(candidates, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)  # atomic, so an interrupted write leaves no half cache
    logger.info("cached candidates to %s", path)
    return candidates
