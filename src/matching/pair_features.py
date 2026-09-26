"""
Stage B features: one numeric row per (Source 1 entity, candidate) pair.

Three families:
  * string similarity  - typo / abbreviation / word-order robustness
  * discrete agreement - numeric address tokens, country, missingness
  * entity context     - rank and score gap within the candidate's own entity

The context family is what lets the model distinguish "one clear winner" from
"four near-ties", which is the difference between a safe match and a false merge
under a precision-heavy metric. It is derived from *blocking* scores only, so no
model output feeds back into its own features.
"""
from collections import defaultdict

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from .blocking import CHANNELS, CHANNEL_WEIGHTS, prelim_score

FEATURE_NAMES = [
    "name_ratio", "name_token_set", "name_token_sort", "name_partial", "name_jaro",
    "core_ratio", "core_token_set", "core_jaro",
    "addr_ratio", "addr_token_set", "addr_token_sort", "addr_partial", "addr_jaro",
    "name_token_jaccard", "name_token_containment", "name_idf_overlap",
    "addr_token_jaccard", "addr_token_containment", "addr_idf_overlap",
    "suffix_agree", "suffix_one_missing",
    "num_shared", "num_jaccard", "num_conflict", "num_either_empty",
    "country_match", "country_either_missing",
    "name_len_ratio", "addr_len_ratio",
    "name_missing_either", "addr_missing_either",
    "n_channels_hit", "prelim_score", "embed_cosine",
    "rank_in_entity", "score_gap_to_top", "score_ratio_to_top",
    "n_candidates", "n_candidates_near_top",
]

_LEGAL_SUFFIXES = {
    "corp", "corporation", "inc", "incorporated", "llc", "ltd", "limited",
    "pvt", "private", "co", "company", "plc", "gmbh", "sa", "sas", "sarl", "bv", "ag",
}


def _safe(value):
    return "" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value)


def _jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _containment(a, b):
    """Overlap relative to the smaller set: tolerant of one side being truncated."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _idf_overlap(a, b, idf):
    """
    Shared tokens weighted by rarity, normalised by the smaller side's mass.

    Summed in sorted order because set iteration order over strings varies
    between processes, and float addition is not associative - an unsorted sum
    makes the feature value, and therefore the prediction, irreproducible.
    """
    if not a or not b:
        return 0.0
    shared = sum(idf.get(t, 0.0) for t in sorted(a & b))
    denom = min(sum(idf.get(t, 0.0) for t in sorted(a)),
                sum(idf.get(t, 0.0) for t in sorted(b)))
    return shared / denom if denom > 0 else 0.0


def build_idf(*token_lists):
    """Document frequency over every record in the corpus -> idf per token."""
    df = defaultdict(int)
    n_docs = 0
    for tokens_per_record in token_lists:
        for tokens in tokens_per_record:
            n_docs += 1
            for token in set(tokens):
                df[token] += 1
    if n_docs == 0:
        return {}
    return {token: float(np.log(n_docs / count)) for token, count in df.items()}


class RecordView:
    """Pre-tokenised view of one record, so token work happens once, not per pair."""

    __slots__ = ("name", "core", "addr", "country", "name_tokens", "addr_tokens",
                 "numbers", "suffixes", "name_missing", "addr_missing")

    def __init__(self, row):
        self.name = _safe(row.get("business_name_clean"))
        self.core = _safe(row.get("business_name_core")) or self.name
        self.addr = _safe(row.get("business_address_clean"))
        self.country = _safe(row.get("country_clean"))
        self.name_tokens = set(self.name.split())
        self.addr_tokens = set(self.addr.split())
        self.numbers = set(_safe(row.get("address_numbers")).split())
        self.suffixes = self.name_tokens & _LEGAL_SUFFIXES
        self.name_missing = not self.name
        self.addr_missing = not self.addr


_VIEW_COLUMNS = ("business_name_clean", "business_name_core", "business_address_clean",
                 "country_clean", "address_numbers")


def build_record_views(df, only=None):
    """
    entity_id -> RecordView for the rows of a preprocessed frame.

    `only` restricts the build to a set of entity ids. That matters on the pool
    side during sharded inference: a RecordView holds several Python sets and
    costs roughly a kilobyte, so materialising one for every record of a
    multi-million-row pool runs to gigabytes, while a shard only ever compares
    against the candidates its own blocking produced.
    """
    columns = [c for c in _VIEW_COLUMNS if c in df.columns]
    frame = df[["entity_id"] + columns]

    if only is not None:
        wanted = only if isinstance(only, (set, frozenset)) else set(only)
        if not wanted:
            return {}
        frame = frame[frame["entity_id"].astype(str).isin(wanted)]

    records = frame.to_dict("records")
    return {str(r["entity_id"]): RecordView(r) for r in records}


def _pair_features(s1, cand, channel_scores, context, name_idf, addr_idf):
    prelim, rank, top_score, n_cands, n_near_top = context

    num_shared = len(s1.numbers & cand.numbers)
    both_have_numbers = bool(s1.numbers) and bool(cand.numbers)

    return [
        fuzz.ratio(s1.name, cand.name) / 100.0,
        fuzz.token_set_ratio(s1.name, cand.name) / 100.0,
        fuzz.token_sort_ratio(s1.name, cand.name) / 100.0,
        fuzz.partial_ratio(s1.name, cand.name) / 100.0,
        JaroWinkler.similarity(s1.name, cand.name),
        fuzz.ratio(s1.core, cand.core) / 100.0,
        fuzz.token_set_ratio(s1.core, cand.core) / 100.0,
        JaroWinkler.similarity(s1.core, cand.core),
        fuzz.ratio(s1.addr, cand.addr) / 100.0,
        fuzz.token_set_ratio(s1.addr, cand.addr) / 100.0,
        fuzz.token_sort_ratio(s1.addr, cand.addr) / 100.0,
        fuzz.partial_ratio(s1.addr, cand.addr) / 100.0,
        JaroWinkler.similarity(s1.addr, cand.addr),
        _jaccard(s1.name_tokens, cand.name_tokens),
        _containment(s1.name_tokens, cand.name_tokens),
        _idf_overlap(s1.name_tokens, cand.name_tokens, name_idf),
        _jaccard(s1.addr_tokens, cand.addr_tokens),
        _containment(s1.addr_tokens, cand.addr_tokens),
        _idf_overlap(s1.addr_tokens, cand.addr_tokens, addr_idf),
        # a differing legal suffix can mean genuinely different entities
        1.0 if (s1.suffixes and cand.suffixes and s1.suffixes == cand.suffixes) else 0.0,
        1.0 if (bool(s1.suffixes) != bool(cand.suffixes)) else 0.0,
        float(num_shared),
        _jaccard(s1.numbers, cand.numbers),
        # disjoint house numbers on both sides is strong evidence *against*
        1.0 if (both_have_numbers and num_shared == 0) else 0.0,
        0.0 if both_have_numbers else 1.0,
        1.0 if (s1.country and s1.country == cand.country) else 0.0,
        1.0 if (not s1.country or not cand.country) else 0.0,
        min(len(s1.name), len(cand.name)) / max(len(s1.name), len(cand.name), 1),
        min(len(s1.addr), len(cand.addr)) / max(len(s1.addr), len(cand.addr), 1),
        1.0 if (s1.name_missing or cand.name_missing) else 0.0,
        1.0 if (s1.addr_missing or cand.addr_missing) else 0.0,
        float(len(channel_scores)),
        prelim,
        # filled in afterwards by set_embedding_feature() when embeddings are
        # enabled; kept in the fixed feature layout either way so the model's
        # input width never depends on configuration
        0.0,
        float(rank),
        top_score - prelim,
        prelim / top_score if top_score > 0 else 0.0,
        float(n_cands),
        float(n_near_top),
    ]


def set_embedding_feature(X, values):
    """
    Write the pre-computed cosine column into an already-built feature matrix.

    Kept separate from build_pair_table because the cosines are computed in a
    vectorised chunked pass over all pairs at once; doing it per pair inside
    the Python loop would dominate the runtime.
    """
    column = FEATURE_NAMES.index("embed_cosine")
    if len(values) != len(X):
        raise ValueError(f"expected {len(X)} cosines, got {len(values)}")
    X[:, column] = np.asarray(values, dtype=np.float32)
    return X


def build_pair_table(candidates, s1_views, pool_views, name_idf, addr_idf,
                     near_top_ratio=0.9):
    """
    Featurise every candidate pair.

    Returns:
        (X, pair_index) where X is a float32 array of shape (n_pairs, n_features)
        and pair_index is a DataFrame with columns s1_entity_id / candidate_entity_id
        aligned row-for-row with X.
    """
    rows, s1_col, cand_col = [], [], []

    for s1_id, matches in candidates.items():
        s1 = s1_views.get(s1_id)
        if s1 is None or not matches:
            continue

        scored = sorted(
            ((cand_id, chans, prelim_score(chans)) for cand_id, chans in matches.items()),
            key=lambda item: -item[2],
        )
        top_score = scored[0][2]
        n_cands = len(scored)
        n_near_top = sum(1 for _, _, p in scored if top_score > 0 and p >= near_top_ratio * top_score)

        for rank, (cand_id, chans, prelim) in enumerate(scored):
            cand = pool_views.get(cand_id)
            if cand is None:
                continue
            context = (prelim, rank, top_score, n_cands, n_near_top)
            rows.append(_pair_features(s1, cand, chans, context, name_idf, addr_idf))
            s1_col.append(s1_id)
            cand_col.append(cand_id)

    if not rows:
        empty = pd.DataFrame({"s1_entity_id": [], "candidate_entity_id": []})
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32), empty

    X = np.asarray(rows, dtype=np.float32)
    pair_index = pd.DataFrame(
        {"s1_entity_id": s1_col, "candidate_entity_id": cand_col}
    )
    return X, pair_index
