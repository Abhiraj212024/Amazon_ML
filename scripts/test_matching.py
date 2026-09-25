"""
Unit tests for the matching pipeline.

Stage C (conflict resolution) gets explicit coverage here because a realistic
dataset may contain very few contested records, so an end-to-end run can leave
that code path completely untested.

Run: python3 scripts/test_matching.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.matching import io as match_io
from src.matching import metrics, resolve, splits
from src.matching.blocking import generate_candidates
from src.matching.decide import expected_f05_select, select_matches, threshold_select
from src.matching.pair_features import build_idf, build_pair_table, build_record_views
from src.preprocessing.pipeline import preprocess_dataframe


def test_f05_closed_form():
    # the worked example from the problem statement
    assert round(metrics.f05_single({"S2-1", "S2-2", "S3-1"}, {"S2-1", "S3-1"}), 3) == 0.714
    # singletons are defined by the rules, not the formula
    assert metrics.f05_single(set(), set()) == 1.0
    assert metrics.f05_single({"S2-1"}, set()) == 0.0
    assert metrics.f05_single(set(), {"S2-1"}) == 0.0
    # a single false positive on a single-match entity costs 0.444
    assert metrics.f05_single({"S2-1"}, {"S2-1"}) == 1.0
    assert round(metrics.f05_single({"S2-1", "S2-9"}, {"S2-1"}), 3) == 0.556
    print("  f05 closed form OK")


def test_macro_and_baseline():
    gt = {"S1-1": {"S2-1"}, "S1-2": set(), "S1-3": {"S2-3", "S3-3"}}
    assert metrics.all_empty_baseline(gt) == 1 / 3
    score, per_entity = metrics.macro_f05({"S1-1": {"S2-1"}, "S1-2": set(), "S1-3": set()}, gt)
    assert per_entity["S1-1"] == 1.0 and per_entity["S1-2"] == 1.0 and per_entity["S1-3"] == 0.0
    assert round(score, 4) == round(2 / 3, 4)
    # entities absent from predictions are scored as empty, not skipped
    score_missing, _ = metrics.macro_f05({}, gt)
    assert round(score_missing, 4) == round(1 / 3, 4)
    print("  macro F0.5 and baseline OK")


def test_expected_f05_prefers_abstention_when_unsure():
    # a lone weak candidate: better to predict nothing than risk a false merge
    k_weak, _ = expected_f05_select([0.2], n_samples=4000)
    assert k_weak == 0
    # a confident one should be taken
    k_strong, _ = expected_f05_select([0.95], n_samples=4000)
    assert k_strong == 1
    # one confident plus one marginal: the marginal should not be added
    k_mixed, _ = expected_f05_select([0.97, 0.25], n_samples=4000)
    assert k_mixed == 1
    # two confident candidates should both be taken
    k_both, _ = expected_f05_select([0.95, 0.9], n_samples=4000)
    assert k_both == 2
    assert expected_f05_select([], n_samples=100)[0] == 0
    print("  expected-F0.5 selection OK")


def test_threshold_select():
    assert threshold_select([0.9, 0.85, 0.2], t_high=0.5, ratio=0.8) == 2
    assert threshold_select([0.4], t_high=0.5) == 0
    assert threshold_select([], t_high=0.5) == 0
    print("  threshold selection OK")


def test_one_to_one_check():
    clean = {"S1-1": {"S2-1", "S2-2"}, "S1-2": {"S3-1"}}
    assert resolve.check_one_to_one(clean)["holds"] is True
    # one S1 entity legitimately owning several differently-worded records is
    # NOT a violation; two entities claiming the same record is
    violating = {"S1-1": {"S2-1"}, "S1-2": {"S2-1"}}
    report = resolve.check_one_to_one(violating)
    assert report["holds"] is False and report["n_shared_records"] == 1
    print("  one-to-one assumption check OK")


def test_conflict_resolution_post():
    scored = {
        "S1-1": [("S2-9", 0.91), ("S2-1", 0.88)],
        "S1-2": [("S2-9", 0.62)],
    }
    predictions = {"S1-1": {"S2-9", "S2-1"}, "S1-2": {"S2-9"}}
    resolved, report = resolve.resolve_predictions(predictions, scored)
    assert resolved["S1-1"] == {"S2-9", "S2-1"}, "higher-scoring claimant keeps it"
    assert resolved["S1-2"] == set(), "lower-scoring claimant loses it"
    assert report["contested_records"] == 1 and report["ids_dropped"] == 1
    # many records for one entity must survive untouched
    uncontested = {"S1-1": {"S2-1", "S2-2", "S3-1"}}
    kept, rep = resolve.resolve_predictions(
        uncontested, {"S1-1": [("S2-1", 0.9), ("S2-2", 0.8), ("S3-1", 0.7)]}
    )
    assert kept["S1-1"] == {"S2-1", "S2-2", "S3-1"} and rep["contested_records"] == 0
    print("  post-decision conflict resolution OK")


def test_conflict_resolution_pre():
    scored = {"S1-1": [("S2-9", 0.91)], "S1-2": [("S2-9", 0.62), ("S2-3", 0.55)]}
    filtered, report = resolve.resolve_scored_pairs(scored)
    assert [c for c, _ in filtered["S1-1"]] == ["S2-9"]
    assert [c for c, _ in filtered["S1-2"]] == ["S2-3"]
    assert report["pairs_dropped"] == 1
    # a generous margin keeps contested claims alive
    kept, _ = resolve.resolve_scored_pairs(scored, margin=0.5)
    assert len(kept["S1-2"]) == 2
    print("  pre-decision conflict resolution OK")


def _toy_frames():
    s1 = pd.DataFrame([
        {"entity_id": "S1-1", "business_name": "Sunrise Textiles Pvt Ltd",
         "business_address": "42 MG Road, Pune, 411001", "country": "India"},
        {"entity_id": "S1-2", "business_name": "Blue Ocean Foods Inc",
         "business_address": "77 Main St, Austin, 73301", "country": "US"},
        {"entity_id": "S1-3", "business_name": "Granite Motors LLC",
         "business_address": "9 Oak Ave, Denver, 80201", "country": "US"},
    ])
    pool = pd.DataFrame([
        # same business as S1-1, worded differently
        {"entity_id": "S2-1", "business_name": "Sunrise Textiles Private Limited",
         "business_address": "42 M.G. Rd, Pune", "country": "India"},
        # same business again, different wording and a typo
        {"entity_id": "S3-1", "business_name": "Sunrise Textile",
         "business_address": "42 MG Road, near SBI ATM, Pune 411001", "country": "India"},
        {"entity_id": "S2-2", "business_name": "Blue Ocean Foods Incorporated",
         "business_address": "77 Main Street, Austin", "country": "US"},
        # a distractor that belongs to no S1 entity
        {"entity_id": "S2-3", "business_name": "Crimson Steel Works",
         "business_address": "400 Industrial Pkwy, Tampa", "country": "US"},
    ])
    return preprocess_dataframe(s1), preprocess_dataframe(pool)


def test_blocking_recall_and_country_separation():
    s1, pool = _toy_frames()
    candidates = generate_candidates(s1, pool)
    gt = {"S1-1": {"S2-1", "S3-1"}, "S1-2": {"S2-2"}, "S1-3": set()}

    report = metrics.blocking_report(candidates, gt, len(pool))
    assert report["pair_recall"] == 1.0, f"blocking lost a true pair: {report}"
    # country blocking must keep an Indian record out of a US entity's candidates
    assert not ({"S2-1", "S3-1"} & set(candidates["S1-2"])), "country blocking leaked"
    print(f"  blocking recall OK (recall={report['pair_recall']:.2f})")


def test_pair_table_shapes_and_multi_match():
    s1, pool = _toy_frames()
    candidates = generate_candidates(s1, pool)
    s1_views, pool_views = build_record_views(s1), build_record_views(pool)
    name_idf = build_idf([t.split() for t in s1["business_name_core"].fillna("")])
    X, index = build_pair_table(candidates, s1_views, pool_views, name_idf, {})

    assert X.shape[0] == len(index) and X.shape[0] > 0
    assert np.isfinite(X).all(), "non-finite feature values"
    # both differently-worded records for S1-1 must reach the model
    s1_1 = set(index[index.s1_entity_id == "S1-1"].candidate_entity_id)
    assert {"S2-1", "S3-1"} <= s1_1
    print(f"  pair table OK ({X.shape[0]} pairs, {X.shape[1]} features)")


def test_sparse_topk_paths_agree():
    """The fast path and the fallback must return identical pairs."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    import src.matching.blocking as blocking

    if not blocking._HAS_SPARSE_DOT_TOPN:
        print("  sparse_dot_topn not installed, skipping path-agreement test")
        return

    # business-like strings sharing tokens, so the top-k is actually populated
    rng = np.random.default_rng(0)
    words = ["sunrise", "textiles", "global", "foods", "lotus", "motors", "apex",
             "trading", "pharma", "steel", "exports", "riverside"]
    texts = [
        " ".join(rng.choice(words, size=int(rng.integers(2, 5)), replace=False))
        for _ in range(300)
    ]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), dtype=np.float32)
    vec.fit(texts)
    query, pool = vec.transform(texts[:80]), vec.transform(texts[80:])

    threshold = 0.15
    fast = sorted(blocking._sparse_topk(query, pool, 10, min_score=threshold))
    blocking._HAS_SPARSE_DOT_TOPN = False
    try:
        slow = sorted(
            (q, p, sc) for q, p, sc in blocking._sparse_topk(query, pool, 10)
            if sc >= threshold
        )
    finally:
        blocking._HAS_SPARSE_DOT_TOPN = True

    assert len(fast) > 200, f"test is too weak to be meaningful ({len(fast)} pairs)"

    # The two paths can pick different members among candidates whose scores are
    # exactly tied at the k-th position, so comparing chosen ids is wrong. The
    # real invariant is that both return the same top-k *scores* per row.
    def scores_by_row(pairs):
        out = {}
        for q, _, score in pairs:
            out.setdefault(q, []).append(round(float(score), 5))
        return {q: sorted(v, reverse=True) for q, v in out.items()}

    fast_scores, slow_scores = scores_by_row(fast), scores_by_row(slow)
    assert set(fast_scores) == set(slow_scores), "paths disagree on which rows have hits"
    for row, values in fast_scores.items():
        assert values == slow_scores[row], f"row {row}: score sets differ"

    shared = {(q, p) for q, p, _ in fast} & {(q, p) for q, p, _ in slow}
    assert len(shared) / len(fast) > 0.95, "paths disagree beyond tie-breaking"
    print(f"  sparse top-k fast/fallback agreement OK "
          f"({len(fast)} pairs, {len(fast) - len(shared)} tie-break differences)")


def test_candidate_cache_roundtrip(tmp_dir):
    from src.matching.blocking import generate_candidates_cached

    s1, pool = _toy_frames()
    first = generate_candidates_cached(s1, pool, cache_dir=tmp_dir)
    cached = generate_candidates_cached(s1, pool, cache_dir=tmp_dir)
    assert first == cached
    # a different blocking config must not reuse the previous cache entry
    other = generate_candidates_cached(s1, pool, {"k_name_char": 3}, cache_dir=tmp_dir)
    files = [f for f in os.listdir(tmp_dir) if f.endswith(".pkl")]
    assert len(files) == 2, f"cache key did not separate configs: {files}"
    assert isinstance(other, dict)
    print("  candidate cache round-trip OK")


def test_tiered_matches_break_even_rule():
    """
    The tiered rule exists because the break-even acceptance probability
    q* = m/(m+0.25n) rises with m, so one threshold cannot serve both the first
    accept and the later ones.
    """
    from src.matching.decide import tiered_select

    # a decent top candidate plus weak followers: take one, not three
    assert tiered_select([0.55, 0.30, 0.10], t_first=0.4, t_rest=0.7, ratio=0.4) == 1
    # a single threshold of 0.7 would have taken nothing at all here
    assert threshold_select([0.55, 0.30, 0.10], t_high=0.7, ratio=0.4) == 0
    # strong followers are still taken
    assert tiered_select([0.95, 0.88, 0.80], t_first=0.4, t_rest=0.7, ratio=0.4) == 3
    assert tiered_select([], t_first=0.4, t_rest=0.7, ratio=0.4) == 0
    print("  tiered rule OK")


def test_vectorised_tuning_matches_scalar_rules():
    """
    The tuner grid-searches with numpy while the pipeline predicts with the
    scalar functions. If those two ever disagree, every tuned parameter is
    optimising something the pipeline does not actually do.
    """
    from src.matching.decide import _threshold_k, _tiered_k, tiered_select

    rng = np.random.default_rng(0)
    max_k = 10
    rows = []
    for _ in range(300):
        n = int(rng.integers(0, max_k + 1))
        rows.append(np.sort(rng.random(n))[::-1] if n else np.array([]))
    P = np.full((len(rows), max_k), -1.0)
    for i, row in enumerate(rows):
        P[i, :len(row)] = row

    for t_first in (0.05, 0.4, 0.8):
        for t_rest in (0.2, 0.6, 0.9):
            for ratio in (0.05, 0.5, 1.0):
                vec = _tiered_k(P, t_first, t_rest, ratio)
                ref = np.array([tiered_select(r, t_first, t_rest, ratio, max_k) for r in rows])
                assert (vec == ref).all(), f"tiered mismatch at {t_first}/{t_rest}/{ratio}"

    for t in (0.1, 0.5, 0.9):
        for ratio in (0.05, 0.6, 1.0):
            vec = _threshold_k(P, t, ratio)
            ref = np.array([threshold_select(r, t, ratio, max_k) for r in rows])
            assert (vec == ref).all(), f"threshold mismatch at {t}/{ratio}"
    print("  vectorised tuner == scalar rules OK")


def _synthetic_scored(seed=0, n_entities=150):
    rng = np.random.default_rng(seed)
    scored, gt = {}, {}
    for i in range(n_entities):
        s1_id = f"S1-{i}"
        n_true = int(rng.integers(0, 5))
        truth = {f"S2-{i}-{j}" for j in range(n_true)}
        pairs = [(c, float(np.clip(rng.normal(0.8, 0.15), 0, 1))) for c in truth]
        pairs += [(f"X-{i}-{j}", float(np.clip(rng.normal(0.2, 0.15), 0, 1))) for j in range(8)]
        scored[s1_id] = pairs
        gt[s1_id] = truth
    return scored, gt


def test_tiered_tuning_never_worse_than_threshold():
    """
    Tiered reduces to the single threshold when t_first == t_rest, and its grid
    covers that case, so its tuned optimum can never be the worse of the two.
    """
    from src.matching.decide import tune_threshold, tune_tiered

    scored, gt = _synthetic_scored()
    flat = tune_threshold(scored, gt)
    tiered = tune_tiered(scored, gt)
    assert tiered["macro_f05"] >= flat["macro_f05"] - 1e-9, (
        f"tiered {tiered['macro_f05']:.4f} < threshold {flat['macro_f05']:.4f}"
    )
    print(f"  tiered >= threshold OK ({tiered['macro_f05']:.4f} vs {flat['macro_f05']:.4f})")


def test_tuned_params_reproduce_when_applied():
    """The score the tuner reports must be the score the pipeline then gets."""
    from src.matching.decide import tune_tiered

    scored, gt = _synthetic_scored(seed=3)
    tuned = tune_tiered(scored, gt)
    preds = select_matches(
        scored, "tiered", t_first=tuned["t_first"], t_rest=tuned["t_rest"],
        ratio=tuned["ratio"], max_k=10,
    )
    applied, _ = metrics.macro_f05(preds, gt)
    assert abs(applied - tuned["macro_f05"]) < 1e-9, f"{applied} != {tuned['macro_f05']}"
    print("  tuned score reproduces when applied OK")


def test_grid_edge_flag_only_when_binding():
    """
    A parameter pinned to an edge because the range is too narrow must be
    flagged; one sitting at an edge because it has no effect must not be, or
    the warning sends you widening a range that changes nothing.
    """
    from src.matching.decide import _edge_flags

    grids = {"a": np.arange(0.0, 1.01, 0.5), "b": np.arange(0.0, 1.01, 0.5)}
    # every optimum pins 'a' to the top edge -> genuinely limited by the range
    pinned = [{"a": 1.0, "b": 0.0}, {"a": 1.0, "b": 0.5}]
    assert "a" in _edge_flags(pinned, grids)
    # 'b' also appears interior among equally-good optima -> not limiting
    assert "b" not in _edge_flags(pinned, grids)
    print("  grid-edge flag semantics OK")


def test_score_cache_roundtrip_and_key(tmp_dir):
    from src.matching import scorecache

    a = {"S1-1": {"S2-1": {}, "S2-2": {}}}
    b = {"S1-1": {"S2-2": {}, "S2-1": {}}}
    assert scorecache.fingerprint(a) == scorecache.fingerprint(b), "key must ignore ordering"
    assert scorecache.fingerprint(a) != scorecache.fingerprint({"S1-1": {"S2-1": {}}})
    assert scorecache.fingerprint(a) != scorecache.fingerprint(a, {"seed": 1})

    key = scorecache.fingerprint(a)
    assert scorecache.load(tmp_dir, "val", key) is None
    scorecache.save(tmp_dir, "val", key, {"scored": {"S1-1": [("S2-1", 0.9)]}})
    assert scorecache.load(tmp_dir, "val", key)["scored"]["S1-1"] == [("S2-1", 0.9)]

    # a corrupt cache must be ignored, not crash the run
    path = os.path.join(tmp_dir, f"val_scores_{key}.pkl")
    with open(path, "wb") as handle:
        handle.write(b"not a pickle")
    assert scorecache.load(tmp_dir, "val", key) is None
    print("  score cache round-trip and corruption handling OK")


def test_calibration_report():
    from src.matching.scorecache import calibration_report

    probs = np.array([0.05, 0.1, 0.9, 0.95])
    perfect = calibration_report(probs, np.array([0, 0, 1, 1]))
    inverted = calibration_report(probs, np.array([1, 1, 0, 0]))
    assert perfect["brier_score"] < inverted["brier_score"]
    assert calibration_report(np.array([]), np.array([])) == {}
    print("  calibration report OK")


def test_channel_pruning_and_zero_idf_guard():
    s1, pool = _toy_frames()
    lean = generate_candidates(s1, pool, {"channels": ["addr_char", "name_char"]})
    used = {c for matches in lean.values() for chans in matches.values() for c in chans}
    assert used <= {"addr_char", "name_char"}, f"disabled channels ran: {used}"

    gt = {"S1-1": {"S2-1", "S3-1"}, "S1-2": {"S2-2"}, "S1-3": set()}
    assert metrics.blocking_report(lean, gt, len(pool))["pair_recall"] == 1.0

    try:
        generate_candidates(s1, pool, {"channels": ["nope"]})
        raise AssertionError("unknown channel should be rejected")
    except ValueError:
        pass

    # a token in every pool record has zero idf; the normalisation used to
    # divide by zero and emit NaN scores into the features
    full = generate_candidates(s1, pool)
    scores = [v for m in full.values() for chans in m.values() for v in chans.values()]
    assert scores and not any(np.isnan(v) for v in scores), "NaN score leaked from blocking"
    print("  channel pruning and zero-idf guard OK")


class _StubEncoder:
    """Deterministic hashed encoder, so the embedding path is testable offline."""

    model_name = "stub"

    def __init__(self, dim=24):
        self.dim = dim

    def encode(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in str(text).split():
                out[row, hash(token) % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-9)


def test_embedding_channel_and_cosines():
    from src.matching.embeddings import (
        build_embedding_lookup, embedding_topk, pair_cosines, serialise_records,
    )

    s1, pool = _toy_frames()
    assert "name:" in serialise_records(s1)[0] and "country:" in serialise_records(s1)[0]

    encoder = _StubEncoder()
    s1_map, s1_vec = build_embedding_lookup(s1, encoder)
    pool_map, pool_vec = build_embedding_lookup(pool, encoder)
    assert s1_vec.dtype == np.float32
    assert np.allclose(np.linalg.norm(s1_vec, axis=1), 1.0, atol=1e-5), "vectors not normalised"

    candidates = generate_candidates(
        s1, pool, {"channels": ["embedding"], "embedding_encoder": encoder,
                   "min_score_embedding": -1.0},
    )
    assert any(candidates.values()), "embedding channel returned nothing"

    # the channel must refuse to run without an encoder rather than silently skip
    try:
        generate_candidates(s1, pool, {"channels": ["embedding"]})
        raise AssertionError("missing encoder should be rejected")
    except ValueError:
        pass

    pairs = pd.DataFrame({
        "s1_entity_id": ["S1-1", "S1-1", "S1-2"],
        "candidate_entity_id": ["S2-1", "S3-1", "UNKNOWN"],
    })
    cos = pair_cosines(pairs, s1_map, s1_vec, pool_map, pool_vec)
    expected = float(s1_vec[s1_map["S1-1"]] @ pool_vec[pool_map["S2-1"]])
    assert abs(cos[0] - expected) < 1e-5
    assert cos[2] == 0.0, "unknown ids must score 0, not crash"
    print("  embedding channel and cosine feature OK")


def test_ann_recall_against_exact():
    """
    The ANN index must not quietly lose recall - the whole point of the channel
    is the pairs the lexical ones miss.
    """
    from src.matching.embeddings import embedding_topk

    rng = np.random.default_rng(1)
    dim, k = 32, 10
    query = rng.normal(size=(150, dim)).astype(np.float32)
    pool = rng.normal(size=(5000, dim)).astype(np.float32)
    query /= np.linalg.norm(query, axis=1, keepdims=True)
    pool /= np.linalg.norm(pool, axis=1, keepdims=True)

    exact = query @ pool.T
    want = {r: set(np.argsort(-exact[r])[:k].tolist()) for r in range(len(query))}
    got = embedding_topk(query, pool, k=k, min_score=-1.0)
    recall = np.mean([len(set(got.get(r, {})) & want[r]) / k for r in range(len(query))])
    assert recall > 0.97, f"ANN recall too low: {recall:.3f}"
    print(f"  ANN recall vs exact OK ({recall:.3f})")


def test_embed_cosine_feature_slot():
    from src.matching.pair_features import FEATURE_NAMES, set_embedding_feature

    assert "embed_cosine" in FEATURE_NAMES
    X = np.zeros((3, len(FEATURE_NAMES)), dtype=np.float32)
    set_embedding_feature(X, [0.1, 0.2, 0.3])
    assert np.allclose(X[:, FEATURE_NAMES.index("embed_cosine")], [0.1, 0.2, 0.3])
    try:
        set_embedding_feature(X, [0.1])
        raise AssertionError("length mismatch should be rejected")
    except ValueError:
        pass
    print("  embed_cosine feature slot OK")


def test_split_is_entity_level_and_stratified():
    gt = {f"S1-{i}": (set() if i % 3 == 0 else {f"S2-{i}"}) for i in range(60)}
    country_of = {f"S1-{i}": ("India" if i % 2 else "US") for i in range(60)}
    train, val = splits.holdout_split(list(gt), gt, country_of, 0.25, seed=1)
    assert not set(train) & set(val), "entity appears in both splits"
    assert len(train) + len(val) == 60
    summary = splits.split_summary(train, val, gt, country_of)
    assert abs(summary["train"]["singleton_rate"] - summary["val"]["singleton_rate"]) < 0.12
    print("  entity-level stratified split OK")


def test_submission_io_roundtrip(tmp_dir):
    gt_path = os.path.join(tmp_dir, "gt.tsv")
    pd.DataFrame([
        {"source1_entity_id": "S1-1", "matched_entity_ids": "S2-1,S3-1"},
        {"source1_entity_id": "S1-2", "matched_entity_ids": ""},
    ]).to_csv(gt_path, sep="\t", index=False)

    gt = match_io.load_ground_truth(gt_path)
    assert gt == {"S1-1": {"S2-1", "S3-1"}, "S1-2": set()}

    out = os.path.join(tmp_dir, "matching_results.tsv")
    match_io.write_matching_results(out, ["S1-1", "S1-2"], {"S1-1": {"S3-1", "S2-1"}})
    written = pd.read_csv(out, sep="\t", dtype=str).fillna("")
    assert list(written.columns) == ["source1_entity_id", "matched_entity_ids"]
    assert written.iloc[0]["matched_entity_ids"] == "S2-1,S3-1"
    assert written.iloc[1]["matched_entity_ids"] == "", "singletons must be an empty cell"
    print("  submission IO round-trip OK")


def test_select_matches_covers_every_entity():
    scored = {"S1-1": [("S2-1", 0.95)], "S1-2": []}
    for strategy in ("expected_f05", "threshold", "top1"):
        preds = select_matches(scored, strategy)
        assert set(preds) == {"S1-1", "S1-2"}, f"{strategy} dropped an entity"
        assert preds["S1-2"] == set()
    print("  every entity retained by the decision layer OK")


def main():
    import tempfile

    print("running matching pipeline tests\n")
    test_f05_closed_form()
    test_macro_and_baseline()
    test_expected_f05_prefers_abstention_when_unsure()
    test_threshold_select()
    test_tiered_matches_break_even_rule()
    test_vectorised_tuning_matches_scalar_rules()
    test_tiered_tuning_never_worse_than_threshold()
    test_tuned_params_reproduce_when_applied()
    test_grid_edge_flag_only_when_binding()
    test_calibration_report()
    test_channel_pruning_and_zero_idf_guard()
    test_embedding_channel_and_cosines()
    test_ann_recall_against_exact()
    test_embed_cosine_feature_slot()
    test_one_to_one_check()
    test_conflict_resolution_post()
    test_conflict_resolution_pre()
    test_blocking_recall_and_country_separation()
    test_sparse_topk_paths_agree()
    test_pair_table_shapes_and_multi_match()
    test_split_is_entity_level_and_stratified()
    test_select_matches_covers_every_entity()
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_submission_io_roundtrip(tmp_dir)
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_candidate_cache_roundtrip(tmp_dir)
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_score_cache_roundtrip_and_key(tmp_dir)
    print("\nall matching tests passed")


if __name__ == "__main__":
    main()
