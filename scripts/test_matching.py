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
    print("\nall matching tests passed")


if __name__ == "__main__":
    main()
