# Business Entity Resolution Preprocessing Pipeline

This directory contains the data preprocessing pipeline for the Amazon ML Challenge 2026 Business Entity Resolution task. 
This pipeline focuses purely on cleaning and normalizing text fields without using any external databases, APIs, or libraries outside of pandas/numpy.

## Preprocessing Steps Included
1. **Data Loading & Validation**: Safely reads Source 1, 2, and 3 TSV files, validates schemas and extracts source origin.
2. **Missing Value Handling**: Converts 'NaN', 'N/A', etc., into standard `pd.NA` for internal consistency, adding `name_missing`, `address_missing`, and `country_missing` flags.
3. **Unicode & Whitespace Normalization**: Uses NFKC normalization and controls repeating whitespace/newlines, maintaining clean string encodings.
4. **Business Name Cleaning**: Generates `business_name_clean` and `business_name_core` (with common legal suffixes removed).
5. **Address Cleaning**: Expands common address abbreviations (e.g. 'rd' -> 'road') into `business_address_clean`.
6. **Country Handling**: Title-cases country strings in `country_clean`.
7. **Feature Extraction**: Generates length features, token counts, and extracts numeric sequences to preserve discriminative information like house/postal numbers.

## Non-Destructive Principles
The original `business_name`, `business_address`, and `country` values are strictly preserved as per challenge requirements. All transformations output to new columns (`_clean`, `_core`, etc.). Ground truth data is untouched.

## Setup and Usage

Install requirements:
```bash
pip install -r requirements.txt
```

Run the pipeline:
```bash
python scripts/run_preprocessing.py \
    --train-dir ../dataset/train \
    --test-dir ../dataset/test \
    --output-dir data/processed \
    --report-file reports/preprocessing_report.json
```

The cleaned dataset produced here (in Parquet format) is designed to be directly ingested by the matching pipeline below.

---

# Matching Pipeline

`src/matching/` implements the staged filtering pipeline that turns preprocessed
records into the two required submission files.

```
blocking  ->  pairwise model  ->  conflict resolution  ->  set selection
(stage A)     (stage B)           (stage C)                (stage D)
```

## Why the pipeline is shaped this way

The scored metric is a macro-average of a per-Source-1-entity F0.5. Expanding
F-beta for a predicted set of size `m`, a true set of size `n` and `c` correct
predictions gives a closed form the whole pipeline is built around:

```
F0.5 = 1.25 * c / (m + 0.25 * n)
```

| Case | Score |
| --- | --- |
| one true match, predicted alone | 1.000 |
| one true match, plus one false positive | 0.556 |
| true singleton, empty prediction | 1.000 |
| true singleton, any prediction | 0.000 |
| three true matches, one predicted alone | 0.714 |

A single false merge on a single-match entity costs 0.44, while dropping a third
true match costs 0.29. Precision on small entities dominates, and correctly
predicting singletons is worth a full point each. Every design choice downstream
follows from that asymmetry.

## Stages

**A. Blocking** (`blocking.py`) — a *union* of five channels, so a true pair only
has to survive one of them. Recall lost here is unrecoverable.

| Channel | Method | Noise it handles |
| --- | --- | --- |
| `name_char` | char 3-5 gram TF-IDF on the name | typos, suffix variants |
| `addr_char` | char 3-5 gram TF-IDF on the address | abbreviations, spacing |
| `name_word` | word-level TF-IDF on the name | word-order transposition |
| `numeric` | shared house / postal numbers | renamed businesses |
| `rare_token` | shared high-IDF name tokens | distinctive long-tail names |

Everything is blocked by country first. Country is treated as an open set of
string labels — nothing is hard-coded to `{US, India}` — and records with a
missing country are pooled into every country block rather than dropped.

**B. Pairwise model** (`model.py`, `pair_features.py`) — gradient-boosted
classifier over 38 features: string similarities, IDF-weighted token overlap,
numeric-address agreement *and conflict*, legal-suffix agreement, plus entity
context (rank within the entity, gap to the top candidate, number of near-ties).
Context features are derived from blocking scores only, so no model output feeds
back into its own features. Output is isotonic-calibrated, because stage D
treats the score as a genuine probability. Negatives come from the pipeline's
own blocking output, so the model learns the boundary it is actually asked about.

**C. Conflict resolution** (`resolve.py`) — Source 1 is deduplicated, which
implies each Source 2/3 record describes at most one Source 1 entity. Note the
direction: one Source 1 entity may legitimately collect many Source 2/3 records
that word the same business differently — that is the one-to-many the task asks
for. The constraint only fires when two *different* Source 1 entities claim the
same record, where at most one can be right. `check_one_to_one()` verifies the
assumption against the training ground truth first, and the stage disables
itself automatically if it does not hold.

**D. Set selection** (`decide.py`) — the measured loss is here. On the real
5% slice, 3,865 true matches reached the model and were rejected against only
1,412 that blocking never retrieved, at 98.3% precision and 93.1% recall: far
too conservative for a metric that trades precision for recall at 2.67:1.

The break-even acceptance probability for an entity holding `m` correct
predictions out of `n` true matches is

```
q* = m / (m + 0.25n)
```

which at n=4 runs 0.00, 0.50, 0.67, 0.75, 0.80 for m = 0..4. A single threshold
cannot express that: tuned for m=3 it discards the first accept, where the bar
should be far lower. `tiered` (the default) splits the first accept from the
rest, capturing most of that shape with one extra parameter and depending only
on the ordering of the scores rather than their calibration.

`expected_f05` (below) — rather than one global threshold,
`expected_f05` estimates the expected F0.5 of each top-k prefix by Monte Carlo
over the calibrated probabilities and returns the best `k`. Predicting a
singleton (`k=0`) falls out of the same computation, so no separate abstain rule
is needed. `threshold` (top-1 plus a margin, tuned directly against macro F0.5)
and `top1` are kept as controls.

## Tuning the decision layer

`tune_tiered` and `tune_threshold` grid-search **directly against macro F0.5**,
vectorised so a three-dimensional grid costs milliseconds per point. That
matters twice over: the ranges can be wide enough for the optimum to be
interior, and the tuner reports `on_grid_edge` when it is not.

That flag is deliberately narrow. A parameter is flagged only when *every*
setting achieving the best score puts it on an edge — a parameter that sits at
an edge merely because it has no effect (when `ratio` binds harder than
`t_rest`, every value scores the same) is not flagged, since widening that
range would change nothing. `n_optimal_settings` shows how flat the optimum is.

The first tuning run on real data picked `ratio = 0.4` from a grid that started
at 0.4 — the range, not the data, chose it. Both grids now start at 0.05.

`--cache-dir` caches the featurise-and-score stage as well as blocking, so a
full sweep is effectively free rather than a 14-minute loop.

Every run reports **calibration** (Brier score plus a reliability table).
`expected_f05` treats a score as a real probability when it weighs adding a
candidate, so a large gap between `mean_predicted` and `observed_rate` is the
signal that it is abstaining for the wrong reason.

## Channel pruning

Measured unique recall on the real data — the share of true pairs *only* that
channel retrieved:

| Channel | Recall | Unique | Cost (US block) |
| --- | --- | --- | --- |
| `addr_char` | 0.898 | **0.1243** | 189s |
| `rare_token` | 0.733 | 0.0029 | 266s |
| `numeric` | 0.325 | 0.0024 | 29–95s |
| `name_char` | 0.720 | 0.0020 | 57s |
| `name_word` | 0.707 | 0.0004 | 5s |

`addr_char` carries the stage almost single-handedly, and `rare_token` is the
slowest channel for the second-smallest unique contribution. Since the loss is
in the decision layer rather than in blocking, trading ~0.5% of pairs for ~40%
of blocking time is usually worth it:

```bash
python3 scripts/run_matching.py --train-dir dataset/train \
    --channels name_char,addr_char,name_word --cache-dir .cache
```

## Embeddings (opt-in)

A sixth blocking channel plus an `embed_cosine` pairwise feature, aimed at the
transliteration and wording variation the character n-grams miss — which is
where the India/US gap sits (0.9425 vs 0.9740) and the only lever available for
the unseen test country.

```bash
pip install sentence-transformers hnswlib
python3 scripts/run_matching.py --train-dir dataset/train --embeddings --cache-dir .cache
```

### Checking the embedding path quickly

Encoding a full pool is the slow part, so verify the path works on a slice that
finishes in minutes before committing to a long run:

```bash
./scripts/smoke_embeddings.sh --train-dir dataset_5pct/train --entities 400
```

It carves a tiny *label-consistent* slice (shrinking the pool too, since that is
what drives encoding cost), runs the pipeline with `--embeddings`, and asserts
the embedding channel actually ran and retrieved true pairs. It answers "does
this work", not "is it any good" - the slice is far too small for its score to
mean anything.

`--embedding-model hashing` swaps in a deterministic character-n-gram encoder
that needs no download. It captures no semantics, so it is only for exercising
the plumbing offline or in CI.

### Making it fast enough to use

Two things dominate, and both are handled:

- **Encoding is cached.** Vectors are stored under `--cache-dir`, keyed by the
  model name and the exact serialised text. Without this every run re-encodes
  the whole pool, which is what makes embeddings feel unusable on a laptop.
  Preprocessing changes invalidate the cache automatically.
- **The device is auto-detected.** On Apple Silicon this picks the MPS backend,
  which is several times faster than CPU for encoding. Override with
  `--embedding-device`.

Off by default: it needs both packages and a model download. `--embedding-model`
picks the model — verify its licence on the model card, since the challenge
requires MIT/Apache-2.0 and at most 8B parameters. Prefer a multilingual model.

Search is approximate (hnswlib); exact search over ~100k x 900k records is not
tractable. The index defaults to `M=32, ef=8k`, measured at 0.999 recall against
exact search on structureless vectors, where `ef=2k` gave only 0.82. The
`embed_cosine` feature is exact regardless, computed in a chunked vectorised
pass because gathering every pair's vectors at once would need tens of GB.

## Usage

```bash
# validation only
python3 scripts/run_matching.py --train-dir dataset/train

# validation plus test predictions and submission files
python3 scripts/run_matching.py \
    --train-dir dataset/train \
    --test-dir dataset/test \
    --output-dir output \
    --report-file reports/matching_report.json

# generalisation to a country the model never saw (proxy for France)
python3 scripts/run_matching.py --train-dir dataset/train --holdout-country India
```

Useful flags: `--strategy {expected_f05,threshold,top1}`,
`--conflict-stage {none,pre,post}`, `--no-ablation`.

Writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`, both of
which pass `utils/validate_submission.py`.

## Evaluation discipline

Entities are split three ways — **fit / tune / validation** — all at the Source 1
entity level, stratified on country and true-match count. Thresholds are
hyperparameters, so they are tuned on the `tune` slice; tuning them on the
validation slice and reporting on it inflates the score. The full Source 2/3 pool
stays available to blocking for validation entities, because thinning the
distractors would inflate precision, the quantity this metric is most sensitive
to.

Every run reports the **all-empty baseline** (the singleton fraction), which is
the score of predicting nothing at all and the floor any model must beat.

`--report-file` also captures blocking recall and reduction ratio, per-channel
and *unique* recall (so a channel earning nothing can be dropped), feature
importance, error attribution by stage (`lost_in_blocking` / `lost_in_decision`
/ `false_merge` / `singleton_broken`), and macro F0.5 sliced by country and by
true-match count.

## Pre-flight check before Kaggle / AWS

One command that proves the pipeline is correct and the data is consistent,
before a full run costs money or a session quota.

```bash
./scripts/preflight.sh                             # code checks only
./scripts/preflight.sh --data-dir dataset/train    # also check your real data
./scripts/preflight.sh --data-dir dataset/train --scale-probe
```

It exits 0 only when everything below passes, so it can gate a launch script:

1. Python version, required and optional packages, core count and memory
   (useful for sizing an instance).
2. Unit tests for both preprocessing and matching.
3. A full synthetic end-to-end run: generate data, block, train, predict, write
   both submission files.
4. **Self-test of the consistency checker** on known-good data — if it cannot
   pass that, its verdict on your real data is worthless.
5. **Cache correctness** — a second run must reuse the cache *and* reproduce the
   first run's output byte for byte.
6. The official `utils/validate_submission.py`, with `--check-ids`.
7. With `--data-dir`: the recall ceiling on your real data, and the all-empty
   baseline you have to beat.
8. With `--scale-probe`: times blocking at several sampling fractions, fits the
   growth curve and projects the full-data cost. Blocking grows with the
   *product* of the two sides, so a 5% sample does roughly 0.25% of the full
   work — extrapolating linearly underestimates it badly.

Flags: `--quick` (skip the synthetic end-to-end), `--keep` (retain artifacts for
inspection), `PYTHON=...` to pick an interpreter.

Execute it, do not `source` it. Sourcing runs it in your current shell, so the
bash shebang is ignored — under zsh that previously resolved the repo root to
`/` and ran everything from there. The script now refuses to be sourced, and
re-execs itself under bash if invoked from another shell.

## Check your data before trusting a score

The fastest way to get a meaningless result is to subsample the three source
files independently of the ground truth. The labels still name Source 2/3
records that are no longer in the pool, so no blocking strategy can retrieve
them and pair recall collapses to roughly the sampling rate.

```bash
python3 scripts/diagnose_data.py --train-dir dataset/train
```

Look at `recall_ceiling` — the best pair recall achievable on those files. If it
is far below 1.0, the data is inconsistent, not the model. `unwinnable_rate` is
the share of entities that cannot score above 0 whatever you predict.

To work on a smaller slice, sample Source 1 *entities* and carry their full
match sets:

```bash
python3 scripts/make_subsample.py \
    --train-dir dataset/train --output-dir dataset_5pct/train --fraction 0.05
python3 scripts/diagnose_data.py --train-dir dataset_5pct/train   # expect 100%
```

`--distractor-multiplier` controls how many unrelated pool records come along.
The default keeps the full data's pool-to-entity ratio; a thinned pool makes
matching artificially easy and inflates precision.

## Performance

Blocking dominates the runtime; everything downstream is comparatively cheap.

- **`sparse_dot_topn`** (in `requirements.txt`) computes the top-k of the sparse
  product without densifying it, multi-threaded. It is roughly 10x faster
  single-threaded than the fallback and ~20x with 4 cores. `scripts/test_matching.py`
  asserts both paths return the same top-k scores. If the package is missing the
  pipeline still runs, just slowly.
- **`--cache-dir`** stores the candidate set keyed by a fingerprint of the entity
  ids plus the blocking config, so tuning the model, thresholds or conflict
  stage costs minutes instead of re-running blocking. The key invalidates
  itself when either the data or the blocking config changes.
- **`--blocking-threads`** sets the thread count for the sparse product
  (default: all cores).

```bash
python3 scripts/run_matching.py --train-dir dataset/train --cache-dir .cache
```

## Building the submission package

```bash
python3 scripts/build_submission.py \
    --team-name your_team \
    --output-dir output --test-dir dataset/test \
    --report-file reports/matching_full.json
```

Produces `dist/your_team_submission.zip` in the required layout:

```
<team_name>_submission.zip
|-- output/{matching_results,candidate_pairs}.tsv
|-- code/business_entity_resolution/{src,scripts,utils}/ + README.md + requirements.txt
`-- Documentation_template.md
```

Two challenge rules drive it. Submissions that fail validation are not
evaluated, so the outputs go through `utils/validate_submission.py` first and
the archive is **refused** if they do not pass (override with
`--skip-validation`, which you should not need). And dependencies must be
pinned, so `requirements.txt` is written from the versions actually installed
in the environment that produced the outputs, not from this repo's ranges.

Without `--test-dir` validation is reported as SKIPPED rather than as a pass -
the central rule is that every test Source 1 entity appears exactly once, which
cannot be checked without the test set.

### The methodology document

The challenge's `Documentation_template.md` is vendored at
`docs/Documentation_template.md` and used as the base automatically; override
with `--documentation`.

Its **measurable** placeholders are filled from the run report: blocking keys,
candidate pair counts, recall, the feature set, the model, the threshold
selection method, the macro F_0.5, and the false positive / false negative
breakdown from the error attribution. Section B gains a full results appendix,
and Appendix A gains the code layout and entry point.

**The prose is left as prompts on purpose** - the executive summary, problem
analysis, solution strategy and conclusion are judgement rather than
measurement, and inventing them would put unverified claims in a document the
organisers review. The team member list is left blank too.

## One command for a full submission run

```bash
./scripts/cloud_run.sh --data-dir dataset --team-name your_team
```

Runs consistency check -> matching -> validation -> packaging, stopping at the
first failure, because every later step is worthless if an earlier one is
wrong. Logs and reports land in `<out-dir>/reports/`. Accepts `--channels`,
`--embeddings`, `--cache-dir` and `--extra "..."` for anything else.

### On SageMaker

A **Processing Job** fits better than a notebook: it is batch, it terminates
itself, and it stages S3 in and out for you.

```python
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput

ScriptProcessor(
    image_uri="<an image with python3 + git>",
    command=["/bin/bash"],
    instance_type="ml.r5.4xlarge",   # memory-optimised: memory binds before time
    instance_count=1, volume_size_in_gb=100, role=role,
).run(
    code="scripts/cloud_run.sh",
    arguments=["--data-dir", "/opt/ml/processing/input",
               "--out-dir", "/opt/ml/processing/output",
               "--team-name", "your_team"],
    inputs=[ProcessingInput(source="s3://bucket/dataset",
                            destination="/opt/ml/processing/input")],
    outputs=[ProcessingOutput(source="/opt/ml/processing/output",
                              destination="s3://bucket/submission")],
)
```

Pick an `r` instance rather than a `c` one. GPU buys nothing here except for
the embedding encoder. Put `--cache-dir` on the mounted volume and keep it
between runs; re-running without it repeats blocking, scoring and encoding.

### On Kaggle

Workable, with two limits to plan around: a **12-hour** session cap and
**~20 GB** of writable space in `/kaggle/working`. Attach the challenge files
as a private Dataset (uploading data you were given is not external data
lookup - the rule forbids looking entities up in outside sources).

```bash
!pip install -q sparse_dot_topn rapidfuzz          # internet must be on
!git clone <your repo> /kaggle/working/repo
!cd /kaggle/working/repo && ./scripts/cloud_run.sh \
    --data-dir /kaggle/input/<your-dataset> \
    --out-dir /kaggle/working/out --team-name your_team
```

`/kaggle/input` is read-only, so `--out-dir` and `--cache-dir` must point at
`/kaggle/working`. If a full run will not fit in 12 hours, split it: one
session to build and save the candidate cache, a second to score from it.

## Running on a bigger machine

Training does not need the full data - the fit slice is already ample - but
inference over the complete test set is fixed by the challenge and is the run
that needs headroom. Blocking cost grows with the *product* of the two sides,
so a 5% slice does roughly 0.25% of the full work; measure before sizing with
`scripts/scale_probe.py`.

Memory binds before time. At the 5% slice the candidate set and feature matrix
are about 1 GB each; at full scale that is ~20 GB each, so prefer a
memory-optimised instance over a merely fast one. GPU only helps the embedding
encoder, not the rest of the pipeline.

Carry `--cache-dir` onto the remote machine if you can: blocking, scores and
embedding vectors all live there, and re-running without it repeats everything.

## Testing without the dataset

```bash
python3 scripts/make_synthetic_data.py --output-dir dataset_synthetic
python3 scripts/run_matching.py \
    --train-dir dataset_synthetic/train --test-dir dataset_synthetic/test
python3 scripts/test_matching.py
```

The generator reproduces the noise patterns the problem statement lists, plus
singleton entities, pool-only distractors and a test-only third country. It is
for smoke-testing and unit tests only — it is easier than the real data, so its
absolute scores mean nothing. In particular an ordinary run may produce zero
contested records, leaving stage C untested end-to-end; `scripts/test_matching.py`
covers that path directly.

## Not yet implemented
- Fine-tuned bi-encoder embeddings and ANN recall channel
- Cross-encoder reranking over the top candidates
- External data requests or geocoding (prohibited by the challenge rules)
