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

**D. Set selection** (`decide.py`) — rather than one global threshold,
`expected_f05` estimates the expected F0.5 of each top-k prefix by Monte Carlo
over the calibrated probabilities and returns the best `k`. Predicting a
singleton (`k=0`) falls out of the same computation, so no separate abstain rule
is needed. `threshold` (top-1 plus a margin, tuned directly against macro F0.5)
and `top1` are kept as controls.

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
