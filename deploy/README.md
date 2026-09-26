# Running on AWS SageMaker

For someone who has not used AWS before. Every step is listed, including the
account setup, and nothing assumes prior knowledge.

**Why this exists:** the full test set does not fit on a laptop. The pool is
roughly 10M records; the TF-IDF matrices, the record views and the dataframe
together need ~25 GB resident, which is why the local run was killed. Sharding
bounds the *per-shard* cost but the pool structures stay loaded throughout.

**How AWS sees your data:** it does not reach into your machine. You upload the
dataset to **S3** (AWS's file storage) once; the job reads from there and
writes results back to S3; you download the results. Nothing else is shared.

```
your Mac  --upload-->  S3 bucket  --mounted-->  Processing Job (r5.4xlarge)
                            ^                            |
                            +-------- results ------------+
your Mac  <--download--  S3 bucket
```

---

## One-time setup (~20 minutes)

### 1. An AWS account

Sign up at [https://aws.amazon.com/](https://aws.amazon.com/). A card is required. Set a **billing
alert** immediately — Billing → Budgets → Create budget → a small monthly cap
with an email alert. Do this before launching anything.

### 2. An IAM user for your laptop

AWS Console → **IAM** → Users → Create user.

- name: `business-er-cli`
- *Do not* tick "Provide user access to the console"
- Permissions → Attach policies directly → tick **AmazonS3FullAccess** and
  **AmazonSageMakerFullAccess**
- Create, then open the user → Security credentials → **Create access key** →
  "Command Line Interface (CLI)" → copy the **Access key ID** and **Secret
  access key**

The secret is shown once. Treat it like a password: never commit it, never
paste it into a notebook. If it leaks, delete the key in IAM straight away.

### 3. An execution role for the job

The job itself needs a role (the job is not you; it needs its own permission to
read and write your bucket).

IAM → **Roles** → Create role → Trusted entity type **AWS service** → Use case
**SageMaker** → Next → Create. Open the new role and copy its **ARN**, which
looks like:

```
arn:aws:iam::123456789012:role/service-role/AmazonSageMaker-ExecutionRole-20260101T000001
```

### 4. The AWS CLI on your Mac

```bash
brew install awscli          # or: https://aws.amazon.com/cli/
aws configure
```

It asks for the access key, the secret, a default region (`us-east-1` is fine;
whichever you pick, stay in it) and output format (`json`).

Check it:

```bash
aws sts get-caller-identity      # prints your account id
```

### 5. A bucket and the Python SDK

Bucket names are globally unique, so add something of your own:

```bash
aws s3 mb s3://business-er-<yourname> --region us-east-1
.venv/bin/python -m pip install 'sagemaker>=2.200,<3' boto3
```

---

## Upload the dataset (once)

```bash
aws s3 sync dataset/ s3://business-er-<yourname>/business-er/dataset/
```

This is the slow step — it is your whole dataset over your home connection.
`sync` is resumable: re-run it if it drops. Verify:

```bash
aws s3 ls --recursive --human-readable --summarize \
    s3://business-er-<yourname>/business-er/dataset/ | tail -5
```

You need `dataset/train/` and `dataset/test/` underneath that prefix. The
`_processed.parquet` files are used in preference to the TSVs when present, and
are smaller to upload.

---

## Run

### First: a smoke run

Never start with the full job. This one trains, runs **a single shard**, and
prints a projection for the whole run — so the full job is sized from a
measurement rather than a guess, and any mistake costs minutes.

```bash
.venv/bin/python deploy/sagemaker_run.py \
    --bucket business-er-<yourname> \
    --role-arn arn:aws:iam::123456789012:role/service-role/AmazonSageMaker-... \
    --team-name your_team \
    --mode smoke --wait
```

`--wait` streams the logs. Without it the job runs on regardless of your
terminal, and you follow it in the console link the script prints.

Look for:

```
  one shard took 42.0s over 40 shards
  projected full inference: 28.0 min (0.47 h)
```

### Then: the full run

```bash
.venv/bin/python deploy/sagemaker_run.py \
    --bucket business-er-<yourname> \
    --role-arn arn:... \
    --team-name your_team \
    --mode full --shards 40 --wait
```

### Fetch the results

```bash
aws s3 sync s3://business-er-<yourname>/business-er/runs/<job-name>/ ./cloud_output
```

You get `matching_results.tsv` (upload this to the portal),
`candidate_pairs.tsv`, `dist/<team>_submission.zip`, and `reports/` with every
log and metric.

---

## Choosing the instance

`ml.r5.4xlarge` (16 vCPU, **128 GB**) is the default, and the memory is the
reason. Measured/derived at full scale:

|                        |                         |
| ---------------------- | ----------------------- |
| TF-IDF pool matrices   | ~11 GB                  |
| pool dataframe         | ~5 GB                   |
| per-shard record views | scales with`--shards` |

A compute-optimised `c5` of the same price has a quarter of the memory and will
die the same way your laptop did. If the smoke run reports plenty of headroom,
`ml.r5.2xlarge` (64 GB) is cheaper; `--shards 80` halves the per-shard part.

Processing Jobs bill **per second while running** and stop on their own. Check
current rates on the SageMaker pricing page before launching — and keep
`--max-runtime-hours` set (default 24) so a hung job cannot bill indefinitely.

---

## If something goes wrong

| Symptom                         | Cause                                                                                                                       |
| ------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `AccessDenied` on S3          | the execution role lacks bucket access — attach**AmazonS3FullAccess** to the *role* from step 3, not just the user |
| `ResourceLimitExceeded`       | your account has no quota for that instance type; request an increase in Service Quotas, or try a smaller one               |
| job fails in "Data consistency" | the uploaded training data is not label-consistent; fix it locally with`scripts/make_subsample.py` before re-uploading    |
| killed / out of memory          | raise`--shards`, then move to a larger `r5`                                                                             |
| a shard failed mid-run          | just re-run; finished shards are on the job's volume only, so prefer`--shards` small enough that a rerun is cheap         |

Every log is in `reports/` in the output, and also in CloudWatch via the
console link the launcher prints.

---

## Testing without AWS

The entrypoint takes its paths from the environment, so the exact cloud path
runs locally:

```bash
mkdir -p /tmp/sm/{input,code,output}
cp -r dataset/train dataset/test /tmp/sm/input/
python3 deploy/sagemaker_run.py --bucket b --role-arn arn:x --team-name t --dry-run
cp dist/repo.tar.gz /tmp/sm/code/

INPUT_DIR=/tmp/sm/input CODE_DIR=/tmp/sm/code OUTPUT_DIR=/tmp/sm/output \
CACHE_DIR=/tmp/sm/cache MODE=smoke TEAM_NAME=t SHARDS=8 \
    ./deploy/entrypoint.sh
```

That is how this was verified end to end before any job was launched: smoke and
full modes both produce a validated submission package.
