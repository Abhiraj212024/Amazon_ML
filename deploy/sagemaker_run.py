"""
Launch the pipeline as a SageMaker Processing Job.

A Processing Job suits this better than a notebook: it is batch, it shuts
itself down when the work finishes (so an idle instance cannot quietly bill),
and it stages data in from S3 and results back out without any code of ours.

    # 1. prove the whole path cheaply, and measure one shard
    python3 deploy/sagemaker_run.py --bucket my-bucket --role-arn arn:... \
        --team-name my_team --mode smoke

    # 2. the real run, sized from what the smoke run measured
    python3 deploy/sagemaker_run.py --bucket my-bucket --role-arn arn:... \
        --team-name my_team --mode full --shards 40

See deploy/README.md for getting an account, a role and a bucket in the first
place.
"""
import argparse
import os
import subprocess
import sys
import tarfile
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Everything the job needs, and nothing else: data, caches and git history are
# excluded so the upload stays small and no local data leaks into the job.
INCLUDE = ["src", "scripts", "utils", "docs", "requirements.txt"]
EXCLUDE_SUFFIXES = (".pyc", ".parquet", ".tsv", ".csv", ".zip", ".pkl", ".npy")
EXCLUDE_DIRS = {"__pycache__", ".git", ".cache", "dataset", "output", "dist", "reports"}


def _filter(info):
    name = os.path.basename(info.name)
    parts = set(info.name.split(os.sep))
    if parts & EXCLUDE_DIRS or name.endswith(EXCLUDE_SUFFIXES) or name == ".DS_Store":
        return None
    return info


def build_code_archive(destination):
    with tarfile.open(destination, "w:gz") as tar:
        for entry in INCLUDE:
            path = os.path.join(REPO_ROOT, entry)
            if os.path.exists(path):
                tar.add(path, arcname=entry, filter=_filter)
    return destination


def _s3_uri(bucket, prefix, *parts):
    pieces = [prefix.strip("/")] + [p.strip("/") for p in parts if p]
    return f"s3://{bucket}/" + "/".join(p for p in pieces if p)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bucket", required=True, help="S3 bucket you own")
    parser.add_argument("--role-arn", required=True,
                        help="SageMaker execution role ARN (see deploy/README.md)")
    parser.add_argument("--team-name", required=True)
    parser.add_argument("--prefix", default="business-er")
    parser.add_argument("--region", default=None)
    parser.add_argument("--mode", default="smoke", choices=("smoke", "full"))
    parser.add_argument("--instance-type", default="ml.r5.4xlarge",
                        help="memory-optimised on purpose: memory binds before CPU")
    parser.add_argument("--volume-size-gb", type=int, default=200)
    parser.add_argument("--shards", type=int, default=40)
    parser.add_argument("--max-fit-entities", type=int, default=150_000)
    parser.add_argument("--channels", default=None)
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--extra-train-flags", default="")
    parser.add_argument("--max-runtime-hours", type=float, default=24.0)
    parser.add_argument("--data-uri", default=None,
                        help="existing s3:// path holding train/ and test/; "
                             "defaults to <bucket>/<prefix>/dataset")
    parser.add_argument("--wait", action="store_true",
                        help="stream logs until the job finishes")
    parser.add_argument("--dry-run", action="store_true",
                        help="build and show everything without launching")
    args = parser.parse_args()

    # Packaging first, and without importing the AWS SDK, so --dry-run works
    # before anything is installed or configured. Getting the archive right is
    # the part worth checking locally.
    data_uri = args.data_uri or _s3_uri(args.bucket, args.prefix, "dataset")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    job_name = f"business-er-{args.mode}-{stamp}"
    output_uri = _s3_uri(args.bucket, args.prefix, "runs", job_name)
    code_uri = _s3_uri(args.bucket, args.prefix, "code", job_name)

    archive = os.path.join(REPO_ROOT, "dist", "repo.tar.gz")
    os.makedirs(os.path.dirname(archive), exist_ok=True)
    build_code_archive(archive)
    size_mb = os.path.getsize(archive) / 1e6
    print(f"mode   : {args.mode}\nbucket : {args.bucket}")
    print(f"code   : {archive} ({size_mb:.1f} MB)")
    if size_mb > 200:
        print("  WARNING: that is large for source only - check EXCLUDE_DIRS")
    print(f"\ninputs\n  data : {data_uri}\n  code : {code_uri}\noutput\n  {output_uri}")

    if args.dry_run:
        print("\n--dry-run: archive built, nothing uploaded or launched")
        return

    try:
        import boto3
        import sagemaker
        from sagemaker.processing import ProcessingInput, ProcessingOutput
        from sagemaker.sklearn.processing import SKLearnProcessor
    except ImportError as error:
        sys.exit(
            f"missing or incompatible dependency: {error}\n\n"
            "  use the project interpreter and SageMaker SDK v2:\n"
            "  .venv/bin/python -m pip install 'sagemaker>=2.200,<3' boto3"
        )

    session = sagemaker.Session(boto3.Session(region_name=args.region)) \
        if args.region else sagemaker.Session()
    region = session.boto_region_name
    print(f"region : {region}")

    subprocess.run(["aws", "s3", "cp", archive, f"{code_uri}/repo.tar.gz"], check=True)

    environment = {
        "MODE": args.mode,
        "TEAM_NAME": args.team_name,
        "SHARDS": str(args.shards),
        "MAX_FIT_ENTITIES": str(args.max_fit_entities),
        "MAX_CANDIDATES": str(args.max_candidates),
        "EXTRA_TRAIN_FLAGS": args.extra_train_flags,
    }
    if args.channels:
        environment["CHANNELS"] = args.channels

    processor = SKLearnProcessor(
        framework_version="1.2-1",
        role=args.role_arn,
        instance_type=args.instance_type,
        instance_count=1,
        volume_size_in_gb=args.volume_size_gb,
        max_runtime_in_seconds=int(args.max_runtime_hours * 3600),
        base_job_name=f"business-er-{args.mode}",
        env=environment,
        sagemaker_session=session,
    )

    print(f"\nlaunching {job_name} on {args.instance_type} ...")
    processor.run(
        code=os.path.join(os.path.dirname(__file__), "entrypoint.sh"),
        inputs=[
            ProcessingInput(source=data_uri, destination="/opt/ml/processing/input",
                            input_name="dataset"),
            ProcessingInput(source=code_uri, destination="/opt/ml/processing/code",
                            input_name="code"),
        ],
        outputs=[
            ProcessingOutput(source="/opt/ml/processing/output", destination=output_uri,
                             output_name="submission"),
        ],
        job_name=job_name,
        wait=args.wait,
        logs=args.wait,
    )

    print(f"\njob      : {job_name}")
    print(f"console  : https://{region}.console.aws.amazon.com/sagemaker/home"
          f"?region={region}#/processing-jobs/{job_name}")
    print(f"\nwhen it finishes:\n  aws s3 sync {output_uri} ./cloud_output")
    if not args.wait:
        print("\nthe job keeps running if you close this terminal; follow it with")
        print(f"  aws sagemaker describe-processing-job --processing-job-name {job_name} "
              f"--query ProcessingJobStatus")


if __name__ == "__main__":
    main()
