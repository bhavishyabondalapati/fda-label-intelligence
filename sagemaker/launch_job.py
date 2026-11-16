"""UNUSED since 2026-11-16: the GPU quota was denied and Phase 4 trains on Kaggle (kaggle/). Kept in case SageMaker is retried.

Create the Phase 4 SageMaker training job. DRY RUN by default: prints the request and the worst-case cost, creates nothing.

To really launch (only after the ml.g4dn.xlarge training quota is approved and infra/phase4 was applied):
    python sagemaker/launch_job.py --launch
Guards: on-demand only (no spot), 1 instance, max_run fixed, worst-case cost must be under $1.50, and the quota is
re-checked from AWS before anything is created.
"""
import argparse
import io
import json
import tarfile
import time
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parent.parent
REGION, PROFILE = "us-east-2", "sky"
INSTANCE, PRICE_PER_HOUR = "ml.g4dn.xlarge", 0.736          # Price List API, 2026-11-06, USE2-Train:ml.g4dn.xlarge
MAX_RUN, EXTRA_BILLED = 6600, 600                            # seconds: hard stop, and allowance for start-up/download
COST_LIMIT = 1.50
IMAGE = ("763104351884.dkr.ecr.us-east-2.amazonaws.com/pytorch-training:"
         "2.8.0-gpu-py312-cu129-ubuntu22.04-sagemaker-v1.35-soci")
TRAIN_ARGS = {"quant": "4bit", "device": "cuda", "dtype": "fp16", "time-budget": "6000"}


def worst_case_cost():
    return (MAX_RUN + EXTRA_BILLED) / 3600 * PRICE_PER_HOUR


def quota_value(sess):
    """Applied quota for 'ml.g4dn.xlarge for training job usage' (0 or missing means not approved)."""
    sq = sess.client("service-quotas")
    for page in sq.get_paginator("list_service_quotas").paginate(ServiceCode="sagemaker"):
        for q in page["Quotas"]:
            if q["QuotaName"] == "ml.g4dn.xlarge for training job usage":
                return q["Value"]
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--launch", action="store_true")
    a = ap.parse_args()
    assert worst_case_cost() < COST_LIMIT, "worst-case cost is over the limit"
    sess = boto3.Session(profile_name=PROFILE, region_name=REGION)
    account = sess.client("sts").get_caller_identity()["Account"]
    bucket = f"fda-label-intelligence-{account}-use2"
    name = "fda-qlora-" + time.strftime("%Y%m%d-%H%M%S")
    prefix = f"s3://{bucket}/phase4"
    request = {
        "TrainingJobName": name,
        "RoleArn": f"arn:aws:iam::{account}:role/fda-label-sagemaker-train",
        "AlgorithmSpecification": {"TrainingImage": IMAGE, "TrainingInputMode": "File"},
        "ResourceConfig": {"InstanceType": INSTANCE, "InstanceCount": 1, "VolumeSizeInGB": 50},
        "StoppingCondition": {"MaxRuntimeInSeconds": MAX_RUN},
        "EnableManagedSpotTraining": False,
        "InputDataConfig": [{"ChannelName": "train", "DataSource": {"S3DataSource": {
            "S3DataType": "S3Prefix", "S3Uri": f"{prefix}/data/", "S3DataDistributionType": "FullyReplicated"}}}],
        "OutputDataConfig": {"S3OutputPath": f"{prefix}/output/"},
        "HyperParameters": {"sagemaker_program": json.dumps("train.py"),
                            "sagemaker_submit_directory": json.dumps(f"{prefix}/source/sourcedir.tar.gz"),
                            "sagemaker_region": json.dumps(REGION), **TRAIN_ARGS},
        "Tags": [{"Key": "project", "Value": "fda-label-intelligence"}],
    }
    print(json.dumps(request, indent=2))
    print(f"\nworst-case cost: ({MAX_RUN}+{EXTRA_BILLED}) s x ${PRICE_PER_HOUR}/h = ${worst_case_cost():.2f} (limit ${COST_LIMIT})")
    if not a.launch:
        print("DRY RUN: nothing created.")
        return
    q = quota_value(sess)
    if q < 1:
        raise SystemExit(f"quota for {INSTANCE} training is {q}: not approved yet, not launching")
    s3 = sess.client("s3")
    buf = io.BytesIO()                                   # source bundle: train.py + requirements.txt
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for f in ("train.py", "requirements.txt"):
            t.add(ROOT / "sagemaker" / f, arcname=f)
    s3.put_object(Bucket=bucket, Key="phase4/source/sourcedir.tar.gz", Body=buf.getvalue())
    for f in ("train.jsonl", "val.jsonl"):
        s3.upload_file(str(ROOT / "data" / "finetune" / f), bucket, f"phase4/data/{f}")
    sess.client("sagemaker").create_training_job(**request)
    print("launched", name)


if __name__ == "__main__":
    main()
