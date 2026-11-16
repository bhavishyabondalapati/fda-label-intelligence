# UNUSED: kept in case SageMaker is retried

Phase 4 training moved to Kaggle (see `PLAN.md`, amendment 2026-11-16) because the `ml.g4dn.xlarge` training quota was denied.
Nothing in this folder was ever applied, and no AWS resource exists for it. If you retry SageMaker after the next billing cycle,
request the quota again, then `terraform apply` here and run `python sagemaker/launch_job.py` (dry run first).
