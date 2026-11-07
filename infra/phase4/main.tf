# Phase 4 AWS setup: one private S3 bucket and one SageMaker training role. NOT APPLIED (see README).
# An empty bucket and a role cost nothing; the only paid thing in Phase 4 is the training job itself,
# which is created by sagemaker/launch_job.py and only after the GPU quota is approved.
terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
}

provider "aws" {
  region  = "us-east-2"   # us-east-2 only
  profile = "sky"
}

data "aws_caller_identity" "me" {}

locals {
  bucket = "fda-label-intelligence-${data.aws_caller_identity.me.account_id}-use2"
}

resource "aws_s3_bucket" "data" {
  bucket = local.bucket
}

resource "aws_s3_bucket_public_access_block" "data" {
  bucket                  = aws_s3_bucket.data.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data" {
  bucket = aws_s3_bucket.data.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "AES256" }
  }
}

# Training data and adapters are small; delete everything after 30 days so nothing lingers.
resource "aws_s3_bucket_lifecycle_configuration" "data" {
  bucket = aws_s3_bucket.data.id
  rule {
    id     = "expire-30d"
    status = "Enabled"
    filter {}
    expiration { days = 30 }
  }
}

data "aws_iam_policy_document" "trust" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }
    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [data.aws_caller_identity.me.account_id]
    }
  }
}

resource "aws_iam_role" "train" {
  name               = "fda-label-sagemaker-train"
  assume_role_policy = data.aws_iam_policy_document.trust.json
}

# Least privilege: this bucket only, the training job's own log group, and pulling the AWS-published training image.
data "aws_iam_policy_document" "train" {
  statement {
    actions   = ["s3:GetObject", "s3:PutObject"]
    resources = ["${aws_s3_bucket.data.arn}/*"]
  }
  statement {
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.data.arn]
  }
  statement {
    actions   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams"]
    resources = ["arn:aws:logs:us-east-2:${data.aws_caller_identity.me.account_id}:log-group:/aws/sagemaker/TrainingJobs*"]
  }
  statement {
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }
  statement {
    actions   = ["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer", "ecr:BatchCheckLayerAvailability"]
    resources = ["arn:aws:ecr:us-east-2:763104351884:repository/pytorch-training"]
  }
}

resource "aws_iam_role_policy" "train" {
  name   = "train-access"
  role   = aws_iam_role.train.id
  policy = data.aws_iam_policy_document.train.json
}

output "bucket" { value = aws_s3_bucket.data.bucket }
output "role_arn" { value = aws_iam_role.train.arn }
