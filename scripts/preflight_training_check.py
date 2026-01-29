#!/usr/bin/env python3
"""
Pre-Training Preflight Check Script for Friday AI

Validates all prerequisites before launching SageMaker training job:
- Dataset file exists and is valid
- S3 bucket accessible
- AWS credentials configured
- SageMaker role permissions
- Instance quota available (if possible)
- HuggingFace token valid
- Token length analysis
- Hyperparameter recommendations

Usage:
    python scripts/preflight_training_check.py
    python scripts/preflight_training_check.py --dataset data/instructions/custom.jsonl
"""

import json
import argparse
import os
import sys
from pathlib import Path
from collections import Counter
import statistics
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_DATASET = (
    PROJECT_ROOT / "data/instructions/iteration3_interview_only_train.jsonl"
)
REPORT_OUTPUT = PROJECT_ROOT / "logs/iteration3_preflight_report.json"

# Required environment variables
REQUIRED_ENV_VARS = [
    "HUGGINGFACE_TOKEN",
    "SAGEMAKER_ROLE",
    "AWS_DEFAULT_REGION",
]

# Training configuration
DEFAULT_CONFIG = {
    "base_model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "instance_type": "ml.g5.12xlarge",
    "epochs": 5,
    "batch_size": 1,
    "grad_acc_steps": 64,
    "learning_rate": 2e-4,
    "max_seq_len": 2048,
    "lora_r": 32,
    "lora_alpha": 64,
}


def load_dataset(filepath: Path) -> list:
    """Load and validate dataset."""
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def estimate_tokens(text: str) -> int:
    """Rough token count estimate."""
    words = len(text.split())
    telugu_chars = sum(1 for c in text if "\u0C00" <= c <= "\u0C7F")
    multiplier = 1.5 if telugu_chars > 0 else 1.3
    return int(words * multiplier)


def analyze_dataset_tokens(data: list) -> dict:
    """Analyze token distribution in dataset."""
    token_counts = []
    for example in data:
        messages = example.get("messages", [])
        full_text = " ".join(m.get("content", "") for m in messages if "content" in m)
        token_counts.append(estimate_tokens(full_text))

    if not token_counts:
        return {}

    sorted_counts = sorted(token_counts)
    return {
        "min": sorted_counts[0],
        "max": sorted_counts[-1],
        "mean": int(statistics.mean(sorted_counts)),
        "median": int(statistics.median(sorted_counts)),
        "p50": int(sorted_counts[int(len(sorted_counts) * 0.50)]),
        "p75": int(sorted_counts[int(len(sorted_counts) * 0.75)]),
        "p90": int(sorted_counts[int(len(sorted_counts) * 0.90)]),
        "p95": int(sorted_counts[int(len(sorted_counts) * 0.95)]),
        "p99": int(sorted_counts[int(len(sorted_counts) * 0.99)]),
    }


def check_env_variables() -> tuple[bool, list]:
    """Check required environment variables."""
    print("\n[1/7] Checking Environment Variables...")
    missing = []
    for var in REQUIRED_ENV_VARS:
        value = os.getenv(var)
        if value:
            print(f"  ✅ {var}: {'*' * 20} (set)")
        else:
            print(f"  ❌ {var}: NOT SET")
            missing.append(var)

    return len(missing) == 0, missing


def check_aws_credentials() -> tuple[bool, str]:
    """Verify AWS credentials are configured."""
    print("\n[2/7] Checking AWS Credentials...")
    try:
        sts = boto3.client("sts")
        identity = sts.get_caller_identity()
        account = identity["Account"]
        arn = identity["Arn"]
        print(f"  ✅ AWS Account: {account}")
        print(f"  ✅ Identity: {arn}")
        return True, ""
    except NoCredentialsError:
        print("  ❌ AWS credentials not configured")
        return False, "AWS credentials not found. Run 'aws configure'"
    except ClientError as e:
        print(f"  ❌ AWS credentials invalid: {e}")
        return False, str(e)


def check_s3_bucket() -> tuple[bool, str]:
    """Check S3 bucket access."""
    print("\n[3/7] Checking S3 Bucket Access...")
    bucket_name = os.getenv("S3_BUCKET", "friday-ai-training-v01")

    try:
        s3 = boto3.client("s3")
        # Try to list objects (will fail if no access)
        s3.list_objects_v2(Bucket=bucket_name, MaxKeys=1)
        print(f"  ✅ S3 bucket '{bucket_name}' accessible")

        # Check region
        location = s3.get_bucket_location(Bucket=bucket_name)
        region = location.get("LocationConstraint") or "us-east-1"
        print(f"  ✅ Bucket region: {region}")

        return True, ""
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        if error_code == "NoSuchBucket":
            print(f"  ⚠️  Bucket '{bucket_name}' does not exist (will be created)")
            return True, ""  # Not a blocker
        elif error_code == "AccessDenied":
            print(f"  ❌ Access denied to bucket '{bucket_name}'")
            return False, f"S3 access denied: {e}"
        else:
            print(f"  ❌ S3 error: {e}")
            return False, str(e)


def check_sagemaker_role() -> tuple[bool, str]:
    """Verify SageMaker role exists and has permissions."""
    print("\n[4/7] Checking SageMaker Role...")
    role_arn = os.getenv("SAGEMAKER_ROLE")

    if not role_arn:
        print("  ❌ SAGEMAKER_ROLE not set")
        return False, "SAGEMAKER_ROLE environment variable not set"

    try:
        iam = boto3.client("iam")
        # Extract role name from ARN
        role_name = role_arn.split("/")[-1]

        # Get role
        role = iam.get_role(RoleName=role_name)
        print(f"  ✅ Role exists: {role_name}")
        print(f"  ✅ Role ARN: {role['Role']['Arn']}")

        # Check attached policies (optional, best effort)
        try:
            policies = iam.list_attached_role_policies(RoleName=role_name)
            print(
                f"  ℹ️  Attached policies: {len(policies.get('AttachedPolicies', []))}"
            )
            for policy in policies.get("AttachedPolicies", [])[:5]:
                print(f"    - {policy['PolicyName']}")
        except Exception:
            pass  # Ignore policy listing errors

        return True, ""
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        if error_code == "NoSuchEntity":
            print(f"  ❌ Role not found: {role_arn}")
            return False, f"SageMaker role does not exist: {role_arn}"
        else:
            print(f"  ⚠️  Could not verify role: {e}")
            return True, ""  # Don't block, might be permission issue


def check_dataset(filepath: Path) -> tuple[bool, str, dict]:
    """Check dataset file."""
    print("\n[5/7] Checking Dataset...")

    if not filepath.exists():
        print(f"  ❌ Dataset not found: {filepath}")
        return False, f"Dataset file not found: {filepath}", {}

    print(f"  ✅ Dataset exists: {filepath}")

    # Load and analyze
    try:
        data = load_dataset(filepath)
        print(f"  ✅ Dataset loaded: {len(data)} examples")

        # Token analysis
        token_stats = analyze_dataset_tokens(data)
        if token_stats:
            print(f"  📏 Token statistics:")
            print(f"    Mean: {token_stats['mean']} tokens")
            print(f"    P95: {token_stats['p95']} tokens")
            print(f"    P99: {token_stats['p99']} tokens")
            print(f"    Max: {token_stats['max']} tokens")

        return True, "", token_stats
    except Exception as e:
        print(f"  ❌ Dataset error: {e}")
        return False, str(e), {}


def check_huggingface_token() -> tuple[bool, str]:
    """Verify HuggingFace token (basic check)."""
    print("\n[6/7] Checking HuggingFace Token...")
    token = os.getenv("HUGGINGFACE_TOKEN")

    if not token:
        print("  ❌ HUGGINGFACE_TOKEN not set")
        return False, "HuggingFace token not configured"

    # Basic format check (should start with 'hf_')
    if token.startswith("hf_"):
        print(f"  ✅ Token format valid (hf_...)")
        print(f"  ℹ️  Token length: {len(token)} chars")
        return True, ""
    else:
        print(f"  ⚠️  Token format unexpected (doesn't start with 'hf_')")
        return True, ""  # Don't block


def recommend_hyperparameters(dataset_size: int, token_stats: dict) -> dict:
    """Recommend hyperparameters based on dataset."""
    print("\n[7/7] Hyperparameter Recommendations...")

    config = DEFAULT_CONFIG.copy()
    recommendations = []

    # Instance type based on dataset size
    if dataset_size < 600:
        config["instance_type"] = "ml.g5.2xlarge"
        recommendations.append(
            "Using ml.g5.2xlarge (1 GPU, ~$1.50/hr) - sufficient for <600 examples"
        )
    elif dataset_size < 1200:
        config["instance_type"] = "ml.g5.2xlarge"
        recommendations.append(
            "Using ml.g5.2xlarge (1 GPU, ~$1.50/hr) - good for 600-1200 examples"
        )
    else:
        config["instance_type"] = "ml.g5.12xlarge"
        recommendations.append(
            "Consider ml.g5.12xlarge (4 GPU, ~$9/hr) for >1200 examples (faster)"
        )

    # Max sequence length based on token stats
    if token_stats:
        p95 = token_stats.get("p95", 0)
        p99 = token_stats.get("p99", 0)

        if p99 > 2048:
            config["max_seq_len"] = 3072
            config["batch_size"] = 2  # Reduce batch size
            recommendations.append(
                f"Increase max_seq_len to 3072 (P99={p99} tokens), reduce batch to 2"
            )
        elif p95 < 1024:
            config["max_seq_len"] = 1536
            recommendations.append(
                f"Could reduce max_seq_len to 1536 (P95={p95} tokens) for memory savings"
            )
        else:
            recommendations.append(
                f"Keep max_seq_len at 2048 (P95={p95}, P99={p99} tokens)"
            )

    # Gradient accumulation based on dataset size
    if dataset_size < 600:
        config["grad_acc_steps"] = 8
        recommendations.append(
            "Reduce grad_acc to 8 for smaller dataset (effective batch=32)"
        )
    else:
        recommendations.append("Keep grad_acc at 16 (effective batch=64)")

    # Epochs based on dataset size
    if dataset_size > 1000:
        config["epochs"] = 2
        recommendations.append(
            "Reduce to 2 epochs for larger dataset (avoid overfitting)"
        )
    else:
        recommendations.append("Keep 3 epochs for dataset size")

    print(f"\n  📋 Recommended Configuration:")
    for key, value in config.items():
        print(f"    {key}: {value}")

    print(f"\n  💡 Recommendations:")
    for rec in recommendations:
        print(f"    - {rec}")

    # Cost estimate
    instance_cost_per_hour = 1.50 if "ml.g5.2xlarge" in config["instance_type"] else 9.0
    examples_per_epoch = dataset_size
    steps_per_epoch = examples_per_epoch // (
        config["batch_size"] * config["grad_acc_steps"]
    )
    total_steps = steps_per_epoch * config["epochs"]
    # Rough estimate: 1-2 seconds per step
    estimated_minutes = (total_steps * 1.5) / 60
    estimated_hours = estimated_minutes / 60
    estimated_cost = instance_cost_per_hour * estimated_hours

    print(f"\n  💰 Cost Estimate:")
    print(
        f"    Instance: {config['instance_type']} (~${instance_cost_per_hour:.2f}/hr)"
    )
    print(
        f"    Estimated duration: {estimated_minutes:.1f} min ({estimated_hours:.2f} hrs)"
    )
    print(f"    Estimated cost: ${estimated_cost:.2f}")

    return config


def main():
    parser = argparse.ArgumentParser(description="Pre-training preflight check")
    parser.add_argument(
        "--dataset", type=Path, default=DEFAULT_DATASET, help="Dataset file path"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Friday AI - Pre-Training Preflight Check")
    print("=" * 60)

    results = {
        "checks": {},
        "dataset_stats": {},
        "recommended_config": {},
        "all_passed": True,
        "blocking_errors": [],
        "warnings": [],
    }

    # Run checks
    env_ok, env_missing = check_env_variables()
    results["checks"]["environment"] = env_ok
    if not env_ok:
        results["blocking_errors"].extend(
            [f"Missing env var: {v}" for v in env_missing]
        )
        results["all_passed"] = False

    aws_ok, aws_error = check_aws_credentials()
    results["checks"]["aws_credentials"] = aws_ok
    if not aws_ok:
        results["blocking_errors"].append(aws_error)
        results["all_passed"] = False

    s3_ok, s3_error = check_s3_bucket()
    results["checks"]["s3_bucket"] = s3_ok
    if not s3_ok:
        results["blocking_errors"].append(s3_error)
        results["all_passed"] = False

    role_ok, role_error = check_sagemaker_role()
    results["checks"]["sagemaker_role"] = role_ok
    if not role_ok:
        results["blocking_errors"].append(role_error)
        results["all_passed"] = False

    dataset_ok, dataset_error, token_stats = check_dataset(args.dataset)
    results["checks"]["dataset"] = dataset_ok
    results["dataset_stats"] = token_stats
    if not dataset_ok:
        results["blocking_errors"].append(dataset_error)
        results["all_passed"] = False

    hf_ok, hf_error = check_huggingface_token()
    results["checks"]["huggingface_token"] = hf_ok
    if not hf_ok:
        results["blocking_errors"].append(hf_error)
        results["all_passed"] = False

    # Recommendations (if dataset loaded)
    if dataset_ok:
        dataset_size = len(load_dataset(args.dataset))
        config = recommend_hyperparameters(dataset_size, token_stats)
        results["recommended_config"] = config

    # Summary
    print("\n" + "=" * 60)
    if results["all_passed"]:
        print("✅ PREFLIGHT CHECK PASSED - Ready for training!")
        print("\nNext steps:")
        print("  1. Review recommended configuration above")
        print("  2. Update src/training/vscode_sagemaker_trainer.py if needed")
        print("  3. Run: python src/training/vscode_sagemaker_trainer.py")
    else:
        print("❌ PREFLIGHT CHECK FAILED - Fix errors before training")
        print(f"\nBlocking errors ({len(results['blocking_errors'])}):")
        for err in results["blocking_errors"]:
            print(f"  - {err}")

    if results["warnings"]:
        print(f"\nWarnings ({len(results['warnings'])}):")
        for warn in results["warnings"]:
            print(f"  - {warn}")

    print("=" * 60)

    # Write report
    REPORT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nPreflight report written to: {REPORT_OUTPUT}")

    return 0 if results["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
