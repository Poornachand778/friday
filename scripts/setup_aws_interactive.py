#!/usr/bin/env python3
"""
Interactive AWS Setup for Friday SageMaker Training
"""

import os
import subprocess
import boto3
from pathlib import Path


def check_aws_cli():
    """Check if AWS CLI is installed"""
    try:
        result = subprocess.run(["aws", "--version"], capture_output=True, text=True)
        print(f"✅ AWS CLI installed: {result.stdout.strip()}")
        return True
    except FileNotFoundError:
        print("❌ AWS CLI not found. Please install it first.")
        return False


def configure_aws_credentials():
    """Interactive AWS credentials setup"""
    print("\n🔐 AWS Credentials Setup")
    print("=" * 40)

    print("\nChoose your setup method:")
    print("1. AWS CLI configuration (Recommended)")
    print("2. Environment variables (.env file)")
    print("3. Skip (already configured)")

    choice = input("\nEnter choice (1-3): ").strip()

    if choice == "1":
        print("\n📝 Running 'aws configure'...")
        print("You'll need:")
        print("- AWS Access Key ID")
        print("- AWS Secret Access Key")
        print("- Default region (suggest: us-east-1)")
        print("- Output format (suggest: json)")
        print()
        subprocess.run(["aws", "configure"])

    elif choice == "2":
        setup_env_file()

    elif choice == "3":
        print("✅ Skipping AWS configuration")

    else:
        print("Invalid choice. Please run script again.")


def setup_env_file():
    """Setup .env file with AWS credentials"""
    print("\n📄 Setting up .env file...")

    # Get credentials from user
    access_key = input("AWS Access Key ID: ").strip()
    secret_key = input("AWS Secret Access Key: ").strip()
    region = input("AWS Region (default: us-east-1): ").strip() or "us-east-1"

    # Create .env file
    env_content = f"""# AWS Credentials for SageMaker
AWS_ACCESS_KEY_ID={access_key}
AWS_SECRET_ACCESS_KEY={secret_key}
AWS_DEFAULT_REGION={region}

# SageMaker Configuration  
SAGEMAKER_ROLE=arn:aws:iam::YOUR_ACCOUNT_ID:role/SageMakerExecutionRole
S3_BUCKET=friday-ai-training
S3_PREFIX=friday-finetuning

# Model Configuration
MODEL_NAME=meta-llama/Meta-Llama-3.1-8B-Instruct
INSTANCE_TYPE=ml.g5.2xlarge
TRAINING_EPOCHS=3
LORA_RANK=16
LORA_ALPHA=32
"""

    with open(".env", "w") as f:
        f.write(env_content)

    print("✅ Created .env file")
    print("📝 Edit the SAGEMAKER_ROLE with your actual AWS account ID")


def test_aws_connection():
    """Test AWS connection"""
    print("\n🧪 Testing AWS connection...")

    try:
        # Try to get caller identity
        result = subprocess.run(
            ["aws", "sts", "get-caller-identity"],
            capture_output=True,
            text=True,
            check=True,
        )
        print("✅ AWS credentials working!")
        print(f"   Account info: {result.stdout.strip()}")
        return True

    except subprocess.CalledProcessError:
        print("❌ AWS credentials not working")

        # Try with environment variables
        try:
            from dotenv import load_dotenv

            load_dotenv()

            session = boto3.Session()
            sts = session.client("sts")
            identity = sts.get_caller_identity()
            print("✅ AWS credentials working (from .env)!")
            print(f"   Account: {identity.get('Account')}")
            return True

        except Exception as e:
            print(f"❌ AWS connection failed: {e}")
            return False


def create_sagemaker_role():
    """Help create SageMaker execution role"""
    print("\n🏗️  SageMaker Role Setup")
    print("=" * 30)

    print("SageMaker needs an IAM role to access AWS resources.")
    print("You can create it via:")
    print()
    print("Option 1: AWS Console")
    print("  1. Go to IAM → Roles → Create Role")
    print("  2. Choose 'AWS Service' → 'SageMaker'")
    print("  3. Attach policies: AmazonSageMakerFullAccess, AmazonS3FullAccess")
    print("  4. Name: SageMakerExecutionRole")
    print()
    print("Option 2: AWS CLI")
    print(
        "  aws iam create-role --role-name SageMakerExecutionRole --assume-role-policy-document file://trust-policy.json"
    )
    print()

    create_role = input("Create role automatically? (y/n): ").strip().lower()

    if create_role == "y":
        try:
            # Create trust policy
            trust_policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {"Service": "sagemaker.amazonaws.com"},
                        "Action": "sts:AssumeRole",
                    }
                ],
            }

            # Save trust policy
            import json

            with open("trust-policy.json", "w") as f:
                json.dump(trust_policy, f, indent=2)

            # Create role
            subprocess.run(
                [
                    "aws",
                    "iam",
                    "create-role",
                    "--role-name",
                    "SageMakerExecutionRole",
                    "--assume-role-policy-document",
                    "file://trust-policy.json",
                ],
                check=True,
            )

            # Attach policies
            policies = [
                "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
                "arn:aws:iam::aws:policy/AmazonS3FullAccess",
            ]

            for policy in policies:
                subprocess.run(
                    [
                        "aws",
                        "iam",
                        "attach-role-policy",
                        "--role-name",
                        "SageMakerExecutionRole",
                        "--policy-arn",
                        policy,
                    ],
                    check=True,
                )

            print("✅ SageMaker role created successfully!")

            # Clean up
            os.remove("trust-policy.json")

        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create role: {e}")
            print("Please create the role manually in AWS Console")


def check_s3_bucket():
    """Check/create S3 bucket"""
    print("\n🪣 S3 Bucket Setup")
    print("=" * 20)

    bucket_name = "friday-ai-training"

    try:
        # Check if bucket exists
        result = subprocess.run(
            ["aws", "s3api", "head-bucket", "--bucket", bucket_name],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print(f"✅ S3 bucket '{bucket_name}' already exists")
        else:
            # Create bucket
            subprocess.run(
                ["aws", "s3", "mb", f"s3://{bucket_name}", "--region", "us-east-1"],
                check=True,
            )
            print(f"✅ Created S3 bucket: {bucket_name}")

    except subprocess.CalledProcessError:
        print(f"❌ Could not access/create S3 bucket: {bucket_name}")
        print("Please check AWS permissions")


def main():
    """Main setup function"""
    print("🤖 Friday AI - AWS SageMaker Setup")
    print("=" * 40)

    # Check AWS CLI
    if not check_aws_cli():
        return

    # Configure credentials
    configure_aws_credentials()

    # Test connection
    if not test_aws_connection():
        print("\n❌ Please fix AWS credentials and run again")
        return

    # Setup SageMaker role
    create_sagemaker_role()

    # Setup S3 bucket
    check_s3_bucket()

    print("\n🎉 Setup Complete!")
    print("\n📋 Next Steps:")
    print("1. python scripts/vscode_sagemaker_trainer.py  # Start training")
    print("2. Monitor progress in VS Code terminal")
    print("3. Check AWS Console for detailed logs")

    print("\n💰 Estimated Cost: $3-5 for complete training")


if __name__ == "__main__":
    main()
