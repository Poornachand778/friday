#!/usr/bin/env python3
"""
Setup AWS Secrets Manager for Friday AI deployment
Creates HuggingFace token secret for secure model access
"""

import boto3
import json
import os
import argparse
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()


def create_hf_token_secret(
    secret_name: str = "friday-ai/hf-token",
    region: str = "us-east-1",
    hf_token: str = None,
) -> bool:
    """Create HuggingFace token secret in AWS Secrets Manager"""

    if not hf_token:
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if not hf_token:
            print("❌ HuggingFace token not provided")
            print("   Set HUGGINGFACE_TOKEN env var or use --token flag")
            return False

    # Initialize Secrets Manager client
    secrets_client = boto3.client("secretsmanager", region_name=region)

    try:
        # Check if secret already exists
        try:
            secrets_client.describe_secret(SecretId=secret_name)
            print(f"🔄 Updating existing secret: {secret_name}")

            # Update the secret
            secrets_client.update_secret(
                SecretId=secret_name, SecretString=json.dumps({"HF_TOKEN": hf_token})
            )

        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceNotFoundException":
                print(f"🆕 Creating new secret: {secret_name}")

                # Create new secret
                secrets_client.create_secret(
                    Name=secret_name,
                    SecretString=json.dumps({"HF_TOKEN": hf_token}),
                    Description="HuggingFace access token for Friday AI model deployment",
                )
            else:
                raise e

        # Verify the secret was created/updated
        response = secrets_client.get_secret_value(SecretId=secret_name)
        stored_token = json.loads(response["SecretString"]).get("HF_TOKEN")

        if stored_token == hf_token:
            print("✅ HuggingFace token successfully stored")
            print(f"   Secret ARN: {response['ARN']}")
            print(f"   Token length: {len(hf_token)} characters")
            print(f"   Token prefix: {hf_token[:10]}...")
            return True
        else:
            print("❌ Token verification failed")
            return False

    except Exception as e:
        print(f"❌ Failed to create secret: {e}")
        return False


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(description="Setup Friday AI secrets in AWS")
    parser.add_argument(
        "--token", help="HuggingFace token (or set HUGGINGFACE_TOKEN env var)"
    )
    parser.add_argument(
        "--secret-name", default="friday-ai/hf-token", help="Secret name"
    )
    parser.add_argument("--region", default="us-east-1", help="AWS region")

    args = parser.parse_args()

    print("🔐 Friday AI Secrets Setup")
    print("=" * 40)

    success = create_hf_token_secret(
        secret_name=args.secret_name, region=args.region, hf_token=args.token
    )

    if success:
        print("\n✅ Setup complete!")
        print("🚀 Ready to deploy Friday AI endpoint")
    else:
        print("\n❌ Setup failed")
        exit(1)


if __name__ == "__main__":
    main()
