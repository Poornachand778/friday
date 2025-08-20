#!/usr/bin/env python3
"""
Quick status checker for Friday AI SageMaker endpoint
"""

import boto3


def check_endpoint_status():
    """Check the status of Friday AI endpoint"""
    try:
        sagemaker = boto3.client("sagemaker", region_name="us-east-1")
        endpoint_name = "friday-rt"

        print(f"🔍 Checking endpoint: {endpoint_name}")
        print("=" * 50)

        try:
            response = sagemaker.describe_endpoint(EndpointName=endpoint_name)
            status = response["EndpointStatus"]
            creation_time = response.get("CreationTime", "Unknown")
            last_modified = response.get("LastModifiedTime", "Unknown")

            # Status with emoji
            status_emoji = {
                "Creating": "🔄",
                "InService": "✅",
                "Failed": "❌",
                "Updating": "⏳",
                "Deleting": "🗑️",
            }.get(status, "❓")

            print(f"Status: {status_emoji} {status}")
            print(f"Created: {creation_time}")
            print(f"Modified: {last_modified}")

            if "FailureReason" in response:
                print(f"❌ Failure Reason: {response['FailureReason']}")

        except sagemaker.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "ValidationException":
                print("📝 Endpoint not found - may still be creating")
            else:
                print(f"❌ Error checking endpoint: {e}")

        # Check if model exists
        try:
            sagemaker.describe_model(ModelName="friday-model-20250816")
            print("📦 Model Status: ✅ Created")
        except Exception:
            print("📦 Model Status: ⏳ Creating or Not Found")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    check_endpoint_status()
