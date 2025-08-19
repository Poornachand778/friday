#!/usr/bin/env python3
"""
Fix Autoscaling for Friday AI Endpoint
Standalone script to set up autoscaling after endpoint is InService
"""

import boto3
import time
import os
from dotenv import load_dotenv
from botocore.exceptions import ClientError

load_dotenv()


def fix_autoscaling():
    """Fix autoscaling for the Friday AI endpoint"""

    region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    endpoint_name = "friday-rt"

    session = boto3.Session(region_name=region)
    sagemaker_client = session.client("sagemaker")
    autoscaling_client = session.client("application-autoscaling")

    print(f"🔧 Fixing autoscaling for endpoint: {endpoint_name}")
    print("=" * 50)

    # Step 1: Verify endpoint is InService
    print("🔍 Checking endpoint status...")
    try:
        response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        status = response["EndpointStatus"]
        print(f"   Status: {status}")

        if status != "InService":
            print(f"❌ Endpoint is {status}, not InService. Cannot set up autoscaling.")
            print("   Wait for endpoint to reach InService status first.")
            return False

        print("✅ Endpoint is InService, proceeding with autoscaling setup")

    except ClientError as e:
        print(f"❌ Error checking endpoint: {e}")
        return False

    # Step 2: Set up autoscaling
    resource_id = f"endpoint/{endpoint_name}/variant/friday-variant"

    try:
        # Register scalable target
        print("📋 Registering scalable target...")
        try:
            autoscaling_client.register_scalable_target(
                ServiceNamespace="sagemaker",
                ResourceId=resource_id,
                ScalableDimension="sagemaker:variant:DesiredInstanceCount",
                MinCapacity=1,
                MaxCapacity=3,
            )
            print("✅ Scalable target registered")
        except ClientError as reg_error:
            if "already exists" in str(
                reg_error
            ).lower() or "ValidationException" in str(reg_error):
                print("✅ Scalable target already exists")
            else:
                raise

        # Wait for target to be visible
        print("⏱️ Waiting for scalable target to be fully available...")
        max_retries = 10
        retry_count = 0
        target_found = False

        while retry_count < max_retries and not target_found:
            try:
                response = autoscaling_client.describe_scalable_targets(
                    ServiceNamespace="sagemaker", ResourceIds=[resource_id]
                )

                if response["ScalableTargets"]:
                    target = response["ScalableTargets"][0]
                    target_found = True
                    print("✅ Scalable target is now visible:")
                    print(f"   Resource ID: {target['ResourceId']}")
                    print(
                        f"   Min/Max Capacity: {target['MinCapacity']}-{target['MaxCapacity']}"
                    )
                    break
                else:
                    retry_count += 1
                    print(
                        f"⏳ Attempt {retry_count}/{max_retries}: Waiting for target visibility..."
                    )
                    time.sleep(3)

            except ClientError as e:
                retry_count += 1
                print(
                    f"⏳ Attempt {retry_count}/{max_retries}: Error checking visibility: {e}"
                )
                time.sleep(3)

        if not target_found:
            print("❌ Scalable target not visible after maximum retries.")
            return False

        # Create scaling policy
        policy_name = f"{endpoint_name}-scaling-policy"
        print(f"📋 Creating scaling policy '{policy_name}'...")

        try:
            autoscaling_client.put_scaling_policy(
                PolicyName=policy_name,
                ServiceNamespace="sagemaker",
                ResourceId=resource_id,
                ScalableDimension="sagemaker:variant:DesiredInstanceCount",
                PolicyType="TargetTrackingScaling",
                TargetTrackingScalingPolicyConfiguration={
                    "TargetValue": 70.0,
                    "PredefinedMetricSpecification": {
                        "PredefinedMetricType": "SageMakerVariantInvocationsPerInstance"
                    },
                    "ScaleOutCooldown": 300,
                    "ScaleInCooldown": 300,
                },
            )
            print(f"✅ Scaling policy '{policy_name}' created successfully!")
            print("📈 Configuration: 1-3 instances, target: 70 invocations/instance")

            # Verify the policy was created
            policies_response = autoscaling_client.describe_scaling_policies(
                ServiceNamespace="sagemaker", ResourceId=resource_id
            )

            if policies_response["ScalingPolicies"]:
                print("✅ Scaling policy verification successful:")
                for policy in policies_response["ScalingPolicies"]:
                    print(f"   Policy: {policy['PolicyName']}")
                    print(f"   Type: {policy['PolicyType']}")

            return True

        except ClientError as policy_error:
            if "already exists" in str(policy_error).lower():
                print(f"✅ Scaling policy '{policy_name}' already exists")
                return True
            else:
                print(f"❌ Error creating scaling policy: {policy_error}")
                return False

    except Exception as e:
        print(f"❌ Autoscaling setup failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Friday AI Autoscaling Fix")
    print("=" * 50)
    success = fix_autoscaling()

    if success:
        print("\n🎉 Autoscaling setup completed successfully!")
        print("📊 Your endpoint can now scale from 1 to 3 instances based on load.")
    else:
        print("\n⚠️ Autoscaling setup failed. Check the messages above.")

    exit(0 if success else 1)
