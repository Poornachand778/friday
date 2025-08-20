#!/usr/bin/env python3
"""
Safe deletion script for Friday AI SageMaker endpoint
Removes endpoint, configuration, and model with proper cleanup
"""

import boto3
import time
import argparse
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()


class FridayEndpointCleaner:
    """Safely delete Friday AI endpoint and associated resources"""

    def __init__(self, region: str = "us-east-1"):
        self.region = region
        self.sagemaker_client = boto3.client("sagemaker", region_name=region)
        self.autoscaling_client = boto3.client(
            "application-autoscaling", region_name=region
        )

        print("🧹 Friday AI Endpoint Cleaner initialized")
        print(f"   Region: {region}")

    def delete_autoscaling_policy(self, endpoint_name: str):
        """Remove autoscaling policy and target"""
        resource_id = f"endpoint/{endpoint_name}/variant/friday-variant"

        try:
            # List and delete scaling policies
            policies = self.autoscaling_client.describe_scaling_policies(
                ServiceNamespace="sagemaker", ResourceId=resource_id
            )

            for policy in policies.get("ScalingPolicies", []):
                print(f"🔄 Removing scaling policy: {policy['PolicyName']}")
                self.autoscaling_client.delete_scaling_policy(
                    PolicyName=policy["PolicyName"],
                    ServiceNamespace="sagemaker",
                    ResourceId=resource_id,
                    ScalableDimension="sagemaker:variant:DesiredInstanceCount",
                )

            # Deregister scalable target
            try:
                self.autoscaling_client.deregister_scalable_target(
                    ServiceNamespace="sagemaker",
                    ResourceId=resource_id,
                    ScalableDimension="sagemaker:variant:DesiredInstanceCount",
                )
                print("✅ Autoscaling target deregistered")
            except ClientError as e:
                if "not found" not in str(e).lower():
                    print(f"⚠️ Autoscaling cleanup warning: {e}")

        except ClientError as e:
            if "not found" not in str(e).lower():
                print(f"⚠️ Autoscaling policy cleanup warning: {e}")

    def delete_endpoint(self, endpoint_name: str, wait: bool = True):
        """Delete SageMaker endpoint"""
        try:
            # Check if endpoint exists
            self.sagemaker_client.describe_endpoint(EndpointName=endpoint_name)

            print(f"🔄 Deleting endpoint: {endpoint_name}")

            # Remove autoscaling first
            self.delete_autoscaling_policy(endpoint_name)

            # Delete endpoint
            self.sagemaker_client.delete_endpoint(EndpointName=endpoint_name)

            if wait:
                print("⏱️ Waiting for endpoint deletion...")
                self._wait_for_endpoint_deletion(endpoint_name)
            else:
                print("⏭️ Endpoint deletion initiated (not waiting)")

            return True

        except ClientError as e:
            if e.response["Error"]["Code"] == "ValidationException":
                print(f"⚠️ Endpoint {endpoint_name} not found (already deleted)")
                return True
            else:
                print(f"❌ Failed to delete endpoint: {e}")
                return False

    def delete_endpoint_config(self, config_name: str):
        """Delete endpoint configuration"""
        try:
            # Check if config exists
            self.sagemaker_client.describe_endpoint_config(
                EndpointConfigName=config_name
            )

            print(f"🔄 Deleting endpoint config: {config_name}")
            self.sagemaker_client.delete_endpoint_config(EndpointConfigName=config_name)
            print("✅ Endpoint config deleted")
            return True

        except ClientError as e:
            if e.response["Error"]["Code"] == "ValidationException":
                print(f"⚠️ Endpoint config {config_name} not found (already deleted)")
                return True
            else:
                print(f"❌ Failed to delete endpoint config: {e}")
                return False

    def delete_model(self, model_name: str):
        """Delete SageMaker model"""
        try:
            # Check if model exists
            self.sagemaker_client.describe_model(ModelName=model_name)

            print(f"🔄 Deleting model: {model_name}")
            self.sagemaker_client.delete_model(ModelName=model_name)
            print("✅ Model deleted")
            return True

        except ClientError as e:
            if e.response["Error"]["Code"] == "ValidationException":
                print(f"⚠️ Model {model_name} not found (already deleted)")
                return True
            else:
                print(f"❌ Failed to delete model: {e}")
                return False

    def _wait_for_endpoint_deletion(self, endpoint_name: str, timeout: int = 600):
        """Wait for endpoint to be fully deleted"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = self.sagemaker_client.describe_endpoint(
                    EndpointName=endpoint_name
                )
                status = response["EndpointStatus"]

                if status == "Deleting":
                    print(f"   Status: {status}... (waiting)")
                    time.sleep(15)
                elif status == "Failed":
                    print(f"⚠️ Endpoint deletion failed with status: {status}")
                    break
                else:
                    print(f"   Status: {status}...")
                    time.sleep(10)

            except ClientError as e:
                if e.response["Error"]["Code"] == "ValidationException":
                    print("✅ Endpoint fully deleted")
                    return
                else:
                    print(f"❌ Error checking endpoint status: {e}")
                    break

        print("⏰ Timeout waiting for endpoint deletion")

    def cleanup_friday_resources(
        self, endpoint_name: str = "friday-rt", wait: bool = True, force: bool = False
    ) -> bool:
        """Complete cleanup of Friday AI resources"""
        print("🎭 Friday AI Resource Cleanup")
        print("=" * 50)

        if not force:
            confirm = input(
                f"⚠️ Delete endpoint '{endpoint_name}' and all associated resources? (y/N): "
            )
            if confirm.lower() != "y":
                print("❌ Cleanup cancelled")
                return False

        success = True

        # Derive resource names
        config_name = f"{endpoint_name}-config"
        model_name = f"{endpoint_name}-model"

        try:
            # 1. Delete endpoint (includes autoscaling cleanup)
            print("\n🎯 Step 1: Deleting endpoint...")
            if not self.delete_endpoint(endpoint_name, wait=wait):
                success = False

            # 2. Delete endpoint configuration
            print("\n🎯 Step 2: Deleting endpoint configuration...")
            if not self.delete_endpoint_config(config_name):
                success = False

            # 3. Delete model
            print("\n🎯 Step 3: Deleting model...")
            if not self.delete_model(model_name):
                success = False

            # Summary
            print("\n📋 Cleanup Summary")
            print("=" * 50)

            if success:
                print("✅ All Friday AI resources deleted successfully")
                print("💰 Billing for endpoint instances has stopped")
                print("📦 Model artifacts remain in S3 (not deleted)")
                print("🔐 Secrets Manager entries remain (not deleted)")
            else:
                print("⚠️ Some resources may not have been deleted")
                print("🔍 Check AWS Console for any remaining resources")

            return success

        except KeyboardInterrupt:
            print("\n⚠️ Cleanup interrupted")
            print("🔍 Some resources may still exist - check AWS Console")
            return False
        except Exception as e:
            print(f"❌ Cleanup failed: {e}")
            return False

    def list_friday_resources(self, endpoint_name: str = "friday-rt"):
        """List all Friday AI related resources"""
        print("🔍 Friday AI Resource Inventory")
        print("=" * 50)

        # Derive resource names
        config_name = f"{endpoint_name}-config"
        model_name = f"{endpoint_name}-model"

        resources_found = []

        # Check endpoint
        try:
            response = self.sagemaker_client.describe_endpoint(
                EndpointName=endpoint_name
            )
            status = response["EndpointStatus"]
            instance_type = response["ProductionVariants"][0]["CurrentInstanceCount"]
            resources_found.append(f"📍 Endpoint: {endpoint_name} ({status})")

            if status == "InService":
                resources_found.append(
                    f"   💰 Cost: ~$0.75/hour × {instance_type} instances"
                )

        except ClientError:
            print(f"⚪ Endpoint: {endpoint_name} (not found)")

        # Check endpoint config
        try:
            self.sagemaker_client.describe_endpoint_config(
                EndpointConfigName=config_name
            )
            resources_found.append(f"⚙️ Config: {config_name} (exists)")
        except ClientError:
            print(f"⚪ Config: {config_name} (not found)")

        # Check model
        try:
            self.sagemaker_client.describe_model(ModelName=model_name)
            resources_found.append(f"🤖 Model: {model_name} (exists)")
        except ClientError:
            print(f"⚪ Model: {model_name} (not found)")

        # Show found resources
        if resources_found:
            print("📋 Active Resources:")
            for resource in resources_found:
                print(f"   {resource}")
            print("\n💡 Use --delete to remove all resources")
        else:
            print("✅ No Friday AI resources found")


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(description="Manage Friday AI endpoint resources")
    parser.add_argument("--endpoint", default="friday-rt", help="Endpoint name")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--delete", action="store_true", help="Delete all resources")
    parser.add_argument(
        "--no-wait", action="store_true", help="Don't wait for deletion"
    )
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompt")
    parser.add_argument("--list", action="store_true", help="List existing resources")

    args = parser.parse_args()

    cleaner = FridayEndpointCleaner(region=args.region)

    if args.list:
        cleaner.list_friday_resources(args.endpoint)
    elif args.delete:
        success = cleaner.cleanup_friday_resources(
            endpoint_name=args.endpoint, wait=not args.no_wait, force=args.force
        )
        exit(0 if success else 1)
    else:
        # Default: list resources
        cleaner.list_friday_resources(args.endpoint)
        print("\n💡 Use --delete to remove resources")


if __name__ == "__main__":
    main()
