#!/usr/bin/env python3
"""
Friday AI Deployment Validation Script
Comprehensive validation of SageMaker endpoint deployment
"""

import boto3
import json
import time
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from botocore.exceptions import ClientError

load_dotenv()


class DeploymentValidator:
    """Validate Friday AI SageMaker endpoint deployment"""

    def __init__(self):
        self.region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        self.endpoint_name = "friday-rt"
        self.model_name = "friday-rt-model"
        self.config_name = "friday-rt-config"

        # Initialize AWS clients
        self.session = boto3.Session(region_name=self.region)
        self.sagemaker = self.session.client("sagemaker")
        self.autoscaling = self.session.client("application-autoscaling")
        self.cloudwatch = self.session.client("cloudwatch")
        self.logs = self.session.client("logs")

        print(f"🔍 Validating Friday AI deployment in {self.region}")
        print("=" * 60)

    def check_model_status(self):
        """Check SageMaker model status"""
        print("📦 Checking Model Status...")
        try:
            response = self.sagemaker.describe_model(ModelName=self.model_name)
            print(f"✅ Model: {self.model_name}")
            print(f"   ARN: {response['ModelArn']}")
            print(f"   Creation Time: {response['CreationTime']}")
            print(f"   Execution Role: {response['ExecutionRoleArn']}")
            return True
        except ClientError as e:
            print(f"❌ Model validation failed: {e}")
            return False

    def check_endpoint_config_status(self):
        """Check endpoint configuration status"""
        print("\n⚙️ Checking Endpoint Config Status...")
        try:
            response = self.sagemaker.describe_endpoint_config(
                EndpointConfigName=self.config_name
            )
            print(f"✅ Endpoint Config: {self.config_name}")
            print(f"   ARN: {response['EndpointConfigArn']}")
            print(f"   Creation Time: {response['CreationTime']}")

            # Check production variants
            for variant in response["ProductionVariants"]:
                print(f"   Variant: {variant['VariantName']}")
                print(f"   Instance Type: {variant['InstanceType']}")
                print(f"   Initial Instance Count: {variant['InitialInstanceCount']}")
                print(f"   Model: {variant['ModelName']}")
            return True
        except ClientError as e:
            print(f"❌ Endpoint config validation failed: {e}")
            return False

    def check_endpoint_status(self):
        """Check endpoint status and health"""
        print("\n🚀 Checking Endpoint Status...")
        try:
            response = self.sagemaker.describe_endpoint(EndpointName=self.endpoint_name)
            status = response["EndpointStatus"]

            print(f"✅ Endpoint: {self.endpoint_name}")
            print(f"   ARN: {response['EndpointArn']}")
            print(f"   Status: {status}")
            print(f"   Creation Time: {response['CreationTime']}")
            print(f"   Last Modified: {response['LastModifiedTime']}")

            if status == "InService":
                print("🟢 Endpoint is healthy and ready for inference!")

                # Check production variants
                for variant in response["ProductionVariants"]:
                    print(f"   Variant: {variant['VariantName']}")
                    print(f"   Current Weight: {variant['CurrentWeight']}")
                    print(
                        f"   Desired Instance Count: {variant['DesiredInstanceCount']}"
                    )
                    print(
                        f"   Current Instance Count: {variant['CurrentInstanceCount']}"
                    )
                return True
            elif status in ["Creating", "Updating"]:
                print(f"🟡 Endpoint is {status.lower()}...")
                return True
            else:
                print(f"🔴 Endpoint status: {status}")
                if "FailureReason" in response:
                    print(f"   Failure Reason: {response['FailureReason']}")
                return False

        except ClientError as e:
            print(f"❌ Endpoint validation failed: {e}")
            return False

    def check_autoscaling_status(self):
        """Check autoscaling configuration"""
        print("\n📈 Checking Autoscaling Configuration...")
        try:
            resource_id = f"endpoint/{self.endpoint_name}/variant/friday-variant"

            # Check scalable target
            try:
                response = self.autoscaling.describe_scalable_targets(
                    ServiceNamespace="sagemaker", ResourceIds=[resource_id]
                )

                if response["ScalableTargets"]:
                    target = response["ScalableTargets"][0]
                    print("✅ Scalable Target Found:")
                    print(f"   Resource ID: {target['ResourceId']}")
                    print(f"   Min Capacity: {target['MinCapacity']}")
                    print(f"   Max Capacity: {target['MaxCapacity']}")
                    print(f"   Creation Time: {target['CreationTime']}")

                    # Check scaling policies
                    policies_response = self.autoscaling.describe_scaling_policies(
                        ServiceNamespace="sagemaker", ResourceId=resource_id
                    )

                    if policies_response["ScalingPolicies"]:
                        for policy in policies_response["ScalingPolicies"]:
                            print(f"✅ Scaling Policy: {policy['PolicyName']}")
                            print(f"   Policy Type: {policy['PolicyType']}")
                            if "TargetTrackingScalingPolicyConfiguration" in policy:
                                config = policy[
                                    "TargetTrackingScalingPolicyConfiguration"
                                ]
                                print(f"   Target Value: {config['TargetValue']}")
                                print(
                                    f"   Metric: {config['PredefinedMetricSpecification']['PredefinedMetricType']}"
                                )
                        return True
                    else:
                        print("⚠️ No scaling policies found")
                        return False
                else:
                    print("⚠️ No scalable targets found")
                    return False

            except ClientError as e:
                print(f"⚠️ Autoscaling check failed: {e}")
                return False

        except Exception as e:
            print(f"❌ Autoscaling validation error: {e}")
            return False

    def check_cloudwatch_logs(self):
        """Check CloudWatch logs for any issues"""
        print("\n📊 Checking CloudWatch Logs...")
        try:
            log_group = f"/aws/sagemaker/Endpoints/{self.endpoint_name}"

            # Check if log group exists
            try:
                self.logs.describe_log_groups(logGroupNamePrefix=log_group)
                print(f"✅ Log Group: {log_group}")

                # Get recent log streams
                streams_response = self.logs.describe_log_streams(
                    logGroupName=log_group,
                    orderBy="LastEventTime",
                    descending=True,
                    limit=3,
                )

                if streams_response["logStreams"]:
                    print(
                        f"📝 Found {len(streams_response['logStreams'])} recent log streams"
                    )

                    # Check for errors in recent logs
                    recent_stream = streams_response["logStreams"][0]
                    events_response = self.logs.get_log_events(
                        logGroupName=log_group,
                        logStreamName=recent_stream["logStreamName"],
                        limit=50,
                        startFromHead=False,
                    )

                    error_count = 0
                    for event in events_response["events"]:
                        message = event["message"].lower()
                        if any(
                            keyword in message
                            for keyword in ["error", "exception", "failed", "critical"]
                        ):
                            error_count += 1

                    if error_count > 0:
                        print(f"⚠️ Found {error_count} error messages in recent logs")
                    else:
                        print("✅ No recent error messages found")

                    return True
                else:
                    print("📝 No log streams found yet")
                    return True

            except ClientError as e:
                if e.response["Error"]["Code"] == "ResourceNotFoundException":
                    print("📝 CloudWatch log group not created yet")
                    return True
                else:
                    raise

        except Exception as e:
            print(f"⚠️ CloudWatch logs check failed: {e}")
            return False

    def run_basic_inference_test(self):
        """Run a basic inference test"""
        print("\n🧪 Running Basic Inference Test...")
        try:
            # Configure runtime with longer timeout for first inference
            runtime = self.session.client(
                "sagemaker-runtime",
                region_name=self.region,
                config=boto3.session.Config(
                    read_timeout=300,  # 5 minutes for first inference
                    connect_timeout=30,
                    retries={"max_attempts": 3, "mode": "standard"},
                ),
            )

            # Simple test payload
            test_payload = {
                "inputs": "Hello, how are you today?",
                "parameters": {
                    "max_new_tokens": 50,
                    "temperature": 0.7,
                    "do_sample": True,
                },
            }

            print("📤 Sending test request...")
            response = runtime.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType="application/json",
                Body=json.dumps(test_payload),
            )

            result = json.loads(response["Body"].read().decode())
            print("📥 Response received!")
            print("✅ Inference test successful")
            print(f"   Response length: {len(str(result))} characters")

            return True

        except Exception as e:
            print(f"❌ Inference test failed: {e}")
            return False

    def generate_cost_estimate(self):
        """Generate cost estimates"""
        print("\n💰 Cost Estimates...")

        # Base costs (approximate)
        g5_2xlarge_hourly = 1.006  # USD per hour

        print("📊 Instance: ml.g5.2xlarge")
        print(f"   Hourly cost: ${g5_2xlarge_hourly:.3f}")
        print(f"   Daily cost (1 instance): ${g5_2xlarge_hourly * 24:.2f}")
        print(f"   Monthly cost (1 instance): ${g5_2xlarge_hourly * 24 * 30:.2f}")
        print(
            f"   Max monthly cost (3 instances): ${g5_2xlarge_hourly * 24 * 30 * 3:.2f}"
        )

    def run_validation(self):
        """Run complete validation suite"""
        print("🎯 Friday AI Deployment Validation")
        print(f"⏰ Started at: {datetime.now()}")
        print("=" * 60)

        results = {
            "model": self.check_model_status(),
            "endpoint_config": self.check_endpoint_config_status(),
            "endpoint": self.check_endpoint_status(),
            "autoscaling": self.check_autoscaling_status(),
            "logs": self.check_cloudwatch_logs(),
            "inference": self.run_basic_inference_test(),
        }

        print("\n" + "=" * 60)
        print("📋 VALIDATION SUMMARY")
        print("=" * 60)

        passed = sum(results.values())
        total = len(results)

        for component, status in results.items():
            icon = "✅" if status else "❌"
            print(
                f"{icon} {component.replace('_', ' ').title()}: {'PASS' if status else 'FAIL'}"
            )

        print(f"\n🎯 Overall Status: {passed}/{total} checks passed")

        if passed == total:
            print("🎉 All validations passed! Deployment is healthy.")
        else:
            print("⚠️ Some validations failed. Check the details above.")

        self.generate_cost_estimate()

        return passed == total


if __name__ == "__main__":
    validator = DeploymentValidator()
    success = validator.run_validation()
    exit(0 if success else 1)
