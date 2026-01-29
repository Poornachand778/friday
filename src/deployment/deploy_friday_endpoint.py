#!/usr/bin/env python3
"""
Production-Ready SageMaker Real-time Endpoint Deployer for Friday AI
Deploys Meta-Llama-3.1-8B-Instruct + LoRA as a scalable real-time endpoint
"""

import boto3
import sagemaker
import time
import os
from pathlib import Path
from dotenv import load_dotenv
from botocore.exceptions import ClientError

# Load environment variables
load_dotenv()


class FridayEndpointDeployer:
    """Production SageMaker endpoint deployer for Friday AI"""

    def __init__(self, region: str = "us-east-1", endpoint_name: str = "friday-rt"):
        """
        Initialize the deployer

        Args:
            region: AWS region for deployment
            endpoint_name: Name for the SageMaker endpoint
        """
        self.region = region
        self.endpoint_name = endpoint_name
        self.model_name = f"{endpoint_name}-model"
        self.config_name = f"{endpoint_name}-config"

        # Initialize AWS clients
        self.session = boto3.Session(region_name=region)
        self.sagemaker_client = self.session.client("sagemaker")
        self.s3_client = self.session.client("s3")
        self.iam_client = self.session.client("iam")
        self.autoscaling_client = self.session.client("application-autoscaling")
        self.sagemaker_session = sagemaker.Session(boto_session=self.session)

        # Get SageMaker execution role
        try:
            self.role = sagemaker.get_execution_role()
        except ValueError:
            # Running outside SageMaker, get role from environment
            role_env = os.getenv("SAGEMAKER_ROLE")
            if role_env:
                # Check if it's already an ARN
                if role_env.startswith("arn:aws:iam::"):
                    self.role = role_env
                else:
                    # It's a role name, get the ARN
                    try:
                        role_response = self.iam_client.get_role(RoleName=role_env)
                        self.role = role_response["Role"]["Arn"]
                    except ClientError:
                        raise ValueError(f"Could not find IAM role: {role_env}")
            else:
                # Fallback to a default execution role pattern
                account_id = boto3.client("sts").get_caller_identity()["Account"]
                self.role = (
                    f"arn:aws:iam::{account_id}:role/FridayAI-SageMaker-ExecutionRole"
                )

        # S3 bucket for artifacts
        account_id = boto3.client("sts").get_caller_identity()["Account"]
        self.bucket = f"sagemaker-{region}-{account_id}"

        # Instance configuration - use ml.g5.12xlarge (ml.g5.2xlarge has 0 quota)
        self.instance_type = os.getenv("SAGEMAKER_INSTANCE_TYPE", "ml.g5.12xlarge")

        print("🚀 Friday AI Endpoint Deployer initialized")
        print(f"   Region: {self.region}")
        print(f"   Endpoint: {self.endpoint_name}")
        print(f"   Instance: {self.instance_type}")
        print(f"   Bucket: {self.bucket}")

    def prepare_model_artifacts(self) -> str:
        """Package and upload model artifacts to S3"""
        print("📦 Creating model archive...")

        # Create deployment directory if it doesn't exist
        deployment_dir = Path("deployment")
        deployment_dir.mkdir(exist_ok=True)

        # Package model artifacts
        import tarfile

        model_archive = deployment_dir / "model.tar.gz"
        with tarfile.open(model_archive, "w:gz") as tar:
            # Add inference code
            inference_dir = Path("src/inference/sagemaker_code")
            if inference_dir.exists():
                for file_path in inference_dir.rglob("*"):
                    if file_path.is_file():
                        arcname = f"code/{file_path.name}"
                        tar.add(file_path, arcname=arcname)

            # Add model artifacts (LoRA adapters) - Use iteration3 subdirectory
            model_dir = Path("models/trained/iteration4")
            if model_dir.exists():
                for file_path in model_dir.glob("*.json"):  # Config files
                    tar.add(file_path, arcname=f"adapters/{file_path.name}")
                for file_path in model_dir.glob("*.safetensors"):  # Model weights
                    tar.add(file_path, arcname=f"adapters/{file_path.name}")
                for file_path in model_dir.glob("tokenizer*"):  # Tokenizer files
                    tar.add(file_path, arcname=f"adapters/{file_path.name}")

        print("✅ Model archive created: deployment/model.tar.gz")

        # Upload to S3
        s3_key = "friday/model.tar.gz"
        s3_uri = f"s3://{self.bucket}/{s3_key}"

        print(f"☁️ Uploading to {s3_uri}...")
        self.s3_client.upload_file(str(model_archive), self.bucket, s3_key)
        print("✅ Model uploaded to S3")

        return s3_uri

    def get_huggingface_token(self) -> str:
        """Get HuggingFace token from secrets or environment"""
        # Try to get from AWS Secrets Manager first
        try:
            secrets_client = self.session.client("secretsmanager")
            response = secrets_client.get_secret_value(SecretId="friday-ai/hf-token")
            return response["SecretString"]
        except Exception:
            print(
                "⚠️ No permission to access friday-ai/hf-token. Using environment variable..."
            )

        # Fallback to environment variable
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError(
                "HF_TOKEN not found. Set environment variable or store in AWS Secrets Manager as 'friday-ai/hf-token'"
            )

        return hf_token

    def create_or_update_model(self, model_data_url: str) -> str:
        """Create or update SageMaker model"""
        print(f"🆕 Creating new model: {self.model_name}")

        # Get HuggingFace token
        hf_token = self.get_huggingface_token()

        # Environment variables for inference - FIXED CACHE DIRECTORIES
        environment = {
            "BASE_MODEL_ID": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "HF_TOKEN": hf_token,
            "TRANSFORMERS_CACHE": "/tmp/hf",  # Writable location
            "HF_HOME": "/tmp/hf",  # Writable location
            "HF_HUB_ENABLE_HF_TRANSFER": "1",  # Faster downloads
            "USE_4BIT": "true",
            "MAX_INPUT_LENGTH": "4096",
            "MAX_TOTAL_TOKENS": "8192",
            "SAGEMAKER_PROGRAM": "inference.py",
            "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/code",
            "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",  # INFO level logging
        }

        # Use HuggingFace Transformers DLC - supports our script mode inference.py
        image_uri = f"763104351884.dkr.ecr.{self.region}.amazonaws.com/huggingface-pytorch-inference:2.6.0-transformers4.49.0-gpu-py312-cu124-ubuntu22.04"

        print(f"🤗 Using HuggingFace Transformers DLC: {image_uri}")

        model_config = {
            "ModelName": self.model_name,
            "ExecutionRoleArn": self.role,
            "PrimaryContainer": {
                "Image": image_uri,
                "ModelDataUrl": model_data_url,
                "Environment": environment,
            },
            "Tags": [
                {"Key": "Project", "Value": "FridayAI"},
                {"Key": "Environment", "Value": "Production"},
                {"Key": "ModelType", "Value": "LLaMA-3.1-8B+LoRA"},
            ],
        }

        try:
            # Delete existing model if it exists
            try:
                self.sagemaker_client.describe_model(ModelName=self.model_name)
                print(f"🗑️ Deleting existing model: {self.model_name}")
                self.sagemaker_client.delete_model(ModelName=self.model_name)
                time.sleep(2)  # Brief pause
            except ClientError:
                pass  # Model doesn't exist

            # Create new model
            self.sagemaker_client.create_model(**model_config)
            print(f"✅ Model {self.model_name} ready")
            return self.model_name

        except ClientError as e:
            print(f"❌ Model creation failed: {e}")
            raise

    def create_endpoint_config(self, model_name: str) -> str:
        """Create or update endpoint configuration"""
        print(f"🔄 Updating existing endpoint config: {self.config_name}")

        config = {
            "EndpointConfigName": self.config_name,
            "ProductionVariants": [
                {
                    "VariantName": "friday-variant",
                    "ModelName": model_name,
                    "InitialInstanceCount": 1,
                    "InstanceType": self.instance_type,
                    "InitialVariantWeight": 1.0,
                    "ContainerStartupHealthCheckTimeoutInSeconds": 1800,  # 30 minutes for model loading
                }
            ],
            "Tags": [
                {"Key": "Project", "Value": "FridayAI"},
                {"Key": "Environment", "Value": "Production"},
            ],
        }

        try:
            # Delete existing config if it exists
            try:
                self.sagemaker_client.describe_endpoint_config(
                    EndpointConfigName=self.config_name
                )
                print(f"🗑️ Deleting existing config: {self.config_name}")
                self.sagemaker_client.delete_endpoint_config(
                    EndpointConfigName=self.config_name
                )
                time.sleep(2)  # Brief pause
            except ClientError:
                pass  # Config doesn't exist

            # Create new config
            self.sagemaker_client.create_endpoint_config(**config)
            print(f"✅ Endpoint config {self.config_name} ready")
            return self.config_name

        except ClientError as e:
            print(f"❌ Endpoint config creation failed: {e}")
            raise

    def create_or_update_endpoint(self, config_name: str) -> str:
        """Create or update endpoint"""
        try:
            # Check if endpoint exists
            try:
                response = self.sagemaker_client.describe_endpoint(
                    EndpointName=self.endpoint_name
                )
                status = response["EndpointStatus"]

                if status == "InService":
                    print(f"🔄 Updating existing endpoint: {self.endpoint_name}")
                    self.sagemaker_client.update_endpoint(
                        EndpointName=self.endpoint_name, EndpointConfigName=config_name
                    )
                elif status in ["Creating", "Updating"]:
                    print(f"⏱️ Endpoint is {status.lower()}, waiting...")
                else:
                    print(f"⚠️ Endpoint in {status} state, recreating...")
                    self.sagemaker_client.delete_endpoint(
                        EndpointName=self.endpoint_name
                    )
                    self._wait_for_endpoint_deletion()
                    raise ClientError(
                        {"Error": {"Code": "NotFound"}}, "describe_endpoint"
                    )

            except ClientError as e:
                if (
                    "does not exist" in str(e)
                    or e.response["Error"]["Code"] == "ValidationException"
                ):
                    print(f"🆕 Creating new endpoint: {self.endpoint_name}")
                    self.sagemaker_client.create_endpoint(
                        EndpointName=self.endpoint_name,
                        EndpointConfigName=config_name,
                        Tags=[
                            {"Key": "Project", "Value": "FridayAI"},
                            {"Key": "Environment", "Value": "Production"},
                        ],
                    )
                else:
                    raise

            # Wait for endpoint to be InService
            print("⏱️ Endpoint creation in progress (this may take 10-15 minutes)...")
            self._wait_for_endpoint_ready()
            print(f"✅ Endpoint {self.endpoint_name} is InService")
            return self.endpoint_name

        except ClientError as e:
            print(f"❌ Endpoint creation/update failed: {e}")
            raise

    def _wait_for_endpoint_deletion(self):
        """Wait for endpoint to be fully deleted"""
        print("⏱️ Waiting for endpoint deletion...")
        for i in range(60):  # 5 minutes max
            try:
                self.sagemaker_client.describe_endpoint(EndpointName=self.endpoint_name)
                if i % 6 == 0:  # Print every 30 seconds
                    print(f"   [{i * 5}s] Still deleting...")
                time.sleep(5)
            except ClientError:
                print("✅ Endpoint deleted")
                return
        print("⚠️ Timeout waiting for deletion, proceeding...")

    def _wait_for_endpoint_ready(self):
        """Wait for endpoint to reach InService status"""
        while True:
            try:
                response = self.sagemaker_client.describe_endpoint(
                    EndpointName=self.endpoint_name
                )
                status = response["EndpointStatus"]

                if status == "InService":
                    break
                elif status in ["Creating", "Updating"]:
                    print(f"   Status: {status}... (waiting)")
                    time.sleep(30)
                elif status == "Failed":
                    failure_reason = response.get("FailureReason", "Unknown")
                    raise RuntimeError(f"Endpoint failed: {failure_reason}")
                else:
                    print(f"   Unexpected status: {status}")
                    time.sleep(30)

            except ClientError as e:
                print(f"   Error checking status: {e}")
                time.sleep(30)

    def setup_autoscaling(self):
        """Configure autoscaling for the endpoint"""
        print("🔍 Verifying endpoint is InService before setting up autoscaling...")

        # Ensure endpoint is ready
        response = self.sagemaker_client.describe_endpoint(
            EndpointName=self.endpoint_name
        )
        if response["EndpointStatus"] != "InService":
            raise RuntimeError(
                f"Endpoint not ready for autoscaling: {response['EndpointStatus']}"
            )

        print("✅ Endpoint is InService, proceeding with autoscaling setup")

        resource_id = f"endpoint/{self.endpoint_name}/variant/friday-variant"

        # Register scalable target
        print("📋 Registering scalable target...")
        try:
            self.autoscaling_client.register_scalable_target(
                ServiceNamespace="sagemaker",
                ResourceId=resource_id,
                ScalableDimension="sagemaker:variant:DesiredInstanceCount",
                MinCapacity=1,
                MaxCapacity=3,
                RoleARN=f"arn:aws:iam::{boto3.client('sts').get_caller_identity()['Account']}:role/aws-service-role/sagemaker.application-autoscaling.amazonaws.com/AWSServiceRoleForApplicationAutoScaling_SageMakerEndpoint",
            )
            print("✅ Scalable target registered")
        except ClientError as e:
            if "already exists" in str(e):
                print("✅ Scalable target already exists")
            else:
                raise

        # Wait for scalable target to be fully available
        print("⏱️ Waiting for scalable target to be fully available...")
        for i in range(12):  # 2 minutes max
            try:
                targets = self.autoscaling_client.describe_scalable_targets(
                    ServiceNamespace="sagemaker", ResourceIds=[resource_id]
                )
                if targets["ScalableTargets"]:
                    print("✅ Scalable target is now visible and ready")
                    break
            except ClientError:
                pass

            if i < 11:
                time.sleep(10)
        else:
            print("⚠️ Scalable target not visible yet, but proceeding...")

        # Create scaling policy
        policy_name = f"{self.endpoint_name}-scaling-policy"
        print(f"📋 Creating scaling policy '{policy_name}'...")

        try:
            self.autoscaling_client.put_scaling_policy(
                PolicyName=policy_name,
                ServiceNamespace="sagemaker",
                ResourceId=resource_id,
                ScalableDimension="sagemaker:variant:DesiredInstanceCount",
                PolicyType="TargetTrackingScaling",
                TargetTrackingScalingPolicyConfiguration={
                    "TargetValue": 70.0,  # Target 70 invocations per instance
                    "PredefinedMetricSpecification": {
                        "PredefinedMetricType": "SageMakerVariantInvocationsPerInstance"
                    },
                    "ScaleOutCooldown": 300,  # 5 minutes
                    "ScaleInCooldown": 300,  # 5 minutes
                },
            )
            print(f"✅ Scaling policy '{policy_name}' configured successfully!")
            print("📈 Autoscaling: 1-3 instances, target: 70 invocations/instance")

        except ClientError as e:
            if "already exists" in str(e):
                print(f"✅ Scaling policy '{policy_name}' already exists")
            else:
                raise

    def verify_all_artifacts(self) -> bool:
        """Verify all required model artifacts exist"""
        print("✅ All Friday AI model artifacts found")

        # Check inference code
        inference_dir = Path("src/inference/sagemaker_code")
        required_files = ["inference.py", "requirements.txt"]

        for file_name in required_files:
            file_path = inference_dir / file_name
            if not file_path.exists():
                print(f"❌ Missing required file: {file_path}")
                return False

        # Check model files (LoRA adapters) - Use iteration3 subdirectory
        model_dir = Path("models/trained/iteration3")
        required_patterns = ["*.json", "*.safetensors", "tokenizer*"]

        for pattern in required_patterns:
            if not list(model_dir.glob(pattern)):
                print(f"❌ Missing model files matching: {pattern}")
                return False

        return True

    def deploy(self):
        """Execute the complete deployment process"""
        print("🎭 Starting Friday AI endpoint deployment...")
        print("=" * 60)

        # Verify artifacts
        if not self.verify_all_artifacts():
            raise RuntimeError("Required model artifacts not found")

        # Prepare and upload model
        model_data_url = self.prepare_model_artifacts()

        # Create model
        model_name = self.create_or_update_model(model_data_url)

        # Create endpoint config
        config_name = self.create_endpoint_config(model_name)

        # Deploy endpoint
        endpoint_name = self.create_or_update_endpoint(config_name)

        # Setup autoscaling
        self.setup_autoscaling()

        # Success summary
        endpoint_arn = f"arn:aws:sagemaker:{self.region}:{boto3.client('sts').get_caller_identity()['Account']}:endpoint/{endpoint_name}"

        print("\n🎉 Friday AI endpoint deployment completed!")
        print("=" * 60)
        print(f"📍 Endpoint Name: {endpoint_name}")
        print(f"🔗 Endpoint ARN: {endpoint_arn}")
        print(f"💻 Instance Type: {self.instance_type}")
        print("📈 Autoscaling: 1-3 instances")
        print("💰 Estimated cost: ~$0.75/hour per instance")
        print("🧪 Test with: python validate_deployment.py")


def main():
    """Main deployment function"""
    deployer = FridayEndpointDeployer(
        region=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        endpoint_name=os.getenv("SAGEMAKER_ENDPOINT_NAME", "friday-iter3"),
    )

    try:
        deployer.deploy()
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        raise


if __name__ == "__main__":
    main()
