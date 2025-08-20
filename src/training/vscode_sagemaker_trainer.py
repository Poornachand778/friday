#!/usr/bin/env python3
"""
VS Code SageMaker Training Script
Run SageMaker training from VS Code
"""

import os
import boto3
from pathlib import Path
from sagemaker import get_execution_role, Session
from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput
from datetime import datetime

# Load environment variables
from dotenv import load_dotenv

load_dotenv()


class VSCodeSageMakerTrainer:
    def __init__(self):
        """Initialize VS Code SageMaker training environment"""
        # Use the region from AWS CLI config
        self.region = os.getenv("AWS_DEFAULT_REGION", "us-east-2")

        # Set the region explicitly
        boto_session = boto3.Session(region_name=self.region)
        self.session = Session(boto_session=boto_session)

        # Use SageMaker's default bucket instead of creating our own
        self.s3_bucket = self.session.default_bucket()
        self.s3_prefix = os.getenv("S3_PREFIX", "friday-finetuning")

        # Try to get SageMaker role from environment or use default
        try:
            self.role = get_execution_role()
        except Exception:
            self.role = os.getenv("SAGEMAKER_ROLE")
            if not self.role:
                print("❌ No SageMaker role found. Please set SAGEMAKER_ROLE in .env")
                raise ValueError("SageMaker role required")

        print("🔧 Initialized SageMaker session")
        print(f"   Region: {self.region}")
        print(f"   Role: {self.role}")
        print(f"   S3 Bucket: {self.s3_bucket}")

    def upload_data(self):
        """Upload training data to S3"""
        print("\n📤 Uploading training data to S3...")

        data_dir = Path("data/instructions")
        files_to_upload = [
            ("iteration1_train.labeled.jsonl", "train.jsonl"),
            ("iteration1_valid.jsonl", "valid.jsonl"),
        ]

        s3_inputs = {}

        for local_file, s3_file in files_to_upload:
            local_path = data_dir / local_file
            if local_path.exists():
                # Upload to S3
                s3_uri = self.session.upload_data(
                    path=str(local_path),
                    bucket=self.s3_bucket,
                    key_prefix=f"{self.s3_prefix}/data",
                )
                s3_inputs[s3_file] = s3_uri
                print(f"   ✅ {local_file} → {s3_uri}")
            else:
                print(f"   ❌ File not found: {local_path}")

        return s3_inputs

    def create_training_job(
        self,
        s3_inputs,
        model_name=None,
        epochs=3,
        batch_size=4,
        learning_rate=2e-4,
        instance_type=None,
        max_length=None,
    ):
        """Create and start SageMaker training job"""
        print("\n🚀 Creating SageMaker training job...")

        # Use model from environment if not provided
        if model_name is None:
            model_name = os.getenv("MODEL_NAME", "meta-llama/Llama-2-7b-chat-hf")

        # Use instance type from parameter or environment
        if instance_type is None:
            instance_type = os.getenv("INSTANCE_TYPE", "ml.g5.2xlarge")

        print(f"   Using model: {model_name}")
        print(f"   Using instance: {instance_type}")

        # Training script
        script_dir = Path("scripts/train")
        script_dir.mkdir(exist_ok=True)

        # Build hyperparameters
        hyperparameters = {
            "model-name": model_name,
            "epochs": epochs,
            "batch-size": batch_size,
            "learning-rate": learning_rate,
        }

        # Add max_length if specified (for memory control)
        if max_length is not None:
            hyperparameters["max-length"] = max_length

        # Use PyTorch estimator (simpler for Transformers with custom versions)
        estimator = PyTorch(
            entry_point="sagemaker_train.py",
            source_dir=str(script_dir),
            instance_type=instance_type,
            instance_count=1,
            role=self.role,
            framework_version="2.0.0",
            py_version="py310",
            base_job_name="friday-finetune",
            hyperparameters=hyperparameters,
            environment={
                "TOKENIZERS_PARALLELISM": "false",
                "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
                "HUGGINGFACE_TOKEN": os.getenv("HUGGINGFACE_TOKEN", ""),
            },
        )

        # Training inputs
        training_inputs = {
            "training": TrainingInput(
                s3_data=f"s3://{self.s3_bucket}/{self.s3_prefix}/data",
                content_type="application/json",
            )
        }

        # Start training
        job_name = f"friday-lora-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        print(f"   Job name: {job_name}")
        print(f"   Instance: {os.getenv('INSTANCE_TYPE', 'ml.g5.2xlarge')}")
        print("   Estimated cost: $3-5")

        estimator.fit(
            inputs=training_inputs,
            job_name=job_name,
            wait=False,  # Don't block VS Code
        )

        return estimator, job_name

    def monitor_training(self, estimator, job_name):
        """Monitor training job from VS Code"""
        print(f"\n👀 Monitoring training job: {job_name}")
        print("   You can:")
        print("   1. Check status in VS Code terminal")
        print("   2. View logs in SageMaker console")
        print("   3. Use AWS Toolkit extension")

        # Print monitoring commands
        print("\n📊 Monitoring commands:")
        print("   # Check status:")
        print(f"   aws sagemaker describe-training-job --training-job-name {job_name}")
        print("   ")
        print("   # View logs:")
        print(
            "   aws logs describe-log-streams --log-group-name /aws/sagemaker/TrainingJobs"
        )

        # Wait for completion (optional)
        user_input = input("\n❓ Wait for completion in VS Code? (y/n): ")
        if user_input.lower() == "y":
            print("   ⏳ Training in progress... (this may take 1-2 hours)")
            try:
                # Actually wait for the training job to complete
                # Use the SageMaker client to wait for job completion
                sagemaker_client = self.session.boto_session.client("sagemaker")
                waiter = sagemaker_client.get_waiter(
                    "training_job_completed_or_stopped"
                )
                waiter.wait(TrainingJobName=job_name)

                # Check final status
                response = sagemaker_client.describe_training_job(
                    TrainingJobName=job_name
                )
                status = response["TrainingJobStatus"]

                if status == "Completed":
                    print("   ✅ Training completed!")
                    # Get model location from the response
                    model_location = response.get("ModelArtifacts", {}).get(
                        "S3ModelArtifacts"
                    )
                    if model_location:
                        print(f"   📦 Model saved to: {model_location}")
                        return model_location
                    else:
                        print("   ⚠️ Training completed but model data not available")
                        return None
                else:
                    failure_reason = response.get("FailureReason", "Unknown error")
                    print(f"   ❌ Training failed: {failure_reason}")
                    return None

            except Exception as e:
                print(f"   ❌ Error monitoring training: {str(e)}")
                print("   💡 Check CloudWatch logs for details")
                return None
        else:
            print("   🏃 Training continues in background")
            print(
                f"   📍 Track at: https://console.aws.amazon.com/sagemaker/home?region={self.region}#/jobs/{job_name}"
            )
            return None


def main():
    """Main training function"""
    print("🤖 Friday AI - VS Code SageMaker Training")
    print("=" * 50)

    try:
        # Initialize trainer
        trainer = VSCodeSageMakerTrainer()

        # Upload data
        s3_inputs = trainer.upload_data()
        if not s3_inputs:
            print("❌ No data uploaded. Exiting.")
            return

        # Start training
        estimator, job_name = trainer.create_training_job(s3_inputs)

        # Monitor training
        model_location = trainer.monitor_training(estimator, job_name)

        if model_location:
            print("\n🎉 Training completed successfully!")
            print(f"📦 Model: {model_location}")
            print("💡 Next: Deploy model or download for local testing")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Check AWS credentials and permissions")


if __name__ == "__main__":
    main()
