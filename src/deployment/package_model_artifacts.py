#!/usr/bin/env python3
"""
Package Friday AI model artifacts for SageMaker deployment
Creates model.tar.gz with inference code and LoRA adapters
"""

import tarfile
import os
import boto3
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class ModelPackager:
    """Packages Friday AI model artifacts for deployment"""

    def __init__(self, bucket_name: str = None, region: str = "us-east-1"):
        self.region = region
        self.session = boto3.Session(region_name=region)
        self.s3_client = self.session.client("s3")

        # Get bucket name from environment or SageMaker default
        if bucket_name:
            self.bucket = bucket_name
        else:
            # Try to get default SageMaker bucket
            try:
                import sagemaker

                sagemaker_session = sagemaker.Session(boto_session=self.session)
                self.bucket = sagemaker_session.default_bucket()
            except Exception:
                # Fallback to environment variable
                self.bucket = os.getenv("S3_BUCKET")
                if not self.bucket:
                    raise ValueError(
                        "S3 bucket must be specified or set S3_BUCKET environment variable"
                    )

        print("📦 Model Packager initialized")
        print(f"   Region: {self.region}")
        print(f"   Bucket: {self.bucket}")

    def validate_artifacts(self) -> bool:
        """Validate all required model artifacts exist"""
        print("🔍 Validating model artifacts...")

        # Check inference code under src/inference/sagemaker_code
        code_dir = Path("src/inference/sagemaker_code")
        code_files = [code_dir / "inference.py", code_dir / "requirements.txt"]

        for file_path in code_files:
            if not file_path.exists():
                print(f"❌ Missing code file: {file_path}")
                return False

        # Check LoRA adapters
        adapter_dir = Path("models/trained")
        required_adapters = [
            "adapter_config.json",
            "adapter_model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ]

        for adapter_file in required_adapters:
            adapter_path = adapter_dir / adapter_file
            if not adapter_path.exists():
                print(f"❌ Missing adapter file: {adapter_path}")
                return False

        print("✅ All required artifacts found")
        return True

    def create_archive(self, output_path: str = "deployment/model.tar.gz") -> str:
        """Create model.tar.gz archive"""
        if not self.validate_artifacts():
            raise ValueError("Missing required model artifacts")

        print(f"📦 Creating model archive: {output_path}")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with tarfile.open(output_path, "w:gz") as tar:
            # Add inference code from src path
            print("   📝 Adding inference code...")
            code_dir = Path("src/inference/sagemaker_code")
            for file_path in code_dir.iterdir():
                if file_path.is_file():
                    tar.add(str(file_path), arcname=f"code/{file_path.name}")

            # Add LoRA adapters
            print("   🎭 Adding LoRA adapters...")
            adapter_base = "models/trained"
            adapters = [
                "adapter_config.json",
                "adapter_model.safetensors",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
            ]

            for adapter in adapters:
                source_path = f"{adapter_base}/{adapter}"
                target_path = f"adapters/{adapter}"
                tar.add(source_path, arcname=target_path)

                # Log file size
                file_size = os.path.getsize(source_path)
                print(f"      {adapter}: {file_size / (1024 * 1024):.1f}MB")

        # Get final archive size
        archive_size = os.path.getsize(output_path)
        print(f"✅ Archive created: {archive_size / (1024 * 1024):.1f}MB")

        return output_path

    def upload_to_s3(
        self, archive_path: str, s3_key: str = "friday/model.tar.gz"
    ) -> str:
        """Upload model archive to S3"""
        s3_uri = f"s3://{self.bucket}/{s3_key}"

        print("☁️ Uploading to S3...")
        print(f"   Source: {archive_path}")
        print(f"   Target: {s3_uri}")

        try:
            # Upload with progress
            file_size = os.path.getsize(archive_path)
            print(f"   Size: {file_size / (1024 * 1024):.1f}MB")

            self.s3_client.upload_file(archive_path, self.bucket, s3_key)

            print("✅ Upload completed")
            return s3_uri

        except Exception as e:
            print(f"❌ Upload failed: {e}")
            raise

    def package_and_upload(
        self,
        archive_path: str = "deployment/model.tar.gz",
        s3_key: str = "friday/model.tar.gz",
    ) -> str:
        """Complete packaging and upload workflow"""
        print("🎭 Friday AI Model Packaging")
        print("=" * 50)

        try:
            # Create archive
            final_archive = self.create_archive(archive_path)

            # Upload to S3
            s3_uri = self.upload_to_s3(final_archive, s3_key)

            print("\n🎉 Model packaging completed!")
            print(f"📦 Archive: {final_archive}")
            print(f"☁️ S3 URI: {s3_uri}")
            print("🚀 Ready for SageMaker deployment")

            return s3_uri

        except Exception as e:
            print(f"❌ Packaging failed: {e}")
            raise


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Package Friday AI model for SageMaker"
    )
    parser.add_argument("--bucket", help="S3 bucket name (optional)")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument(
        "--archive", default="deployment/model.tar.gz", help="Archive output path"
    )
    parser.add_argument(
        "--s3-key", default="friday/model.tar.gz", help="S3 key for upload"
    )
    parser.add_argument("--no-upload", action="store_true", help="Skip S3 upload")

    args = parser.parse_args()

    packager = ModelPackager(bucket_name=args.bucket, region=args.region)

    if args.no_upload:
        # Just create the archive
        archive_path = packager.create_archive(args.archive)
        print(f"✅ Archive created: {archive_path}")
    else:
        # Full workflow
        s3_uri = packager.package_and_upload(args.archive, args.s3_key)
        print(f"✅ Model available at: {s3_uri}")


if __name__ == "__main__":
    main()
