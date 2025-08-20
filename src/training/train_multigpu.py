#!/usr/bin/env python3
"""
Friday AI Fine-tuning - Multi-GPU Version
Run this to start multi-GPU training after gradient fix success
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from vscode_sagemaker_trainer import VSCodeSageMakerTrainer


def main():
    print("🚀 Friday AI - Multi-GPU Training")
    print("=" * 50)
    print("   Instance: ml.g5.12xlarge (4x A10G, 96GB total)")
    print("   Strategy: Use the proven gradient fix with more memory")

    # Set environment variable for instance type
    os.environ["INSTANCE_TYPE"] = "ml.g5.12xlarge"

    try:
        # Initialize trainer
        trainer = VSCodeSageMakerTrainer()

        # Upload data
        s3_inputs = trainer.upload_data()
        if not s3_inputs:
            print("❌ No data uploaded. Exiting.")
            return

        # Start training with multi-GPU settings
        estimator, job_name = trainer.create_training_job(
            s3_inputs,
            epochs=1,
            batch_size=2,  # Per device, so 2x4=8 effective batch size
            learning_rate=1e-4,
            instance_type="ml.g5.12xlarge",
        )

        # Monitor training
        trainer.monitor_training(estimator, job_name)

    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Check AWS credentials and permissions")


if __name__ == "__main__":
    main()
