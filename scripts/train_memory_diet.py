#!/usr/bin/env python3
"""
Friday AI Fine-tuning - Memory Diet Version  
Run this for ultra-conservative memory usage on single GPU
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from vscode_sagemaker_trainer import VSCodeSageMakerTrainer


def main():
    print("🥗 Friday AI - Memory Diet Training")
    print("=" * 50)
    print("   Instance: ml.g5.2xlarge (single A10G, 24GB)")
    print("   Strategy: Minimal memory usage, proven gradient fix")

    try:
        # Initialize trainer
        trainer = VSCodeSageMakerTrainer()

        # Upload data
        s3_inputs = trainer.upload_data()
        if not s3_inputs:
            print("❌ No data uploaded. Exiting.")
            return

        # Start training with ultra-conservative settings
        estimator, job_name = trainer.create_training_job(
            s3_inputs,
            epochs=1,
            batch_size=1,  # Minimal batch size
            learning_rate=1e-4,
            max_length=512,  # Conservative sequence length
            instance_type="ml.g5.2xlarge",  # Single GPU
        )

        # Monitor training
        trainer.monitor_training(estimator, job_name)

    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Check AWS credentials and permissions")


if __name__ == "__main__":
    main()
