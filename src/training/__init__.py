"""
Friday AI Training Module
Model fine-tuning, multi-GPU training, and SageMaker integration

Status:
    DONE: SageMaker training pipeline
    DONE: LoRA fine-tuning configuration
    DONE: 120 interview exchanges collected
    REVIEW: Transformed interview data (behavioral not biographical)
    NEXT: Build iteration2 combined dataset (~525 examples)
    TODO: Curate 350 WhatsApp examples from high-quality pool
    TODO: Create 25 contrastive pairs for DPO-style training
    TODO: Expand MCP tool examples from 12 to 30
    FIXME: Set MODEL_NAME=meta-llama/Meta-Llama-3.1-8B-Instruct in .env
"""

__all__ = ["train_multigpu", "train_memory_diet", "vscode_sagemaker_trainer"]
