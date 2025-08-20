# Friday AI - Production Deployment Guide

## Quick Start

### Deploy SageMaker Endpoint

```bash
# Setup AWS secrets (one-time)
python src/deployment/setup_secrets.py

# Package and deploy the model
python src/deployment/package_model_artifacts.py
python src/deployment/deploy_friday_endpoint.py

# Test the deployment
python src/testing/smoke_test.py
```

### Organized Codebase Structure

```
src/
├── deployment/          # Production SageMaker deployment
│   ├── deploy_friday_endpoint.py    # Main deployment script
│   ├── package_model_artifacts.py   # Model packaging
│   ├── delete_friday_endpoint.py    # Cleanup script
│   └── setup_secrets.py             # AWS secrets management
├── training/            # Model fine-tuning
│   ├── train_multigpu.py            # Multi-GPU training
│   ├── train_memory_diet.py         # Memory-optimized training
│   └── vscode_sagemaker_trainer.py  # SageMaker training
├── data_processing/     # Data preparation
│   ├── convert_scene_to_chatml.py   # Format conversion
│   ├── ingest/                      # Data ingestion
│   └── clean/                       # Data cleaning
├── inference/           # Model serving
│   └── sagemaker_code/
│       └── inference.py             # SageMaker inference server
├── testing/             # Comprehensive testing
│   ├── smoke_test.py                # Endpoint validation
│   ├── launch_jupyter_test.py       # Jupyter testing
│   └── chat_friday.py               # Interactive testing
└── utils/               # Shared utilities
    └── setup_aws_interactive.py     # AWS configuration
```

## Production Features

- ✅ **Idempotent Deployment**: Safe to run multiple times
- ✅ **Auto-scaling**: 1-3 instances based on traffic
- ✅ **4-bit Quantization**: Memory optimized inference
- ✅ **PEFT LoRA Loading**: Efficient fine-tuned model serving
- ✅ **AWS Secrets Manager**: Secure credential management
- ✅ **Comprehensive Testing**: 6 test scenarios including edge cases
- ✅ **CloudWatch Logging**: Full observability
- ✅ **Error Handling**: Production-grade exception handling

## Usage

All Python files have been reorganized into the `src/` directory with proper module structure. Import paths have been preserved and the codebase is now production-ready with clean organization.

Run scripts from the project root:

```bash
python src/deployment/deploy_friday_endpoint.py
python src/testing/smoke_test.py
```
