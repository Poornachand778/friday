#!/usr/bin/env python3
"""
Deploy Friday Model to SageMaker Endpoint
==========================================

Deploys your fine-tuned Llama model for testing.

Cost: ~$1.20/hour for ml.g5.xlarge (24GB GPU)
      ~$2.00/hour for ml.g5.2xlarge (better for 8B model)

Usage:
    python scripts/deploy_friday_endpoint.py --deploy
    python scripts/deploy_friday_endpoint.py --test
    python scripts/deploy_friday_endpoint.py --delete  # IMPORTANT: Delete when done!
"""

import argparse
import json
import os
import sys
from pathlib import Path

import boto3

# Configuration
ENDPOINT_NAME = "friday-test-endpoint"
MODEL_NAME = "friday-core-8b"
INSTANCE_TYPE = "ml.g5.xlarge"  # ~$1.20/hr, 24GB VRAM

# Your model artifacts from training
MODEL_S3_URI = os.environ.get(
    "FRIDAY_MODEL_S3_URI",
    "s3://friday-ai-training/models/iteration1/",  # Update with your path
)

# HuggingFace base model
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"


def get_sagemaker_client():
    return boto3.client("sagemaker")


def get_runtime_client():
    return boto3.client("sagemaker-runtime")


def deploy_endpoint():
    """Deploy the Friday model to SageMaker"""
    sm = get_sagemaker_client()

    print(f"Deploying Friday model to endpoint: {ENDPOINT_NAME}")
    print(f"Instance type: {INSTANCE_TYPE}")
    print(f"Estimated cost: ~$1.20/hour")
    print()

    # Check if endpoint already exists
    try:
        sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
        print(f"Endpoint {ENDPOINT_NAME} already exists!")
        print("Use --delete first if you want to redeploy.")
        return
    except sm.exceptions.ClientError:
        pass  # Endpoint doesn't exist, good to proceed

    # HuggingFace LLM container
    from sagemaker.huggingface import HuggingFaceModel, get_huggingface_llm_image_uri
    import sagemaker

    role = os.environ.get("SAGEMAKER_ROLE")
    if not role:
        print("ERROR: Set SAGEMAKER_ROLE environment variable")
        sys.exit(1)

    session = sagemaker.Session()

    # Get the HuggingFace TGI container
    image_uri = get_huggingface_llm_image_uri(
        backend="huggingface", region=session.boto_region_name, version="2.0.0"
    )

    print(f"Using container: {image_uri}")

    # Environment for the model
    hub_env = {
        "HF_MODEL_ID": BASE_MODEL,
        "HF_TOKEN": os.environ.get("HUGGINGFACE_TOKEN", ""),
        "SM_NUM_GPUS": "1",
        "MAX_INPUT_LENGTH": "2048",
        "MAX_TOTAL_TOKENS": "4096",
        # If you have LoRA adapters, add:
        # "LORA_ADAPTERS": MODEL_S3_URI,
    }

    # Create model
    model = HuggingFaceModel(
        image_uri=image_uri,
        env=hub_env,
        role=role,
        name=f"{MODEL_NAME}-model",
    )

    print("Creating endpoint (this takes 5-10 minutes)...")

    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=INSTANCE_TYPE,
        endpoint_name=ENDPOINT_NAME,
        wait=True,
    )

    print()
    print("=" * 60)
    print("ENDPOINT DEPLOYED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Endpoint name: {ENDPOINT_NAME}")
    print(
        f"Endpoint URL: https://runtime.sagemaker.{session.boto_region_name}.amazonaws.com"
    )
    print()
    print("IMPORTANT: Remember to delete when done to avoid charges!")
    print("  python scripts/deploy_friday_endpoint.py --delete")
    print()

    return predictor


def test_endpoint():
    """Test the deployed endpoint"""
    runtime = get_runtime_client()

    print(f"Testing endpoint: {ENDPOINT_NAME}")
    print()

    # Test messages
    test_prompts = [
        "Hello Friday, introduce yourself.",
        "Boss, show me scene 5",
        "ఈ script లో conflict ఎలా build చేయాలి?",
    ]

    for prompt in test_prompts:
        print(f"User: {prompt}")

        payload = {
            "inputs": f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are Friday, Poorna's AI assistant. Address him as 'Boss'. Be concise and direct.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            "parameters": {
                "max_new_tokens": 256,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True,
            },
        }

        try:
            response = runtime.invoke_endpoint(
                EndpointName=ENDPOINT_NAME,
                ContentType="application/json",
                Body=json.dumps(payload),
            )

            result = json.loads(response["Body"].read().decode())
            generated = (
                result[0]["generated_text"]
                if isinstance(result, list)
                else result.get("generated_text", "")
            )

            # Extract just the assistant's response
            if "<|start_header_id|>assistant<|end_header_id|>" in generated:
                assistant_response = generated.split(
                    "<|start_header_id|>assistant<|end_header_id|>"
                )[-1]
                assistant_response = assistant_response.replace(
                    "<|eot_id|>", ""
                ).strip()
            else:
                assistant_response = generated

            print(f"Friday: {assistant_response}")
            print()

        except Exception as e:
            print(f"Error: {e}")
            print()

    print("Test complete!")


def delete_endpoint():
    """Delete the endpoint to stop charges"""
    sm = get_sagemaker_client()

    print(f"Deleting endpoint: {ENDPOINT_NAME}")

    try:
        # Delete endpoint
        sm.delete_endpoint(EndpointName=ENDPOINT_NAME)
        print("Endpoint deleted.")

        # Delete endpoint config
        sm.delete_endpoint_config(EndpointConfigName=ENDPOINT_NAME)
        print("Endpoint config deleted.")

        # Delete model
        sm.delete_model(ModelName=f"{MODEL_NAME}-model")
        print("Model deleted.")

        print()
        print("All resources cleaned up. No more charges.")

    except Exception as e:
        print(f"Error during cleanup: {e}")
        print("You may need to manually delete resources in AWS Console.")


def get_endpoint_url():
    """Get the endpoint URL for orchestrator config"""
    sm = get_sagemaker_client()
    session = boto3.session.Session()
    region = session.region_name

    try:
        sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
        print(f"Endpoint is active!")
        print()
        print("Add to your environment:")
        print(f"  export FRIDAY_LLM_BACKEND=sagemaker")
        print(f"  export FRIDAY_SAGEMAKER_ENDPOINT={ENDPOINT_NAME}")
        print(f"  export AWS_DEFAULT_REGION={region}")

    except sm.exceptions.ClientError:
        print(f"Endpoint {ENDPOINT_NAME} not found.")
        print("Deploy first with: python scripts/deploy_friday_endpoint.py --deploy")


def main():
    parser = argparse.ArgumentParser(description="Deploy Friday to SageMaker")
    parser.add_argument("--deploy", action="store_true", help="Deploy the endpoint")
    parser.add_argument("--test", action="store_true", help="Test the endpoint")
    parser.add_argument("--delete", action="store_true", help="Delete the endpoint")
    parser.add_argument("--status", action="store_true", help="Check endpoint status")
    args = parser.parse_args()

    if args.deploy:
        deploy_endpoint()
    elif args.test:
        test_endpoint()
    elif args.delete:
        delete_endpoint()
    elif args.status:
        get_endpoint_url()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
