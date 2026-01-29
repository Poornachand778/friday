#!/usr/bin/env python3
"""
Friday AI Smoke Test
Quick validation that the endpoint is working correctly
"""

import boto3
import json
import time
import os
from dotenv import load_dotenv

load_dotenv()


def run_smoke_test():
    """Run basic smoke test on Friday AI endpoint"""

    region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    endpoint_name = "friday-rt"

    print("🚀 Friday AI Smoke Test")
    print("=" * 50)
    print(f"🎯 Testing endpoint: {endpoint_name}")
    print(f"📍 Region: {region}")

    # Initialize SageMaker runtime with extended timeouts for HuggingFace DLC
    runtime = boto3.client(
        "sagemaker-runtime",
        region_name=region,
        config=boto3.session.Config(
            read_timeout=300,  # 5 minutes for HuggingFace model inference
            connect_timeout=30,
            retries={"max_attempts": 3, "mode": "standard"},
        ),
    )

    # Test payload
    test_payload = {
        "inputs": "Hello! How are you doing today?",
        "parameters": {
            "max_new_tokens": 100,
            "temperature": 0.7,
            "do_sample": True,
            "top_p": 0.9,
        },
    }

    print("\n🧪 Test Configuration:")
    print(f"   Input: {test_payload['inputs']}")
    print(f"   Max tokens: {test_payload['parameters']['max_new_tokens']}")
    print(f"   Temperature: {test_payload['parameters']['temperature']}")

    try:
        print("\n📤 Sending inference request...")
        print("⏱️ This may take 1-2 minutes for the first request (model warm-up)")

        start_time = time.time()

        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Body=json.dumps(test_payload),
        )

        end_time = time.time()
        latency = end_time - start_time

        # Parse response
        result = json.loads(response["Body"].read().decode())

        print(f"\n✅ SUCCESS! Request completed in {latency:.2f} seconds")
        print("=" * 50)
        print("📥 Response:")

        if isinstance(result, list) and len(result) > 0:
            generated_text = result[0].get("generated_text", str(result))
            print(f"   {generated_text}")
        elif isinstance(result, dict):
            if "generated_text" in result:
                print(f"   {result['generated_text']}")
            else:
                print(f"   {json.dumps(result, indent=2)}")
        else:
            print(f"   {result}")

        print("=" * 50)
        print(f"📊 Performance Metrics:")
        print(f"   Latency: {latency:.2f} seconds")
        print(f"   Response size: {len(str(result))} characters")
        print(f"   Status: Healthy ✅")

        return True

    except Exception as e:
        error_msg = str(e)
        print(f"\n❌ FAILED! Error: {error_msg}")

        if "timeout" in error_msg.lower():
            print("\n💡 Troubleshooting tips for timeouts:")
            print("   1. The model may still be loading (first request takes longest)")
            print("   2. Try a smaller max_new_tokens value")
            print("   3. Check CloudWatch logs for detailed error messages")
        elif "validation" in error_msg.lower():
            print("\n💡 This might be a request format issue:")
            print("   1. Check the input payload format")
            print("   2. Verify the model is expecting the correct input structure")

        return False


def check_endpoint_health():
    """Quick health check of the endpoint"""

    region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    endpoint_name = "friday-rt"

    sagemaker = boto3.client("sagemaker", region_name=region)

    try:
        response = sagemaker.describe_endpoint(EndpointName=endpoint_name)
        status = response["EndpointStatus"]

        print(f"🏥 Endpoint Health Check:")
        print(f"   Status: {status}")
        print(f"   ARN: {response['EndpointArn']}")

        for variant in response.get("ProductionVariants", []):
            print(f"   Variant: {variant['VariantName']}")
            print(
                f"   Instance Count: {variant['CurrentInstanceCount']}/{variant['DesiredInstanceCount']}"
            )
            print(f"   Weight: {variant['CurrentWeight']}")

        return status == "InService"

    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


if __name__ == "__main__":
    print("🔍 Starting Friday AI validation...")

    # Check endpoint health first
    if not check_endpoint_health():
        print("❌ Endpoint is not healthy. Aborting smoke test.")
        exit(1)

    # Run smoke test
    success = run_smoke_test()

    if success:
        print("\n🎉 SMOKE TEST PASSED!")
        print("🚀 Friday AI endpoint is ready for production use!")
    else:
        print("\n⚠️ SMOKE TEST FAILED!")
        print("🔧 Check the error messages above and CloudWatch logs.")

    exit(0 if success else 1)
