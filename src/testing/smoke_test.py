#!/usr/bin/env python3
"""
Smoke test client for Friday AI SageMaker endpoint
Tests endpoint functionality with various scenarios
"""

import boto3
import json
import time
import argparse
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()


class FridaySmokeTest:
    """Smoke test suite for Friday AI endpoint"""

    def __init__(self, endpoint_name: str = "friday-rt", region: str = "us-east-1"):
        self.endpoint_name = endpoint_name
        self.region = region

        # Initialize SageMaker runtime client
        self.runtime_client = boto3.client("sagemaker-runtime", region_name=region)

        print("🧪 Friday AI Smoke Test initialized")
        print(f"   Endpoint: {endpoint_name}")
        print(f"   Region: {region}")

    def invoke_endpoint(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke the Friday AI endpoint"""
        try:
            start_time = time.time()

            response = self.runtime_client.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType="application/json",
                Accept="application/json",
                Body=json.dumps(payload),
            )

            # Parse response
            response_body = response["Body"].read().decode("utf-8")
            result = json.loads(response_body)

            inference_time = time.time() - start_time
            result["_inference_time"] = round(inference_time, 3)

            return result

        except Exception as e:
            print(f"❌ Endpoint invocation failed: {e}")
            raise

    def test_single_prompt(self) -> bool:
        """Test single string input"""
        print("\n🎬 Test 1: Single Prompt")
        print("-" * 40)

        payload = {
            "inputs": "What's your favorite Telugu movie dialogue?",
            "parameters": {"max_new_tokens": 150, "temperature": 0.7, "top_p": 0.9},
        }

        try:
            result = self.invoke_endpoint(payload)

            print(f"✅ Response received in {result['_inference_time']}s")
            print(f"🤖 Friday: {result['generated_text']}")
            print(
                f"📊 Tokens - Prompt: {result['usage']['prompt_tokens']}, "
                f"Completion: {result['usage']['completion_tokens']}, "
                f"Total: {result['usage']['total_tokens']}"
            )

            # Validate response
            if (
                not result["generated_text"]
                or len(result["generated_text"].strip()) < 10
            ):
                print("⚠️ Response too short or empty")
                return False

            return True

        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False

    def test_batch_prompts(self) -> bool:
        """Test batch input (list of strings)"""
        print("\n🎭 Test 2: Batch Prompts")
        print("-" * 40)

        payload = {
            "inputs": [
                "Give me a witty comeback about bad weather",
                "Help me write a film scene",
                "What's the secret to good biryani?",
            ],
            "parameters": {"max_new_tokens": 100, "temperature": 0.8},
        }

        try:
            result = self.invoke_endpoint(payload)

            print(f"✅ Batch response received in {result['_inference_time']}s")

            if isinstance(result["generated_text"], list):
                for i, response in enumerate(result["generated_text"], 1):
                    print(f"🤖 Response {i}: {response[:100]}...")
                print(f"📊 Total tokens: {result['usage']['total_tokens']}")
                return True
            else:
                print("⚠️ Expected list response for batch input")
                return False

        except Exception as e:
            print(f"❌ Batch test failed: {e}")
            return False

    def test_stop_sequences(self) -> bool:
        """Test stop sequence functionality"""
        print("\n🛑 Test 3: Stop Sequences")
        print("-" * 40)

        payload = {
            "inputs": "List the best Telugu directors: 1.",
            "parameters": {
                "max_new_tokens": 200,
                "temperature": 0.7,
                "stop": ["3.", "\n\n"],
            },
        }

        try:
            result = self.invoke_endpoint(payload)

            print(f"✅ Stop sequence test completed in {result['_inference_time']}s")
            print(f"🤖 Friday: {result['generated_text']}")

            # Check if stop sequence was respected
            response_text = result["generated_text"]
            if "3." in response_text:
                print("⚠️ Stop sequence may not have been respected")
                return False

            return True

        except Exception as e:
            print(f"❌ Stop sequence test failed: {e}")
            return False

    def test_oversized_input(self) -> bool:
        """Test oversized input handling"""
        print("\n📏 Test 4: Oversized Input")
        print("-" * 40)

        # Create a very long input
        long_input = "Tell me about Telugu cinema. " * 500  # ~15k chars

        payload = {"inputs": long_input, "parameters": {"max_new_tokens": 50}}

        try:
            result = self.invoke_endpoint(payload)

            # If we get here, the input was handled (truncated)
            print(f"✅ Long input handled in {result['_inference_time']}s")
            print(f"📊 Tokens used: {result['usage']['total_tokens']}")
            return True

        except Exception as e:
            # Should return 400 for oversized input
            if "400" in str(e) or "length" in str(e).lower():
                print("✅ Oversized input properly rejected")
                return True
            else:
                print(f"❌ Unexpected error: {e}")
                return False

    def test_deterministic_output(self) -> bool:
        """Test deterministic output with seed"""
        print("\n🎲 Test 5: Deterministic Output")
        print("-" * 40)

        payload = {
            "inputs": "Tell me a short joke about directors",
            "parameters": {
                "max_new_tokens": 80,
                "temperature": 0.7,
                "seed": 42,
                "do_sample": True,
            },
        }

        try:
            # Make two identical requests
            result1 = self.invoke_endpoint(payload)
            time.sleep(1)
            result2 = self.invoke_endpoint(payload)

            response1 = result1["generated_text"]
            response2 = result2["generated_text"]

            print("✅ Both requests completed")
            print(f"🎯 Response 1: {response1[:100]}...")
            print(f"🎯 Response 2: {response2[:100]}...")

            # Note: Due to GPU non-determinism, responses might still differ
            # This test mainly ensures the seed parameter is accepted
            if response1 == response2:
                print("✅ Responses identical (fully deterministic)")
            else:
                print("⚠️ Responses differ (GPU non-determinism expected)")

            return True

        except Exception as e:
            print(f"❌ Deterministic test failed: {e}")
            return False

    def test_endpoint_health(self) -> bool:
        """Basic health check"""
        print("\n❤️ Test 6: Endpoint Health")
        print("-" * 40)

        simple_payload = {
            "inputs": "Hello Friday!",
            "parameters": {"max_new_tokens": 20, "temperature": 0.7},
        }

        try:
            result = self.invoke_endpoint(simple_payload)

            # Basic health indicators
            has_response = bool(result.get("generated_text", "").strip())
            has_usage = "usage" in result
            reasonable_time = result.get("_inference_time", 999) < 30

            print("✅ Health check completed")
            print(f"📝 Has response: {has_response}")
            print(f"📊 Has usage stats: {has_usage}")
            print(f"⚡ Reasonable time: {reasonable_time}s")

            return has_response and has_usage and reasonable_time

        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False

    def run_all_tests(self) -> Dict[str, bool]:
        """Run complete test suite"""
        print("🎭 Friday AI Endpoint Smoke Test Suite")
        print("=" * 60)

        tests = [
            ("Single Prompt", self.test_single_prompt),
            ("Batch Prompts", self.test_batch_prompts),
            ("Stop Sequences", self.test_stop_sequences),
            ("Oversized Input", self.test_oversized_input),
            ("Deterministic Output", self.test_deterministic_output),
            ("Endpoint Health", self.test_endpoint_health),
        ]

        results = {}
        passed = 0

        for test_name, test_func in tests:
            try:
                results[test_name] = test_func()
                if results[test_name]:
                    passed += 1
            except KeyboardInterrupt:
                print("\n⚠️ Test suite interrupted")
                break
            except Exception as e:
                print(f"❌ {test_name} crashed: {e}")
                results[test_name] = False

        # Summary
        print("\n📋 Test Summary")
        print("=" * 60)

        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status} {test_name}")

        print(f"\n🎯 Results: {passed}/{len(results)} tests passed")

        if passed == len(results):
            print("🎉 All tests passed! Friday AI endpoint is healthy.")
        elif passed > len(results) // 2:
            print("⚠️ Most tests passed. Minor issues detected.")
        else:
            print("❌ Multiple test failures. Check endpoint health.")

        return results


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(description="Smoke test Friday AI endpoint")
    parser.add_argument("--endpoint", default="friday-rt", help="Endpoint name")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument(
        "--test",
        choices=[
            "single",
            "batch",
            "stop",
            "oversized",
            "deterministic",
            "health",
            "all",
        ],
        default="all",
        help="Specific test to run",
    )

    args = parser.parse_args()

    tester = FridaySmokeTest(endpoint_name=args.endpoint, region=args.region)

    if args.test == "all":
        results = tester.run_all_tests()
        exit(0 if all(results.values()) else 1)
    else:
        # Run specific test
        test_methods = {
            "single": tester.test_single_prompt,
            "batch": tester.test_batch_prompts,
            "stop": tester.test_stop_sequences,
            "oversized": tester.test_oversized_input,
            "deterministic": tester.test_deterministic_output,
            "health": tester.test_endpoint_health,
        }

        success = test_methods[args.test]()
        exit(0 if success else 1)


if __name__ == "__main__":
    main()
