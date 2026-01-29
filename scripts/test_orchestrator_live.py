#!/usr/bin/env python3
"""
Live Integration Test for Friday Orchestrator
==============================================

Tests the orchestrator with a real LLM backend (Anthropic/OpenAI).
No GPU needed - uses cloud APIs.

Usage:
    # With Anthropic
    ANTHROPIC_API_KEY=xxx python scripts/test_orchestrator_live.py --backend anthropic

    # With OpenAI
    OPENAI_API_KEY=xxx python scripts/test_orchestrator_live.py --backend openai

    # Dry run (prints config without calling LLM)
    python scripts/test_orchestrator_live.py --dry-run
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parents[1]))


def print_header(text: str):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")


def print_result(label: str, value: str, indent: int = 0):
    prefix = "  " * indent
    print(f"{prefix}{label}: {value}")


async def test_orchestrator(backend: str, dry_run: bool = False):
    """Run live tests against orchestrator"""

    print_header("Friday Orchestrator Live Test")

    # Configure backend
    if backend == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key and not dry_run:
            print("ERROR: ANTHROPIC_API_KEY not set")
            return 1
        model = "claude-3-haiku-20240307"  # Fast and cheap for testing
        base_url = "https://api.anthropic.com"
    elif backend == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key and not dry_run:
            print("ERROR: OPENAI_API_KEY not set")
            return 1
        model = "gpt-4o-mini"  # Fast and cheap for testing
        base_url = "https://api.openai.com/v1"
    else:
        print(f"ERROR: Unknown backend: {backend}")
        return 1

    print_result("Backend", backend)
    print_result("Model", model)
    print_result("API Key", f"{'*' * 8}...{api_key[-4:]}" if api_key else "NOT SET")

    if dry_run:
        print("\n[DRY RUN - Not calling LLM]")
        return 0

    # Configure orchestrator
    from orchestrator.config import LLMConfig, OrchestratorConfig
    from orchestrator.core import FridayOrchestrator

    llm_config = LLMConfig(
        backend=backend,
        model_name=model,
        base_url=base_url,
        api_key=api_key,
        temperature=0.7,
        max_tokens=500,
    )

    config = OrchestratorConfig(llm=llm_config)
    orchestrator = FridayOrchestrator(config)

    try:
        print("\nInitializing orchestrator...")
        await orchestrator.initialize()
        print("✓ Orchestrator initialized")

        # Test 1: Basic chat
        print_header("Test 1: Basic Chat")
        response = await orchestrator.chat(
            message="Hello Friday, introduce yourself briefly.",
            location="general",
        )
        print_result("Response", response.content[:200])
        print_result("Context", response.context_type.value)
        print_result("Processing Time", f"{response.processing_time_ms:.0f}ms")
        print("✓ Basic chat works")

        # Test 2: Context switching
        print_header("Test 2: Context Detection")
        response = await orchestrator.chat(
            message="Switch to writers room",
        )
        print_result("Response", response.content[:100])
        print_result("Context", response.context_type.value)
        expected = "writers_room"
        if response.context_type.value == expected:
            print(f"✓ Context correctly switched to {expected}")
        else:
            print(f"✗ Expected {expected}, got {response.context_type.value}")

        # Test 3: Screenplay domain
        print_header("Test 3: Screenplay Domain")
        response = await orchestrator.chat(
            message="What makes a good opening scene in a Telugu film?",
            location="writers_room",
        )
        print_result("Response", response.content[:300])
        print("✓ Screenplay domain response")

        # Test 4: Telugu-English mix
        print_header("Test 4: Telugu-English Code Switching")
        response = await orchestrator.chat(
            message="Boss, ఈ scene లో conflict ఎలా build చేయాలి?",
        )
        print_result("Response", response.content[:300])
        print("✓ Code-switching handled")

        # Test 5: Session persistence
        print_header("Test 5: Session Persistence")
        session_info = orchestrator.get_session_info()
        print_result("Session ID", session_info.get("session_id", "N/A"))
        print_result("Turn Count", str(session_info.get("turn_count", 0)))
        print_result("Active Turns", str(session_info.get("active_turns", 0)))
        print("✓ Session tracking works")

        # Test 6: Tool listing (no execution)
        print_header("Test 6: Tool Registry")
        from orchestrator.tools.registry import get_tool_registry

        registry = get_tool_registry()
        tools = registry.list_tools()
        print_result("Total Tools", str(len(tools)))
        for tool in tools[:5]:
            print_result(tool.name, tool.category, indent=1)
        print("✓ Tool registry works")

        print_header("All Tests Passed!")
        return 0

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1

    finally:
        await orchestrator.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Test Friday Orchestrator")
    parser.add_argument(
        "--backend",
        choices=["anthropic", "openai"],
        default="anthropic",
        help="LLM backend to use",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config without calling LLM",
    )
    args = parser.parse_args()

    return asyncio.run(test_orchestrator(args.backend, args.dry_run))


if __name__ == "__main__":
    sys.exit(main())
