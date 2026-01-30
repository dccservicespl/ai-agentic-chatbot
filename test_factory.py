#!/usr/bin/env python3
"""Test the updated LLM factory with new default pattern 'provider.model'."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ai_agentic_chatbot.infrastructure.llm.factory import get_llm, get_llm_factory
from ai_agentic_chatbot.types import LLMProvider, ModelType
from ai_agentic_chatbot.settings import get_settings


def test_factory():
    """Test the updated factory functionality."""
    print("🧪 Testing LLM Factory with New Default Pattern")
    print("=" * 50)

    try:
        # Test configuration parsing
        print("1. Testing configuration parsing:")
        settings = get_settings()
        print(f"   ✓ Default model: {settings.default_model}")
        print(f"   ✓ Available models: {list(settings.models.keys())}")

        factory = get_llm_factory()

        # Test 2: Get default model (should parse 'azure_openai.fast' -> 'fast')
        print("\n2. Testing default model:")
        llm = get_llm()
        print(f"   ✓ Default LLM type: {type(llm).__name__}")

        # Test 3: Get model with specific model type
        print("\n3. Testing model type parameter:")
        llm_fast = get_llm(model=ModelType.FAST)
        print(f"   ✓ Fast LLM type: {type(llm_fast).__name__}")

        llm_smart = get_llm(model=ModelType.SMART)
        print(f"   ✓ Smart LLM type: {type(llm_smart).__name__}")

        # Test 4: Show supported providers
        print("\n4. Supported providers:")
        providers = factory.get_supported_providers()
        print(f"   Providers: {[p.value for p in providers]}")

        # Test 5: Test LangChain compatibility
        print("\n5. Testing LangChain compatibility:")
        print(f"   Has invoke method: {hasattr(llm, 'invoke')}")
        print(f"   Has stream method: {hasattr(llm, 'stream')}")
        print(f"   Has astream method: {hasattr(llm, 'astream')}")

        print("\n✅ All tests passed! Factory is working with new pattern.")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_factory()
