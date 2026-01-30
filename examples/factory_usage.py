#!/usr/bin/env python3
"""
Simple example showing how to use the LLM factory with LangChain-compatible types.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_agentic_chatbot.infrastructure.llm.factory import (
    get_llm,
    get_azure_llm,
    get_openai_llm,
    get_llm_factory
)


def main():
    """Demonstrate factory usage with LangChain-compatible types."""
    
    print("🚀 LLM Factory Usage Examples")
    print("=" * 40)
    
    try:
        factory = get_llm_factory()
        
        # Show available models
        print(f"Available models: {factory.get_available_models()}")
        print(f"Supported providers: {[p.value for p in factory.get_supported_providers()]}")
        
        # Get default LLM (LangChain-compatible)
        print("\n1. Getting default LLM:")
        llm = get_llm()
        print(f"   Type: {type(llm).__name__}")
        print(f"   LangChain compatible: {hasattr(llm, 'invoke')}")
        
        # Get specific model
        print("\n2. Getting specific model:")
        fast_llm = get_llm("fast")
        print(f"   Type: {type(fast_llm).__name__}")
        
        # Get Azure-specific LLM
        print("\n3. Getting Azure LLM:")
        try:
            azure_llm = get_azure_llm("fast")
            print(f"   Type: {type(azure_llm).__name__}")
            print(f"   Azure deployment: {azure_llm.deployment_name}")
        except ValueError as e:
            print(f"   Error: {e}")
        
        # Usage with LangChain
        print("\n4. LangChain usage example:")
        print("   # All returned LLMs are LangChain-compatible")
        print("   # llm = get_llm('fast')")
        print("   # response = llm.invoke('Hello, world!')")
        print("   # chain = llm | output_parser")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure config.yaml has at least one provider configured")


if __name__ == "__main__":
    main()
