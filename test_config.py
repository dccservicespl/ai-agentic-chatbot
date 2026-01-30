#!/usr/bin/env python3
"""Test script to verify LLM configuration and factory setup."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ai_agentic_chatbot.settings import get_settings
from ai_agentic_chatbot.infrastructure.llm.factory import get_llm_factory

def test_config_loading():
    """Test that configuration loads correctly."""
    print("Testing configuration loading...")
    
    try:
        settings = get_settings()
        print(f"✓ Settings loaded successfully")
        print(f"  Default model: {settings.default_model}")
        print(f"  Available models: {list(settings.models.keys())}")
        print(f"  Azure endpoint: {settings.azure.endpoint}")
        
        # Test model configs
        for model_key in settings.models.keys():
            model_config = settings.get_model_config(model_key)
            print(f"  Model '{model_key}': {model_config.model_name}")
            
    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
        return False
    
    return True

def test_llm_factory():
    """Test that LLM factory works correctly."""
    print("\nTesting LLM factory...")
    
    try:
        factory = get_llm_factory()
        print(f"✓ Factory created successfully")
        
        # Test getting available models
        available_models = factory.get_available_models()
        print(f"  Available models: {available_models}")
        
        # Test getting default LLM (don't actually create client to avoid API calls)
        print(f"✓ Factory methods accessible")
        
    except Exception as e:
        print(f"✗ LLM factory test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("=== LLM Configuration Test ===")
    
    config_ok = test_config_loading()
    factory_ok = test_llm_factory()
    
    if config_ok and factory_ok:
        print("\n✓ All tests passed! Configuration and factory are working correctly.")
    else:
        print("\n✗ Some tests failed. Check the errors above.")
        sys.exit(1)
