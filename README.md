# AI Agentic Chatbot

A multi-provider LLM factory system with LangChain compatibility, supporting dynamic provider addition and type-safe configuration.

## Features

- 🏭 **Multi-Provider Support**: Azure OpenAI, OpenAI, Anthropic, AWS Bedrock, and extensible for new providers
- 🔧 **LangChain Compatible**: All LLMs return `BaseChatModel` instances
- 🛡️ **Type Safety**: Enum-based providers and model types with Pydantic validation
- ⚙️ **Environment Overrides**: Secure credential management via environment variables
- 🔄 **Dynamic Extension**: Easy addition of new providers at runtime
- 📦 **Singleton Factory**: Efficient client caching and management

## Quick Start

### Basic Usage

```python
from ai_agentic_chatbot.infrastructure.llm.factory import get_llm
from ai_agentic_chatbot.types import ModelType

# Get default model
llm = get_llm()

# Get specific model type
fast_llm = get_llm(model=ModelType.FAST)
smart_llm = get_llm(model=ModelType.SMART)

# Use with LangChain
response = llm.invoke("Hello, world!")
```

### Configuration

The system uses a `config.yaml` file with the following structure:

```yaml
llm:
  default: azure_openai.fast # provider.model format

  azure_openai:
    fast:
      model_name: "gpt-4o-mini"
      api_key: "your-azure-openai-api-key"
      endpoint: "https://your-endpoint.cognitiveservices.azure.com"
      api_version: "2024-02-15-preview"
      temperature: 0.0
      max_tokens: 4000
      timeout: 60
      max_retries: 3
    smart:
      model_name: "gpt-4o"
      # ... other config
```

## Adding New LLM Providers

### Step 1: Define Provider Enum

Add your provider to `src/ai_agentic_chatbot/types.py`:

```python
class LLMProvider(Enum):
    AZURE_OPENAI = "azure_openai"
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"  # New provider
    # ... existing providers
```

### Step 2: Create Configuration Class

Add a configuration class in `src/ai_agentic_chatbot/config.py`:

```python
class HuggingFaceConfig(BaseLLMConfig):
    """Configuration for HuggingFace models."""

    api_key: str = Field(..., description="HuggingFace API key")
    model_id: str = Field(..., description="HuggingFace model ID")
    task: str = Field(default="text-generation", description="HuggingFace task")

    def get_client_kwargs(self) -> Dict[str, Any]:
        """Get HuggingFace client initialization arguments."""
        return {
            "model": self.model_name,
            "huggingfacehub_api_token": self.api_key,
            "model_kwargs": {
                "temperature": self.temperature,
                "max_length": self.max_tokens,
            },
            "task": self.task,
        }

# Register the configuration class
PROVIDER_CONFIG_REGISTRY[LLMProvider.HUGGINGFACE] = HuggingFaceConfig
```

### Step 3: Add Factory Support

Update the factory in `src/ai_agentic_chatbot/infrastructure/llm/factory.py`:

```python
# Add optional import
try:
    from langchain_community.llms import HuggingFacePipeline
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HuggingFacePipeline = None
    HUGGINGFACE_AVAILABLE = False

class LLMFactory:
    def _create_client(self, model_config: ModelConfiguration) -> LangChainLLM:
        """Create a LangChain-compatible client based on provider type."""
        provider = model_config.provider
        config = model_config.config

        if provider == LLMProvider.AZURE_OPENAI:
            return self._create_azure_openai_client(config)
        elif provider == LLMProvider.OPENAI:
            return self._create_openai_client(config)
        elif provider == LLMProvider.HUGGINGFACE:  # New provider
            return self._create_huggingface_client(config)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _create_huggingface_client(self, config: HuggingFaceConfig) -> HuggingFacePipeline:
        """Create HuggingFace LangChain client."""
        if not HUGGINGFACE_AVAILABLE:
            raise ImportError(
                "langchain-community is required for HuggingFace models. "
                "Install with: pip install langchain-community"
            )
        return HuggingFacePipeline(**config.get_client_kwargs())
```

### Step 4: Update Configuration

Add your provider to `config.yaml`:

```yaml
llm:
  default: huggingface.fast

  huggingface:
    fast:
      model_name: "microsoft/DialoGPT-medium"
      api_key: "your-huggingface-api-key"
      model_id: "microsoft/DialoGPT-medium"
      task: "text-generation"
      temperature: 0.7
      max_tokens: 1000
```

### Step 5: Environment Variables

Add environment variable support in `src/ai_agentic_chatbot/settings.py`:

```python
@staticmethod
def _apply_env_overrides(model_data: Dict[str, Any], provider: LLMProvider) -> Dict[str, Any]:
    """Apply environment variable overrides to model configuration."""
    model_data = model_data.copy()

    if provider == LLMProvider.HUGGINGFACE:
        if "HUGGINGFACE_API_KEY" in os.environ:
            model_data["api_key"] = os.environ["HUGGINGFACE_API_KEY"]
    # ... existing overrides

    return model_data
```

## Environment Variables

Set these environment variables for secure credential management:

```bash
# Azure OpenAI
export AZURE_OPENAI_API_KEY="your-azure-key"
export AZURE_OPENAI_ENDPOINT="https://your-endpoint.cognitiveservices.azure.com"
export AZURE_OPENAI_API_VERSION="2024-02-15-preview"

# OpenAI
export OPENAI_API_KEY="your-openai-key"
export OPENAI_ORGANIZATION="your-org-id"

# Anthropic
export ANTHROPIC_API_KEY="your-anthropic-key"

# AWS Bedrock
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="us-east-1"

# HuggingFace (example)
export HUGGINGFACE_API_KEY="your-hf-token"
```

## Configuration Examples

### Azure OpenAI

```yaml
azure_openai:
  fast:
    model_name: "gpt-4o-mini"
    api_key: "your-key" # Or use AZURE_OPENAI_API_KEY env var
    endpoint: "https://your-endpoint.cognitiveservices.azure.com"
    api_version: "2024-02-15-preview"
    temperature: 0.0
    max_tokens: 4000
  smart:
    model_name: "gpt-4o"
    temperature: 0.1
    max_tokens: 8000
```

### OpenAI Direct

```yaml
openai:
  fast:
    model_name: "gpt-4o-mini"
    api_key: "your-key" # Or use OPENAI_API_KEY env var
    temperature: 0.0
    max_tokens: 4000
  smart:
    model_name: "gpt-4o"
    temperature: 0.1
    max_tokens: 8000
```

### Anthropic Claude

```yaml
anthropic:
  fast:
    model_name: "claude-3-haiku-20240307"
    api_key: "your-key" # Or use ANTHROPIC_API_KEY env var
    temperature: 0.0
    max_tokens: 4000
  smart:
    model_name: "claude-3-5-sonnet-20241022"
    temperature: 0.1
    max_tokens: 8000
```

### AWS Bedrock

```yaml
aws_bedrock:
  fast:
    model_name: "anthropic.claude-3-haiku-20240307-v1:0"
    region_name: "us-east-1" # Or use AWS_DEFAULT_REGION env var
    temperature: 0.0
    max_tokens: 4000
  smart:
    model_name: "anthropic.claude-3-5-sonnet-20241022-v2:0"
    temperature: 0.1
    max_tokens: 8000
```

## Model Types

The system supports these model types:

- `FAST`: Quick, cost-effective models for simple tasks
- `SMART`: Advanced models for complex reasoning
- `EMBEDDING`: Text embedding models
- `VISION`: Multimodal models with vision capabilities

## Advanced Usage

### Factory Methods

```python
from ai_agentic_chatbot.infrastructure.llm.factory import get_llm_factory

factory = get_llm_factory()

# Get available models
models = factory.get_available_models()

# Get supported providers
providers = factory.get_supported_providers()

# Clear cache
factory.clear_cache()

# Reload settings
factory.reload_settings()
```

### Custom Provider Integration

For providers not yet supported, you can extend the system by:

1. Adding the provider enum
2. Creating a configuration class
3. Implementing the factory method
4. Registering the configuration class

The system is designed for easy extension without modifying core functionality.

## Dependencies

### Core Dependencies

```bash
pip install pydantic pyyaml langchain-core
```

### Provider-Specific Dependencies

```bash
# Azure OpenAI
pip install langchain-openai

# Anthropic
pip install langchain-anthropic

# AWS Bedrock
pip install langchain-aws

# HuggingFace (example)
pip install langchain-community transformers
```

## Development

### Running Tests

```bash
python test_factory.py
```

### Project Structure

```
src/ai_agentic_chatbot/
├── types.py                    # Provider and model type enums
├── config.py                   # Configuration classes
├── settings.py                 # Settings management
└── infrastructure/llm/
    └── factory.py              # LLM factory implementation
```

## Contributing

When adding new providers:

1. Follow the type-safe pattern with enums
2. Implement proper error handling
3. Add environment variable support
4. Include configuration examples
5. Update this README

## License

[Your License Here]
