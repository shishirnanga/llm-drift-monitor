"""
src/models/__init__.py

This file makes the models/ directory a Python package and defines
what gets exported when someone does:
    from src.models import OpenAIModel, AnthropicModel

WHY THIS FILE EXISTS:
In Python, a directory is only treated as a package if it contains __init__.py.
This file also lets you control what's "public" - what users of your package see.

Without this file:
    from src.models.openai_model import OpenAIModel  # Have to know the file name

With this file:
    from src.models import OpenAIModel  # Cleaner!
"""

"""Model wrappers package - supports multiple AI providers."""

from .base import BaseModel, ModelResponse
from .openai_model import OpenAIModel
from .anthropic_model import AnthropicModel

# Import new models with error handling
try:
    from .gpt4o_model import GPT4oModel, GPT35TurboModel
except ImportError:
    GPT4oModel = None
    GPT35TurboModel = None

try:
    from .gemini_model import GeminiModel, GeminiFlashModel
except ImportError:
    GeminiModel = None
    GeminiFlashModel = None

try:
    from .mistral_model import MistralModel
except ImportError:
    MistralModel = None

try:
    from .llama_model import LlamaModel
except ImportError:
    LlamaModel = None


def get_all_models() -> list[BaseModel]:
    """
    Get all available models based on which API keys are set.
    Only includes models where the API key is configured.
    """
    models = []
    
    # OpenAI GPT-4 Turbo (original)
    try:
        models.append(OpenAIModel())
        print("✅ Added: GPT-4 Turbo")
    except ValueError as e:
        print(f"⚠️  Skipping GPT-4 Turbo: {e}")
    
    # OpenAI GPT-4o
    if GPT4oModel:
        try:
            models.append(GPT4oModel())
            print("✅ Added: GPT-4o")
        except ValueError as e:
            print(f"⚠️  Skipping GPT-4o: {e}")
    
    # OpenAI GPT-3.5 Turbo
    if GPT35TurboModel:
        try:
            models.append(GPT35TurboModel())
            print("✅ Added: GPT-3.5 Turbo")
        except ValueError as e:
            print(f"⚠️  Skipping GPT-3.5 Turbo: {e}")
    
    # Anthropic Claude
    try:
        models.append(AnthropicModel())
        print("✅ Added: Claude Sonnet")
    except ValueError as e:
        print(f"⚠️  Skipping Claude: {e}")
    
    # Google Gemini Pro
    if GeminiModel:
        try:
            models.append(GeminiModel())
            print("✅ Added: Gemini 1.5 Pro")
        except ValueError as e:
            print(f"⚠️  Skipping Gemini Pro: {e}")
    
    # Google Gemini Flash
    if GeminiFlashModel:
        try:
            models.append(GeminiFlashModel())
            print("✅ Added: Gemini 1.5 Flash")
        except ValueError as e:
            print(f"⚠️  Skipping Gemini Flash: {e}")
    
    # Mistral Large
    if MistralModel:
        try:
            models.append(MistralModel())
            print("✅ Added: Mistral Large")
        except ValueError as e:
            print(f"⚠️  Skipping Mistral: {e}")
    
    # Llama 3.1 70B
    if LlamaModel:
        try:
            models.append(LlamaModel())
            print("✅ Added: Llama 3.1 70B")
        except ValueError as e:
            print(f"⚠️  Skipping Llama: {e}")
    
    if not models:
        raise ValueError("No models available. Check your API keys in .env file")
    
    print(f"\n🤖 Total models loaded: {len(models)}")
    return models


def get_model_by_name(name: str) -> BaseModel:
    """Get a specific model by name."""
    models = get_all_models()
    for model in models:
        if model.name == name:
            return model
    raise ValueError(f"Model '{name}' not found or not configured")


__all__ = [
    "BaseModel",
    "ModelResponse",
    "OpenAIModel",
    "AnthropicModel",
    "GPT4oModel",
    "GPT35TurboModel",
    "GeminiModel",
    "GeminiFlashModel",
    "MistralModel",
    "LlamaModel",
    "get_all_models",
    "get_model_by_name",
]