"""Mistral AI model wrapper."""

import os
import time
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from .base import BaseModel, ModelResponse


class MistralModel(BaseModel):
    """Mistral model wrapper."""
    
    def __init__(self, model_name: str = "mistral-large-latest"):
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY not found")
        
        self.client = MistralClient(api_key=api_key)
        self._model_name = model_name
    
    @property
    def name(self) -> str:
        if "large" in self._model_name:
            return "Mistral Large"
        elif "medium" in self._model_name:
            return "Mistral Medium"
        elif "small" in self._model_name:
            return "Mistral Small"
        return f"Mistral ({self._model_name})"
    @property
    def model_id(self) -> str:
        return self._model_name
    
    def query(self, prompt: str, temperature: float = 0.0) -> ModelResponse:
        """Query Mistral model."""
        start_time = time.time()
        
        try:
            messages = [ChatMessage(role="user", content=prompt)]
            
            response = self.client.chat(
                model=self._model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=1000
            )
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            text = response.choices[0].message.content
            tokens = response.usage.total_tokens
            
            return ModelResponse(
                model_name=self.name,
                prompt=prompt,
                response=text,
                latency_ms=latency_ms,
                tokens_input=response.usage.prompt_tokens,
                tokens_output=response.usage.completion_tokens,
                tokens_total=response.usage.total_tokens,
                success=True,
                error=None,
            )
            
        except Exception as e:
            return ModelResponse(
                model_name=self.name,
                prompt=prompt,
                response=f"ERROR: {str(e)}",
                latency_ms=int((time.time() - start_time) * 1000),
                tokens_input=0,
                tokens_output=0,
                tokens_total=0,
                success=False,
                error=str(e),
            )


if __name__ == "__main__":
    try:
        print("Testing Mistral...")
        model = MistralModel()
        response = model.query("What is 2+2?")
        print(f"✅ {response.model_name}: {response.response[:50]}")
    except Exception as e:
        print(f"❌ Error: {e}")