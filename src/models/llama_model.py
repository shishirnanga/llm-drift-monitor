"""Together.ai model wrapper (for Llama and others)."""

import os
import time
from together import Together
from .base import BaseModel, ModelResponse


class LlamaModel(BaseModel):
    """Llama model wrapper via Together.ai."""
    
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"):
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            raise ValueError("TOGETHER_API_KEY not found")
        
        self.client = Together(api_key=api_key)
        self._model_name = model_name
    
    @property
    def name(self) -> str:
        if "70b" in self._model_name.lower():
            return "Llama 3.1 70B"
        elif "8b" in self._model_name.lower():
            return "Llama 3.1 8B"
        elif "405b" in self._model_name.lower():
            return "Llama 3.1 405B"
        return "Llama 3.1"
    @property
    def model_id(self) -> str:
        return self._model_name
    
    def query(self, prompt: str, temperature: float = 0.0) -> ModelResponse:
        """Query Llama via Together.ai."""
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self._model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=1000
            )
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            text = response.choices[0].message.content
            tokens = response.usage.total_tokens if hasattr(response, 'usage') else 0
            
            return ModelResponse(
                model_name=self.name,
                prompt=prompt,
                response=text,
                latency_ms=latency_ms,
                tokens_input=response.usage.prompt_tokens if hasattr(response, 'usage') else 0,
                tokens_output=response.usage.completion_tokens if hasattr(response, 'usage') else 0,
                tokens_total=response.usage.total_tokens if hasattr(response, 'usage') else 0,
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
        print("Testing Llama...")
        model = LlamaModel()
        response = model.query("What is 2+2?")
        print(f"✅ {response.model_name}: {response.response[:50]}")
    except Exception as e:
        print(f"❌ Error: {e}")