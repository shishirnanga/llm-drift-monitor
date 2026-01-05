"""OpenAI GPT-4o model wrapper."""

import os
import time
from openai import OpenAI
from .base import BaseModel, ModelResponse


class GPT4oModel(BaseModel):
    """GPT-4o model wrapper."""
    
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found")
        
        self.client = OpenAI(api_key=api_key)
        self._model_name = "gpt-4o"
    
    @property
    def name(self) -> str:
        return "GPT-4o"
    
    @property
    def model_id(self) -> str:
        return "gpt-4o"
    
    def query(self, prompt: str, temperature: float = 0.0) -> ModelResponse:
        """Query GPT-4o model."""
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


class GPT35TurboModel(BaseModel):
    """GPT-3.5 Turbo model wrapper."""
    
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found")
        
        self.client = OpenAI(api_key=api_key)
        self._model_name = "gpt-3.5-turbo"
    
    @property
    def name(self) -> str:
        return "GPT-3.5 Turbo"
    
    @property
    def model_id(self) -> str:
        return "gpt-3.5-turbo"  # ADD THIS
    
    def query(self, prompt: str, temperature: float = 0.0) -> ModelResponse:
        """Query GPT-3.5 Turbo model."""
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
        print("Testing GPT-4o...")
        model = GPT4oModel()
        response = model.query("What is 2+2?")
        print(f"✅ {response.model_name}: {response.response[:50]}")
        
        print("\nTesting GPT-3.5 Turbo...")
        model35 = GPT35TurboModel()
        response = model35.query("What is 3+3?")
        print(f"✅ {response.model_name}: {response.response[:50]}")
    except Exception as e:
        print(f"❌ Error: {e}")