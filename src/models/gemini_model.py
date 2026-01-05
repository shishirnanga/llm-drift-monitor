"""Google Gemini model wrapper."""

import os
import time
import google.generativeai as genai
from .base import BaseModel, ModelResponse


class GeminiModel(BaseModel):
    """Google Gemini Pro model wrapper."""
    
    def __init__(self, model_name: str = "gemini-pro-latest"):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self._model_name = model_name
    
    @property
    def name(self) -> str:
        if "pro" in self._model_name:
            return "Gemini 1.5 Pro"
        elif "flash" in self._model_name:
            return "Gemini 1.5 Flash"
        return f"Gemini ({self._model_name})"
    
    @property
    def model_id(self) -> str:
        return self._model_name
    
    def query(self, prompt: str, temperature: float = 0.0) -> ModelResponse:
        """Query Gemini model."""
        start_time = time.time()
        
        try:
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=1000,
            )
            
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            # Extract text
            if hasattr(response, 'text'):
                text = response.text
            else:
                text = str(response)
            
            # Estimate tokens (~4 chars per token)
            tokens = len(prompt + text) // 4
            
            return ModelResponse(
                model_name=self.name,
                prompt=prompt,
                response=text,
                latency_ms=latency_ms,
                tokens_input=len(prompt) // 4,  # Estimate
                tokens_output=len(text) // 4,    # Estimate
                tokens_total=len(prompt + text) // 4,
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


class GeminiFlashModel(GeminiModel):
    """Gemini Flash (faster) model wrapper."""
    
    def __init__(self):
        super().__init__(model_name="gemini-flash-latest")


if __name__ == "__main__":
    try:
        print("Testing Gemini Pro...")
        model = GeminiModel()
        response = model.query("What is 2+2?")
        print(f"✅ {response.model_name}: {response.response[:50]}")
        
        print("\nTesting Gemini Flash...")
        flash = GeminiFlashModel()
        response = flash.query("What is 3+3?")
        print(f"✅ {response.model_name}: {response.response[:50]}")
    except Exception as e:
        print(f"❌ Error: {e}")