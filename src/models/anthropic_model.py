import os
import time
from typing import Optional

from dotenv import load_dotenv
from anthropic import Anthropic

from .base import BaseModel, ModelResponse

load_dotenv()


class AnthropicModel(BaseModel):
    """
    Wrapper for Anthropic's Claude models.
    
    Example usage:
        model = AnthropicModel()  # Uses default Claude Sonnet
        response = model.query("What is the capital of France?")
        print(response.response)  # "The capital of France is Paris."
        
        # Or specify a different model
        model = AnthropicModel(model_id="claude-opus-4-20250514")
    """
    
    def __init__(
        self,
        model_id: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
        max_tokens: int = 1000
    ):
        """
        Initialize the Anthropic model wrapper.
        
        Args:
            model_id: Which Claude model to use. Options include:
                - "claude-sonnet-4-20250514" (balanced - recommended)
                - "claude-opus-4-20250514" (most capable)
                - "claude-haiku-4-20250514" (fastest, cheapest)
            api_key: Anthropic API key. If None, reads from env var
            max_tokens: Maximum tokens in the response
        """
        self._model_id = model_id
        self._max_tokens = max_tokens
        
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        
        if not self._api_key:
            raise ValueError(
                "Anthropic API key not found. Either:\n"
                "1. Pass api_key parameter\n"
                "2. Set ANTHROPIC_API_KEY in your .env file\n"
                "3. Set ANTHROPIC_API_KEY environment variable"
            )
        
        # Create the Anthropic client
        self._client = Anthropic(api_key=self._api_key)
    
    @property
    def name(self) -> str:
        """Human-readable name."""
        names = {
            "claude-sonnet-4-20250514": "Claude Sonnet 4",
            "claude-opus-4-20250514": "Claude Opus 4",
            "claude-haiku-4-20250514": "Claude Haiku 4",
        }
        return names.get(self._model_id, self._model_id)
    
    @property
    def model_id(self) -> str:
        """The exact model identifier used in API calls."""
        return self._model_id
    
    def query(self, prompt: str, temperature: float = 0.0) -> ModelResponse:
        """
        Send a prompt to Claude and get a response.
        
        Note: Anthropic's API is slightly different from OpenAI's.
        The main differences are:
        - max_tokens is required (not optional)
        - system prompt is a separate parameter
        - Response structure is different
        
        Args:
            prompt: The text to send to the model
            temperature: Randomness control (0.0 = deterministic)
        
        Returns:
            ModelResponse with the model's output and metadata
        """
        start_time = time.time()
        
        try:
            # Make the API call
            # Note: Anthropic requires max_tokens, unlike OpenAI where it's optional
            response = self._client.messages.create(
                model=self._model_id,
                max_tokens=self._max_tokens,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                # Note: Anthropic doesn't support temperature=0 exactly
                # Using a very small value instead for near-deterministic output
                temperature=max(temperature, 0.0),
            )
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            # Anthropic returns a list of content blocks; we take the first text block
            response_text = response.content[0].text
            
            # Anthropic splits this differently than OpenAI
            tokens_input = response.usage.input_tokens
            tokens_output = response.usage.output_tokens
            tokens_total = tokens_input + tokens_output
            
            return ModelResponse(
                model_name=self._model_id,
                prompt=prompt,
                response=response_text,
                latency_ms=latency_ms,
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                tokens_total=tokens_total,
                success=True,
                error=None,
            )
            
        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            
            return ModelResponse(
                model_name=self._model_id,
                prompt=prompt,
                response="",
                latency_ms=latency_ms,
                tokens_input=0,
                tokens_output=0,
                tokens_total=0,
                success=False,
                error=str(e),
            )


def test_anthropic():
    """Quick test to verify Anthropic integration works."""
    print("Testing Anthropic integration...")
    
    try:
        model = AnthropicModel()
        response = model.query("What is 2 + 2? Reply with just the number.")
        
        print(f" Success!")
        print(f"   Model: {model.name}")
        print(f"   Response: {response.response}")
        print(f"   Latency: {response.latency_ms}ms")
        print(f"   Tokens: {response.tokens_total}")
        
    except Exception as e:
        print(f" Failed: {e}")


if __name__ == "__main__":
    test_anthropic()