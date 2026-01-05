"""
src/models/openai_model.py

OpenAI GPT-4 model wrapper.

This file implements the BaseModel interface for OpenAI's API.
All the OpenAI-specific code is isolated here.

OPENAI API BASICS:
- You send a list of "messages" (conversation history)
- Each message has a "role" (system, user, assistant) and "content"
- The API returns a "completion" with the model's response
- You pay per token (roughly 4 characters = 1 token)

AUTHENTICATION:
- Requires an API key from https://platform.openai.com/api-keys
- Key should be in .env file as OPENAI_API_KEY
- Never hardcode API keys in your code!
"""

import os
import time
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

from .base import BaseModel, ModelResponse

# Load environment variables from .env file
load_dotenv()


class OpenAIModel(BaseModel):
    """
    Wrapper for OpenAI's GPT models.
    
    Example usage:
        model = OpenAIModel()  # Uses default GPT-4 Turbo
        response = model.query("What is the capital of France?")
        print(response.response)  # "The capital of France is Paris."
        print(response.latency_ms)  # 523
        
        # Or specify a different model
        model = OpenAIModel(model_id="gpt-4o")
    """
    
    def __init__(
        self, 
        model_id: str = "gpt-4-turbo-preview",
        api_key: Optional[str] = None,
        max_tokens: int = 1000
    ):
        """
        Initialize the OpenAI model wrapper.
        
        Args:
            model_id: Which OpenAI model to use. Options include:
                - "gpt-4-turbo-preview" (recommended - fast and capable)
                - "gpt-4o" (newest, multimodal)
                - "gpt-4" (original, slower)
                - "gpt-3.5-turbo" (cheaper, less capable)
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var
            max_tokens: Maximum tokens in the response
        """
        self._model_id = model_id
        self._max_tokens = max_tokens
        
        # Get API key from parameter or environment
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self._api_key:
            raise ValueError(
                "OpenAI API key not found. Either:\n"
                "1. Pass api_key parameter\n"
                "2. Set OPENAI_API_KEY in your .env file\n"
                "3. Set OPENAI_API_KEY environment variable"
            )
        
        # Create the OpenAI client
        # This doesn't make any API calls yet - just sets up the connection
        self._client = OpenAI(api_key=self._api_key)
    
    @property
    def name(self) -> str:
        """Human-readable name."""
        # Map model IDs to friendly names
        names = {
            "gpt-4-turbo-preview": "GPT-4 Turbo",
            "gpt-4o": "GPT-4o",
            "gpt-4": "GPT-4",
            "gpt-3.5-turbo": "GPT-3.5 Turbo",
        }
        return names.get(self._model_id, self._model_id)
    
    @property
    def model_id(self) -> str:
        """The exact model identifier used in API calls."""
        return self._model_id
    
    def query(self, prompt: str, temperature: float = 0.0) -> ModelResponse:
        """
        Send a prompt to GPT and get a response.
        
        This method:
        1. Records the start time
        2. Sends the request to OpenAI's API
        3. Records the end time
        4. Extracts all the data we need
        5. Returns a standardized ModelResponse
        
        Args:
            prompt: The text to send to the model
            temperature: Randomness control (0.0 = deterministic)
        
        Returns:
            ModelResponse with the model's output and metadata
        """
        # Record start time for latency measurement
        start_time = time.time()
        
        try:
            # Make the API call
            # This is where the actual network request happens
            response = self._client.chat.completions.create(
                model=self._model_id,
                messages=[
                    # We just send a single user message
                    # For more complex use cases, you'd include conversation history
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=self._max_tokens,
            )
            
            # Calculate latency in milliseconds
            latency_ms = int((time.time() - start_time) * 1000)
            
            # Extract the response text
            # The API returns a complex object; we just want the text
            response_text = response.choices[0].message.content
            
            # Extract token usage
            # This is important for cost tracking and analysis
            tokens_input = response.usage.prompt_tokens
            tokens_output = response.usage.completion_tokens
            tokens_total = response.usage.total_tokens
            
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
            # If anything goes wrong, return an error response
            # This is better than crashing - we can still record that this test failed
            latency_ms = int((time.time() - start_time) * 1000)
            
            return ModelResponse(
                model_name=self._model_id,
                prompt=prompt,
                response="",  # Empty response on error
                latency_ms=latency_ms,
                tokens_input=0,
                tokens_output=0,
                tokens_total=0,
                success=False,
                error=str(e),
            )


# Convenience function for quick testing
def test_openai():
    """Quick test to verify OpenAI integration works."""
    print("Testing OpenAI integration...")
    
    try:
        model = OpenAIModel()
        response = model.query("What is 2 + 2? Reply with just the number.")
        
        print(f"✅ Success!")
        print(f"   Model: {model.name}")
        print(f"   Response: {response.response}")
        print(f"   Latency: {response.latency_ms}ms")
        print(f"   Tokens: {response.tokens_total}")
        
    except Exception as e:
        print(f"❌ Failed: {e}")


if __name__ == "__main__":
    # This runs only if you execute this file directly:
    # python src/models/openai_model.py
    test_openai()