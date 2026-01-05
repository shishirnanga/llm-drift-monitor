from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any


@dataclass
class ModelResponse:
    """
    Standardized response from any LLM.
    
    Attributes:
        model_name: Which model was called (e.g., "gpt-4-turbo-preview")
        prompt: The input we sent
        response: The text output from the model
        latency_ms: How long the API call took in milliseconds
        tokens_input: Number of tokens in the prompt
        tokens_output: Number of tokens in the response
        tokens_total: Total tokens used (input + output)
        timestamp: When this response was generated
        success: Whether the call succeeded
        error: Error message if it failed
        raw_response: The original API response (for debugging)
    """
    model_name: str
    prompt: str
    response: str
    latency_ms: int
    tokens_input: int
    tokens_output: int
    tokens_total: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    success: bool = True
    error: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_name": self.model_name,
            "prompt": self.prompt,
            "response": self.response,
            "latency_ms": self.latency_ms,
            "tokens_input": self.tokens_input,
            "tokens_output": self.tokens_output,
            "tokens_total": self.tokens_total,
            "timestamp": self.timestamp,
            "success": self.success,
            "error": self.error,
        }


class BaseModel(ABC):
    """
    Abstract base class for all LLM model wrappers.
    
    Any model we want to test must implement this interface.
    This ensures all models work the same way from the caller's perspective.
    
    The @abstractmethod decorator means subclasses MUST implement these methods.
    If they don't, Python will raise an error when you try to create an instance.
    
    Example usage:
        class GPT4Model(BaseModel):
            def query(self, prompt: str) -> ModelResponse:
                # ... implementation specific to GPT-4
                
        class ClaudeModel(BaseModel):
            def query(self, prompt: str) -> ModelResponse:
                # ... implementation specific to Claude
    
    Then in our test runner, we can do:
        for model in [GPT4Model(), ClaudeModel()]:
            response = model.query("What is 2+2?")
            # Works the same regardless of which model!
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Human-readable name for this model.
        Example: "GPT-4 Turbo", "Claude 3 Sonnet"
        """
        pass
    
    @property
    @abstractmethod
    def model_id(self) -> str:
        """
        The exact model identifier used in API calls.
        Example: "gpt-4-turbo-preview", "claude-sonnet-4-20250514"
        """
        pass
    
    @abstractmethod
    def query(self, prompt: str, temperature: float = 0.0) -> ModelResponse:
        """
        Send a prompt to the model and get a response.
        
        Args:
            prompt: The text to send to the model
            temperature: Controls randomness (0.0 = deterministic, 1.0 = creative)
                        We default to 0.0 for benchmarking consistency.
        
        Returns:
            ModelResponse with all the data we need for analysis
        """
        pass
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"{self.__class__.__name__}(name='{self.name}', model_id='{self.model_id}')"