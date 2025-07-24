"""Gemini API client with retry logic and error handling."""

import time
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import google.generativeai as genai
from google.generativeai.types import GenerateContentResponse
from google.api_core import exceptions as google_exceptions

from config.settings import settings


class GeminiError(Exception):
    """Base exception for Gemini API errors."""
    pass


class RetryableError(GeminiError):
    """Error that can be retried."""
    pass


class NonRetryableError(GeminiError):
    """Error that should not be retried."""
    pass


class MockMode(Enum):
    """Mock modes for testing."""
    DISABLED = "disabled"
    SUCCESS = "success"
    FAILURE = "failure"
    RATE_LIMIT = "rate_limit"


@dataclass
class GeminiResponse:
    """Response from Gemini API."""
    content: str
    confidence: float
    model_used: str
    token_usage: Dict[str, int]
    response_time: float
    is_mocked: bool = False


@dataclass
class RetryConfig:
    """Configuration for retry logic."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True


class GeminiClient:
    """Gemini API client with retry logic and error handling."""
    
    def __init__(self, api_key: Optional[str] = None, mock_mode: MockMode = MockMode.DISABLED):
        """Initialize Gemini client.
        
        Args:
            api_key: Gemini API key. If None, will use settings.gemini_api_key
            mock_mode: Mock mode for testing
        """
        self.logger = logging.getLogger(__name__)
        self.mock_mode = mock_mode
        self.retry_config = RetryConfig()
        
        # Configure API key
        self.api_key = api_key or settings.gemini_api_key
        if not self.api_key and mock_mode == MockMode.DISABLED:
            raise ValueError("Gemini API key is required. Set GEMINI_API_KEY environment variable.")
        
        # Initialize Gemini API
        if self.api_key and mock_mode == MockMode.DISABLED:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.5-flash')
        else:
            self.model = None
            
        self.logger.info(f"Gemini client initialized with mock_mode={mock_mode.value}")
    
    def generate_content(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_output_tokens: int = 2048,
        retry_config: Optional[RetryConfig] = None
    ) -> GeminiResponse:
        """Generate content using Gemini API with retry logic.
        
        Args:
            prompt: Input prompt for generation
            temperature: Sampling temperature (0.0 to 1.0)
            max_output_tokens: Maximum tokens to generate
            retry_config: Custom retry configuration
            
        Returns:
            GeminiResponse with generated content and metadata
            
        Raises:
            NonRetryableError: For errors that shouldn't be retried
            RetryableError: For errors after all retries exhausted
        """
        if self.mock_mode != MockMode.DISABLED:
            return self._generate_mock_response(prompt)
        
        config = retry_config or self.retry_config
        last_exception = None
        
        for attempt in range(config.max_retries + 1):
            try:
                start_time = time.time()
                
                # Configure generation parameters
                generation_config = genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                )
                
                # Make API call
                response = self.model.generate_content(
                    prompt,
                    generation_config=generation_config
                )
                
                response_time = time.time() - start_time
                
                # Process response
                return self._process_response(response, response_time)
                
            except google_exceptions.ResourceExhausted as e:
                # Rate limiting - retryable
                last_exception = RetryableError(f"Rate limit exceeded: {e}")
                self.logger.warning(f"Rate limit hit on attempt {attempt + 1}: {e}")
                
            except google_exceptions.DeadlineExceeded as e:
                # Timeout - retryable
                last_exception = RetryableError(f"Request timeout: {e}")
                self.logger.warning(f"Timeout on attempt {attempt + 1}: {e}")
                
            except google_exceptions.ServiceUnavailable as e:
                # Service unavailable - retryable
                last_exception = RetryableError(f"Service unavailable: {e}")
                self.logger.warning(f"Service unavailable on attempt {attempt + 1}: {e}")
                
            except google_exceptions.InvalidArgument as e:
                # Invalid request - not retryable
                raise NonRetryableError(f"Invalid request: {e}")
                
            except google_exceptions.PermissionDenied as e:
                # Permission denied - not retryable
                raise NonRetryableError(f"Permission denied: {e}")
                
            except Exception as e:
                # Unknown error - treat as retryable for now
                last_exception = RetryableError(f"Unknown error: {e}")
                self.logger.error(f"Unknown error on attempt {attempt + 1}: {e}")
            
            # Calculate delay for next retry
            if attempt < config.max_retries:
                delay = self._calculate_delay(attempt, config)
                self.logger.info(f"Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
        
        # All retries exhausted
        raise last_exception or RetryableError("All retries exhausted")
    
    def _process_response(self, response: GenerateContentResponse, response_time: float) -> GeminiResponse:
        """Process Gemini API response."""
        if not response.candidates:
            raise NonRetryableError("No candidates in response")
        
        candidate = response.candidates[0]
        
        # Check for safety issues
        if candidate.finish_reason and candidate.finish_reason.name != 'STOP':
            raise NonRetryableError(f"Generation stopped due to: {candidate.finish_reason.name}")
        
        content = candidate.content.parts[0].text if candidate.content.parts else ""
        
        # Calculate confidence based on safety ratings and finish reason
        confidence = self._calculate_confidence(candidate)
        
        # Extract token usage if available
        token_usage = {}
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            token_usage = {
                'prompt_tokens': response.usage_metadata.prompt_token_count,
                'completion_tokens': response.usage_metadata.candidates_token_count,
                'total_tokens': response.usage_metadata.total_token_count
            }
        
        return GeminiResponse(
            content=content,
            confidence=confidence,
            model_used="gemini-2.5-flash",
            token_usage=token_usage,
            response_time=response_time,
            is_mocked=False
        )
    
    def _calculate_confidence(self, candidate) -> float:
        """Calculate confidence score based on safety ratings and other factors."""
        base_confidence = 0.9  # Base confidence for successful generation
        
        # Reduce confidence based on safety ratings
        if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
            for rating in candidate.safety_ratings:
                if rating.probability.name in ['HIGH', 'MEDIUM']:
                    base_confidence -= 0.2
                elif rating.probability.name == 'LOW':
                    base_confidence -= 0.05
        
        # Ensure confidence is between 0 and 1
        return max(0.0, min(1.0, base_confidence))
    
    def _calculate_delay(self, attempt: int, config: RetryConfig) -> float:
        """Calculate delay for exponential backoff with jitter."""
        delay = config.base_delay * (config.exponential_base ** attempt)
        delay = min(delay, config.max_delay)
        
        if config.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)  # Add 0-50% jitter
        
        return delay
    
    def _generate_mock_response(self, prompt: str) -> GeminiResponse:
        """Generate mock response for testing."""
        import random
        
        if self.mock_mode == MockMode.FAILURE:
            raise NonRetryableError("Mock failure")
        elif self.mock_mode == MockMode.RATE_LIMIT:
            raise RetryableError("Mock rate limit")
        
        # Generate mock successful response
        mock_responses = [
            "This is a mock response for testing purposes.",
            "Mock AI assistant response with helpful information.",
            "Simulated Gemini response for offline testing."
        ]
        
        return GeminiResponse(
            content=random.choice(mock_responses),
            confidence=random.uniform(0.7, 0.95),
            model_used="mock-gemini",
            token_usage={
                'prompt_tokens': len(prompt.split()),
                'completion_tokens': 20,
                'total_tokens': len(prompt.split()) + 20
            },
            response_time=random.uniform(0.5, 2.0),
            is_mocked=True
        )
    
    def set_mock_mode(self, mode: MockMode) -> None:
        """Set mock mode for testing."""
        self.mock_mode = mode
        self.logger.info(f"Mock mode set to: {mode.value}")
    
    def health_check(self) -> bool:
        """Perform a health check on the Gemini API."""
        try:
            response = self.generate_content("Hello", max_output_tokens=10)
            return response.confidence > 0.5
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False