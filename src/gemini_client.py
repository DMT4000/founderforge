"""Gemini API client with retry logic and error handling."""

import time
import logging
from .logging_manager import get_logging_manager, LogLevel, LogCategory
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
        self.logger = get_logging_manager().get_logger(__name__.split(".")[-1], LogCategory.SYSTEM)
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
        max_output_tokens: int = 20000,
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
        
        # Map finish reason numbers to names for better debugging
        finish_reason_map = {
            0: 'FINISH_REASON_UNSPECIFIED',
            1: 'STOP',
            2: 'MAX_TOKENS',
            3: 'SAFETY',
            4: 'RECITATION',
            5: 'OTHER'
        }
        
        finish_reason_name = finish_reason_map.get(candidate.finish_reason, f"UNKNOWN_{candidate.finish_reason}")
        
        # Check for safety or other blocking issues
        if candidate.finish_reason in [3, 4, 5]:  # SAFETY, RECITATION, OTHER
            raise NonRetryableError(f"Generation blocked due to: {finish_reason_name}")
        
        # Handle MAX_TOKENS as a warning, not an error
        if candidate.finish_reason == 2:  # MAX_TOKENS
            self.logger.warning("Response truncated due to max tokens limit")
        
        # Extract content - handle case where no parts are returned
        content = ""
        if candidate.content and candidate.content.parts:
            content = candidate.content.parts[0].text
        else:
            # If no content but tokens were used, there might be a safety issue
            if candidate.finish_reason == 3:  # SAFETY
                content = "Response blocked due to safety filters."
            elif candidate.finish_reason == 2:  # MAX_TOKENS with no content
                content = "Response was truncated and no content was generated."
            else:
                content = f"No content generated (finish reason: {finish_reason_name})"
        
        # Calculate confidence based on safety ratings and finish reason
        confidence = self._calculate_confidence(candidate)
        
        # Reduce confidence if no actual content was generated
        if not content or "blocked" in content.lower() or "truncated" in content.lower():
            confidence *= 0.5
        
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
        import json
        
        if self.mock_mode == MockMode.FAILURE:
            raise NonRetryableError("Mock failure")
        elif self.mock_mode == MockMode.RATE_LIMIT:
            raise RetryableError("Mock rate limit")
        
        # Generate appropriate mock response based on prompt content
        content = self._get_mock_content_for_prompt(prompt)
        
        return GeminiResponse(
            content=content,
            confidence=random.uniform(0.7, 0.95),
            model_used="mock-gemini",
            token_usage={
                'prompt_tokens': len(prompt.split()),
                'completion_tokens': len(content.split()),
                'total_tokens': len(prompt.split()) + len(content.split())
            },
            response_time=random.uniform(0.1, 0.5),
            is_mocked=True
        )
    
    def _get_mock_content_for_prompt(self, prompt: str) -> str:
        """Generate appropriate mock content based on prompt type."""
        import json
        
        prompt_lower = prompt.lower()
        
        # Mock response for planning agent
        if "action plan" in prompt_lower or "planning" in prompt_lower or "daily_planning" in prompt_lower:
            mock_plan = {
                "executive_summary": "Mock daily action plan focusing on high-priority business tasks and strategic objectives",
                "action_items": [
                    {
                        "title": "Review product roadmap",
                        "description": "Analyze current product development priorities and adjust timeline based on customer feedback",
                        "priority": "high",
                        "timeline": "2 hours",
                        "resources_needed": ["product team", "customer data", "market research"]
                    },
                    {
                        "title": "Prepare investor presentation",
                        "description": "Update pitch deck with latest metrics and prepare talking points for upcoming investor meetings",
                        "priority": "high",
                        "timeline": "3 hours",
                        "resources_needed": ["financial data", "presentation template", "design team"]
                    },
                    {
                        "title": "Team standup meeting",
                        "description": "Daily sync with engineering and product teams to discuss progress and blockers",
                        "priority": "medium",
                        "timeline": "30 minutes",
                        "resources_needed": ["team members", "meeting room", "agenda"]
                    },
                    {
                        "title": "Customer feedback analysis",
                        "description": "Review recent customer feedback and identify patterns for product improvements",
                        "priority": "medium",
                        "timeline": "1 hour",
                        "resources_needed": ["customer support data", "feedback analysis tools"]
                    }
                ],
                "success_metrics": [
                    "Complete all high-priority tasks by end of day",
                    "Gather actionable insights from customer feedback",
                    "Advance investor presentation to review-ready state"
                ],
                "risks": [
                    {
                        "risk": "Investor meeting preparation may take longer than expected",
                        "mitigation": "Focus on key metrics and defer detailed slides to follow-up"
                    },
                    {
                        "risk": "Team meeting may reveal unexpected blockers",
                        "mitigation": "Prepare backup plans and allocate buffer time for problem-solving"
                    }
                ],
                "next_steps": [
                    "Schedule follow-up meetings based on today's outcomes",
                    "Prepare tomorrow's priorities based on progress made",
                    "Document key decisions and learnings from customer feedback"
                ]
            }
            return json.dumps(mock_plan, indent=2)
        
        # Mock response for coaching agent
        elif "coaching" in prompt_lower or "motivational" in prompt_lower or "coach" in prompt_lower:
            mock_coaching = {
                "message": "Today is a great opportunity to make significant progress on your most important goals. Focus on your high-impact activities first, and remember that consistent progress is more valuable than perfection. You've got the skills and determination to succeed - trust in your abilities and stay focused on what matters most.",
                "insights": [
                    "Prioritize deep work during your highest energy periods",
                    "Break large tasks into smaller, manageable chunks",
                    "Celebrate small wins to maintain momentum throughout the day"
                ],
                "motivation_level": "high",
                "confidence": 0.9
            }
            return json.dumps(mock_coaching, indent=2)
        
        # Mock response for validation tasks
        elif "validate" in prompt_lower or "validation" in prompt_lower:
            mock_validation = {
                "is_valid": True,
                "confidence": 0.85,
                "validation_results": {
                    "completeness": "passed",
                    "format": "passed",
                    "content_quality": "passed"
                },
                "errors": [],
                "warnings": [
                    "Consider adding more specific timelines for some action items"
                ],
                "recommendations": [
                    "Include success metrics for each major task",
                    "Add contingency plans for high-risk activities"
                ]
            }
            return json.dumps(mock_validation, indent=2)
        
        # Mock response for task analysis
        elif "analyze" in prompt_lower or "analysis" in prompt_lower:
            mock_analysis = {
                "complexity": "medium",
                "estimated_time": "4-6 hours",
                "required_agents": ["planner", "tool_caller", "coach"],
                "confidence": 0.8,
                "key_factors": [
                    "Multiple high-priority tasks requiring coordination",
                    "Mix of strategic and operational activities",
                    "Need for stakeholder communication"
                ]
            }
            return json.dumps(mock_analysis, indent=2)
        
        # Default fallback response
        else:
            mock_response = {
                "response": "I understand your request and am ready to help. Please provide more specific details about what you'd like me to assist with.",
                "confidence": 0.7,
                "suggestions": [
                    "Clarify your main objective",
                    "Provide relevant context",
                    "Specify your preferences"
                ]
            }
            return json.dumps(mock_response, indent=2)
    
    def set_mock_mode(self, mode: MockMode) -> None:
        """Set mock mode for testing."""
        self.mock_mode = mode
        self.logger.info(f"Mock mode set to: {mode.value}")
    
    def health_check(self) -> bool:
        """Perform a health check on the Gemini API."""
        try:
            response = self.generate_content("Hello, please respond briefly.", max_output_tokens=20000)
            return response.confidence > 0.5 and len(response.content.strip()) > 0
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if the Gemini API is available."""
        if self.mock_mode != MockMode.DISABLED:
            return True
        return self.api_key is not None and self.model is not None