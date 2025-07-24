"""Tests for Gemini API client."""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from google.api_core import exceptions as google_exceptions

from src.gemini_client import (
    GeminiClient, GeminiResponse, RetryConfig, MockMode,
    GeminiError, RetryableError, NonRetryableError
)


class TestGeminiClient:
    """Test cases for GeminiClient."""
    
    def test_init_with_api_key(self):
        """Test client initialization with API key."""
        client = GeminiClient(api_key="test_key")
        assert client.api_key == "test_key"
        assert client.mock_mode == MockMode.DISABLED
    
    def test_init_without_api_key_mock_mode(self):
        """Test client initialization without API key in mock mode."""
        client = GeminiClient(mock_mode=MockMode.SUCCESS)
        assert client.mock_mode == MockMode.SUCCESS
        assert client.model is None
    
    @patch.dict('os.environ', {}, clear=True)
    def test_init_without_api_key_raises_error(self):
        """Test that missing API key raises error in non-mock mode."""
        with pytest.raises(ValueError, match="Gemini API key is required"):
            GeminiClient()
    
    def test_mock_success_response(self):
        """Test mock successful response generation."""
        client = GeminiClient(mock_mode=MockMode.SUCCESS)
        response = client.generate_content("Test prompt")
        
        assert isinstance(response, GeminiResponse)
        assert response.is_mocked is True
        assert response.confidence > 0.0
        assert len(response.content) > 0
        assert response.model_used == "mock-gemini"
        assert response.token_usage['total_tokens'] > 0
    
    def test_mock_failure_response(self):
        """Test mock failure response."""
        client = GeminiClient(mock_mode=MockMode.FAILURE)
        
        with pytest.raises(NonRetryableError, match="Mock failure"):
            client.generate_content("Test prompt")
    
    def test_mock_rate_limit_response(self):
        """Test mock rate limit response."""
        client = GeminiClient(mock_mode=MockMode.RATE_LIMIT)
        
        with pytest.raises(RetryableError, match="Mock rate limit"):
            client.generate_content("Test prompt")
    
    @patch('src.gemini_client.genai')
    def test_successful_api_call(self, mock_genai):
        """Test successful API call."""
        # Setup mock response
        mock_candidate = Mock()
        mock_candidate.content.parts = [Mock(text="Generated response")]
        mock_candidate.finish_reason.name = 'STOP'
        mock_candidate.safety_ratings = []
        
        mock_response = Mock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata.prompt_token_count = 10
        mock_response.usage_metadata.candidates_token_count = 5
        mock_response.usage_metadata.total_token_count = 15
        
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        client = GeminiClient(api_key="test_key")
        response = client.generate_content("Test prompt")
        
        assert response.content == "Generated response"
        assert response.confidence > 0.0
        assert response.is_mocked is False
        assert response.token_usage['total_tokens'] == 15
    
    @patch('src.gemini_client.genai')
    def test_rate_limit_retry(self, mock_genai):
        """Test retry logic for rate limiting."""
        mock_model = Mock()
        
        # First call raises rate limit, second succeeds
        mock_candidate = Mock()
        mock_candidate.content.parts = [Mock(text="Success after retry")]
        mock_candidate.finish_reason.name = 'STOP'
        mock_candidate.safety_ratings = []
        
        mock_success_response = Mock()
        mock_success_response.candidates = [mock_candidate]
        mock_success_response.usage_metadata.total_token_count = 10
        
        mock_model.generate_content.side_effect = [
            google_exceptions.ResourceExhausted("Rate limit"),
            mock_success_response
        ]
        mock_genai.GenerativeModel.return_value = mock_model
        
        client = GeminiClient(api_key="test_key")
        client.retry_config.base_delay = 0.1  # Speed up test
        
        response = client.generate_content("Test prompt")
        assert response.content == "Success after retry"
        assert mock_model.generate_content.call_count == 2
    
    @patch('src.gemini_client.genai')
    def test_non_retryable_error(self, mock_genai):
        """Test non-retryable error handling."""
        mock_model = Mock()
        mock_model.generate_content.side_effect = google_exceptions.InvalidArgument("Bad request")
        mock_genai.GenerativeModel.return_value = mock_model
        
        client = GeminiClient(api_key="test_key")
        
        with pytest.raises(NonRetryableError, match="Invalid request"):
            client.generate_content("Test prompt")
        
        # Should not retry
        assert mock_model.generate_content.call_count == 1
    
    @patch('src.gemini_client.genai')
    def test_retry_exhaustion(self, mock_genai):
        """Test behavior when all retries are exhausted."""
        mock_model = Mock()
        mock_model.generate_content.side_effect = google_exceptions.ServiceUnavailable("Service down")
        mock_genai.GenerativeModel.return_value = mock_model
        
        client = GeminiClient(api_key="test_key")
        client.retry_config.max_retries = 2
        client.retry_config.base_delay = 0.01  # Speed up test
        
        with pytest.raises(RetryableError, match="Service unavailable"):
            client.generate_content("Test prompt")
        
        # Should try initial + 2 retries = 3 total
        assert mock_model.generate_content.call_count == 3
    
    def test_retry_delay_calculation(self):
        """Test exponential backoff delay calculation."""
        client = GeminiClient(mock_mode=MockMode.SUCCESS)
        config = RetryConfig(base_delay=1.0, exponential_base=2.0, max_delay=10.0, jitter=False)
        
        # Test exponential backoff
        assert client._calculate_delay(0, config) == 1.0
        assert client._calculate_delay(1, config) == 2.0
        assert client._calculate_delay(2, config) == 4.0
        
        # Test max delay cap
        config.max_delay = 3.0
        assert client._calculate_delay(2, config) == 3.0
    
    def test_confidence_calculation(self):
        """Test confidence score calculation."""
        client = GeminiClient(mock_mode=MockMode.SUCCESS)
        
        # Mock candidate with no safety issues
        mock_candidate = Mock()
        mock_candidate.safety_ratings = []
        confidence = client._calculate_confidence(mock_candidate)
        assert confidence == 0.9
        
        # Mock candidate with high safety risk
        mock_rating = Mock()
        mock_rating.probability.name = 'HIGH'
        mock_candidate.safety_ratings = [mock_rating]
        confidence = client._calculate_confidence(mock_candidate)
        assert confidence == 0.7  # 0.9 - 0.2
    
    @patch('src.gemini_client.genai')
    def test_health_check_success(self, mock_genai):
        """Test successful health check."""
        mock_candidate = Mock()
        mock_candidate.content.parts = [Mock(text="OK")]
        mock_candidate.finish_reason.name = 'STOP'
        mock_candidate.safety_ratings = []
        
        mock_response = Mock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata.total_token_count = 5
        
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        client = GeminiClient(api_key="test_key")
        assert client.health_check() is True
    
    @patch('src.gemini_client.genai')
    def test_health_check_failure(self, mock_genai):
        """Test failed health check."""
        mock_model = Mock()
        mock_model.generate_content.side_effect = Exception("API Error")
        mock_genai.GenerativeModel.return_value = mock_model
        
        client = GeminiClient(api_key="test_key")
        assert client.health_check() is False
    
    def test_set_mock_mode(self):
        """Test setting mock mode."""
        client = GeminiClient(mock_mode=MockMode.SUCCESS)
        assert client.mock_mode == MockMode.SUCCESS
        
        client.set_mock_mode(MockMode.FAILURE)
        assert client.mock_mode == MockMode.FAILURE


class TestRetryConfig:
    """Test cases for RetryConfig."""
    
    def test_default_values(self):
        """Test default retry configuration values."""
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True
    
    def test_custom_values(self):
        """Test custom retry configuration values."""
        config = RetryConfig(
            max_retries=5,
            base_delay=0.5,
            max_delay=30.0,
            exponential_base=1.5,
            jitter=False
        )
        assert config.max_retries == 5
        assert config.base_delay == 0.5
        assert config.max_delay == 30.0
        assert config.exponential_base == 1.5
        assert config.jitter is False


class TestGeminiResponse:
    """Test cases for GeminiResponse."""
    
    def test_response_creation(self):
        """Test GeminiResponse creation."""
        response = GeminiResponse(
            content="Test content",
            confidence=0.85,
            model_used="gemini-2.5-flash",
            token_usage={'total_tokens': 100},
            response_time=1.5,
            is_mocked=False
        )
        
        assert response.content == "Test content"
        assert response.confidence == 0.85
        assert response.model_used == "gemini-2.5-flash"
        assert response.token_usage['total_tokens'] == 100
        assert response.response_time == 1.5
        assert response.is_mocked is False