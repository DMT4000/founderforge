"""Tests for confidence manager and content filtering."""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch

from src.confidence_manager import (
    ConfidenceManager, ContentFilter, ConfidenceScore, FallbackResponse,
    FallbackType, FilterResult, FilterType
)


class TestContentFilter:
    """Test cases for ContentFilter."""
    
    def test_safe_content(self):
        """Test filtering of safe content."""
        filter = ContentFilter()
        result = filter.filter_content("This is a safe message about business planning.")
        
        assert result.is_safe is True
        assert result.filter_type == FilterType.SAFE
        assert len(result.detected_patterns) == 0
        assert result.confidence > 0.8
    
    def test_pii_detection_email(self):
        """Test PII detection for email addresses."""
        filter = ContentFilter()
        result = filter.filter_content("Contact me at john.doe@example.com for more info.")
        
        assert result.is_safe is False
        assert result.filter_type == FilterType.PII_DETECTED
        assert "pii_email" in result.detected_patterns
        assert result.confidence < 0.5
    
    def test_pii_detection_phone(self):
        """Test PII detection for phone numbers."""
        filter = ContentFilter()
        result = filter.filter_content("Call me at (555) 123-4567 tomorrow.")
        
        assert result.is_safe is False
        assert result.filter_type == FilterType.PII_DETECTED
        assert "pii_phone" in result.detected_patterns
    
    def test_pii_detection_ssn(self):
        """Test PII detection for SSN."""
        filter = ContentFilter()
        result = filter.filter_content("My SSN is 123-45-6789.")
        
        assert result.is_safe is False
        assert result.filter_type == FilterType.PII_DETECTED
        assert "pii_ssn" in result.detected_patterns
    
    def test_toxicity_detection_profanity(self):
        """Test toxicity detection for profanity."""
        filter = ContentFilter()
        result = filter.filter_content("That's a stupid idea and you're an idiot.")
        
        assert result.is_safe is False
        assert result.filter_type == FilterType.TOXIC
        assert any("toxicity_profanity" in pattern for pattern in result.detected_patterns)
    
    def test_toxicity_detection_threats(self):
        """Test toxicity detection for threats."""
        filter = ContentFilter()
        result = filter.filter_content("I will destroy you and your business.")
        
        assert result.is_safe is False
        assert result.filter_type == FilterType.TOXIC
        assert any("toxicity_threats" in pattern for pattern in result.detected_patterns)
    
    def test_multiple_violations(self):
        """Test content with multiple violations."""
        filter = ContentFilter()
        result = filter.filter_content("Email me at bad@example.com, you stupid person!")
        
        assert result.is_safe is False
        assert len(result.detected_patterns) >= 2
        assert any("pii_" in pattern for pattern in result.detected_patterns)
        assert any("toxicity_" in pattern for pattern in result.detected_patterns)


class TestConfidenceManager:
    """Test cases for ConfidenceManager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = ConfidenceManager(confidence_threshold=0.8)
    
    def test_init_default_threshold(self):
        """Test initialization with default threshold."""
        with patch('src.confidence_manager.settings.get_feature_flag', return_value=0.75):
            manager = ConfidenceManager()
            assert manager.confidence_threshold == 0.75
    
    def test_init_custom_threshold(self):
        """Test initialization with custom threshold."""
        manager = ConfidenceManager(confidence_threshold=0.9)
        assert manager.confidence_threshold == 0.9
    
    def test_high_confidence_calculation(self):
        """Test confidence calculation for high-confidence response."""
        result = self.manager.calculate_confidence(
            ai_confidence=0.95,
            content="This is a helpful business advice response.",
            context_quality=0.9,
            response_length=150
        )
        
        assert isinstance(result, ConfidenceScore)
        assert result.score > 0.8
        assert result.meets_threshold is True
        assert 'ai_model' in result.factors
        assert 'content_safety' in result.factors
    
    def test_low_confidence_calculation(self):
        """Test confidence calculation for low-confidence response."""
        result = self.manager.calculate_confidence(
            ai_confidence=0.3,
            content="I'm not sure about this.",
            context_quality=0.4,
            response_length=20
        )
        
        assert result.score < 0.8
        assert result.meets_threshold is False
        assert result.explanation is not None
    
    def test_unsafe_content_confidence(self):
        """Test confidence calculation with unsafe content."""
        result = self.manager.calculate_confidence(
            ai_confidence=0.9,
            content="Contact me at test@example.com, you idiot!",
            context_quality=0.9,
            response_length=100
        )
        
        assert result.score < 0.8  # Should be low due to content safety
        assert result.meets_threshold is False
    
    def test_fallback_response_checklist(self):
        """Test fallback response generation for checklist type."""
        confidence_score = ConfidenceScore(
            score=0.5,
            factors={'ai_model': 0.5},
            meets_threshold=False,
            explanation="Low confidence"
        )
        
        fallback = self.manager.get_fallback_response(
            "How to start a business?",
            confidence_score
        )
        
        assert isinstance(fallback, FallbackResponse)
        assert fallback.fallback_type == FallbackType.CHECKLIST
        assert fallback.is_fallback is True
        assert "checklist" in fallback.content.lower()
    
    def test_fallback_response_interactive_script(self):
        """Test fallback response generation for interactive script type."""
        confidence_score = ConfidenceScore(
            score=0.5,
            factors={'ai_model': 0.5},
            meets_threshold=False,
            explanation="Low confidence"
        )
        
        fallback = self.manager.get_fallback_response(
            "What should be my business strategy?",
            confidence_score
        )
        
        assert fallback.fallback_type == FallbackType.INTERACTIVE_SCRIPT
        assert "step by step" in fallback.content.lower()
    
    def test_fallback_response_template(self):
        """Test fallback response generation for template type."""
        confidence_score = ConfidenceScore(
            score=0.5,
            factors={'ai_model': 0.5},
            meets_threshold=False,
            explanation="Low confidence"
        )
        
        fallback = self.manager.get_fallback_response(
            "What is venture capital?",
            confidence_score
        )
        
        assert fallback.fallback_type == FallbackType.TEMPLATE
        assert "template" in fallback.content.lower()
    
    def test_fallback_response_error_message(self):
        """Test fallback response generation for error message type."""
        confidence_score = ConfidenceScore(
            score=0.2,
            factors={'content_safety': 0.1},  # Very low safety score
            meets_threshold=False,
            explanation="Content safety issue"
        )
        
        fallback = self.manager.get_fallback_response(
            "Some unsafe query",
            confidence_score
        )
        
        assert fallback.fallback_type == FallbackType.ERROR_MESSAGE
        assert "safety concerns" in fallback.content.lower()
    
    def test_update_confidence_threshold(self):
        """Test updating confidence threshold."""
        with patch('src.confidence_manager.settings.update_feature_flag') as mock_update:
            self.manager.update_confidence_threshold(0.75)
            assert self.manager.confidence_threshold == 0.75
            mock_update.assert_called_once_with('confidence_threshold', 0.75)
    
    def test_update_confidence_threshold_invalid(self):
        """Test updating confidence threshold with invalid value."""
        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
            self.manager.update_confidence_threshold(1.5)
        
        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
            self.manager.update_confidence_threshold(-0.1)
    
    def test_confidence_factors_weighting(self):
        """Test that confidence factors are properly weighted."""
        # Test with perfect AI confidence but poor content safety
        result = self.manager.calculate_confidence(
            ai_confidence=1.0,
            content="Email me at bad@example.com",  # Contains PII
            context_quality=1.0,
            response_length=100
        )
        
        # Should be significantly reduced due to content safety
        assert result.score < 0.8
        assert result.factors['ai_model'] == 1.0
        assert result.factors['content_safety'] < 0.5
    
    def test_response_length_factor(self):
        """Test response length factor calculation."""
        # Very short response
        result_short = self.manager.calculate_confidence(
            ai_confidence=0.9,
            content="Yes.",
            context_quality=0.9,
            response_length=4
        )
        
        # Normal length response
        result_normal = self.manager.calculate_confidence(
            ai_confidence=0.9,
            content="This is a normal length response with helpful information.",
            context_quality=0.9,
            response_length=100
        )
        
        assert result_short.factors['response_length'] < result_normal.factors['response_length']
    
    @patch('pathlib.Path.mkdir')
    @patch('pathlib.Path.exists', return_value=False)
    @patch('builtins.open')
    def test_initialize_fallback_resources(self, mock_open, mock_exists, mock_mkdir):
        """Test initialization of fallback resources."""
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        manager = ConfidenceManager()
        
        # Should create directory and files
        mock_mkdir.assert_called()
        assert mock_open.call_count >= 2  # At least checklists.json and scripts.json


class TestConfidenceScore:
    """Test cases for ConfidenceScore dataclass."""
    
    def test_confidence_score_creation(self):
        """Test ConfidenceScore creation."""
        score = ConfidenceScore(
            score=0.85,
            factors={'ai_model': 0.9, 'content_safety': 0.8},
            meets_threshold=True,
            explanation="High confidence response"
        )
        
        assert score.score == 0.85
        assert score.factors['ai_model'] == 0.9
        assert score.meets_threshold is True
        assert score.explanation == "High confidence response"


class TestFallbackResponse:
    """Test cases for FallbackResponse dataclass."""
    
    def test_fallback_response_creation(self):
        """Test FallbackResponse creation."""
        response = FallbackResponse(
            content="Fallback content",
            fallback_type=FallbackType.CHECKLIST,
            confidence=0.6,
            source="fallback_checklist",
            is_fallback=True
        )
        
        assert response.content == "Fallback content"
        assert response.fallback_type == FallbackType.CHECKLIST
        assert response.confidence == 0.6
        assert response.is_fallback is True