"""Confidence scoring and fallback mechanisms for AI responses."""

import re
import logging
from .logging_manager import get_logging_manager, LogLevel, LogCategory
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from config.settings import settings


class FallbackType(Enum):
    """Types of fallback mechanisms."""
    INTERACTIVE_SCRIPT = "interactive_script"
    CHECKLIST = "checklist"
    TEMPLATE = "template"
    ERROR_MESSAGE = "error_message"


class FilterType(Enum):
    """Types of content filtering results."""
    SAFE = "safe"
    TOXIC = "toxic"
    PII_DETECTED = "pii_detected"
    BLOCKED = "blocked"


@dataclass
class ConfidenceScore:
    """Confidence scoring result."""
    score: float
    factors: Dict[str, float]
    meets_threshold: bool
    explanation: str


@dataclass
class FilterResult:
    """Content filtering result."""
    is_safe: bool
    filter_type: FilterType
    detected_patterns: List[str]
    confidence: float
    explanation: str


@dataclass
class FallbackResponse:
    """Fallback response when confidence is low."""
    content: str
    fallback_type: FallbackType
    confidence: float
    source: str
    is_fallback: bool = True


class ContentFilter:
    """Basic content filtering for toxicity and PII detection."""
    
    def __init__(self):
        self.logger = get_logging_manager().get_logger(__name__.split(".")[-1], LogCategory.SYSTEM)
        
        # PII patterns (basic regex-based detection)
        self.pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})',
            'ssn': r'\b\d{3}-?\d{2}-?\d{4}\b',
            'credit_card': r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
            'ip_address': r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
            'url': r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:\w*))?)?'
        }
        
        # Toxicity patterns (basic keyword-based detection)
        self.toxicity_patterns = {
            'profanity': r'\b(?:damn|hell|crap|stupid|idiot|moron)\b',
            'hate_speech': r'\b(?:hate|despise|loathe)\s+(?:you|them|people)\b',
            'threats': r'\b(?:kill|destroy|harm|hurt|attack)\s+(?:you|them|people)\b',
            'harassment': r'\b(?:shut up|go away|leave me alone|stop bothering)\b'
        }
    
    def filter_content(self, content: str) -> FilterResult:
        """Filter content for PII and toxicity.
        
        Args:
            content: Text content to filter
            
        Returns:
            FilterResult with safety assessment
        """
        detected_patterns = []
        filter_type = FilterType.SAFE
        confidence = 0.9  # Base confidence for safe content
        
        # Check for PII
        pii_matches = self._detect_pii(content)
        if pii_matches:
            detected_patterns.extend(pii_matches)
            filter_type = FilterType.PII_DETECTED
            confidence = 0.3
        
        # Check for toxicity
        toxicity_matches = self._detect_toxicity(content)
        if toxicity_matches:
            detected_patterns.extend(toxicity_matches)
            filter_type = FilterType.TOXIC
            confidence = 0.2
        
        # Determine if content is safe
        is_safe = filter_type == FilterType.SAFE
        
        explanation = self._generate_filter_explanation(filter_type, detected_patterns)
        
        return FilterResult(
            is_safe=is_safe,
            filter_type=filter_type,
            detected_patterns=detected_patterns,
            confidence=confidence,
            explanation=explanation
        )
    
    def _detect_pii(self, content: str) -> List[str]:
        """Detect PII patterns in content."""
        matches = []
        for pii_type, pattern in self.pii_patterns.items():
            if re.search(pattern, content, re.IGNORECASE):
                matches.append(f"pii_{pii_type}")
        return matches
    
    def _detect_toxicity(self, content: str) -> List[str]:
        """Detect toxicity patterns in content."""
        matches = []
        for toxicity_type, pattern in self.toxicity_patterns.items():
            if re.search(pattern, content, re.IGNORECASE):
                matches.append(f"toxicity_{toxicity_type}")
        return matches
    
    def _generate_filter_explanation(self, filter_type: FilterType, patterns: List[str]) -> str:
        """Generate explanation for filter result."""
        if filter_type == FilterType.SAFE:
            return "Content passed all safety filters"
        elif filter_type == FilterType.PII_DETECTED:
            return f"PII detected: {', '.join([p for p in patterns if p.startswith('pii_')])}"
        elif filter_type == FilterType.TOXIC:
            return f"Toxicity detected: {', '.join([p for p in patterns if p.startswith('toxicity_')])}"
        else:
            return f"Content blocked due to: {', '.join(patterns)}"


class ConfidenceManager:
    """Manages confidence scoring and fallback mechanisms."""
    
    def __init__(self, confidence_threshold: float = None):
        """Initialize confidence manager.
        
        Args:
            confidence_threshold: Minimum confidence threshold (default from settings)
        """
        self.logger = get_logging_manager().get_logger(__name__.split(".")[-1], LogCategory.SYSTEM)
        self.confidence_threshold = confidence_threshold or settings.get_feature_flag('confidence_threshold', 0.8)
        self.content_filter = ContentFilter()
        
        # Initialize fallback resources
        self.fallback_dir = Path("data/fallbacks")
        self.fallback_dir.mkdir(exist_ok=True)
        self._initialize_fallback_resources()
    
    def calculate_confidence(
        self,
        ai_confidence: float,
        content: str,
        context_quality: float = 1.0,
        response_length: int = None
    ) -> ConfidenceScore:
        """Calculate overall confidence score.
        
        Args:
            ai_confidence: Confidence from AI model
            content: Generated content
            context_quality: Quality of input context (0.0-1.0)
            response_length: Length of response in characters
            
        Returns:
            ConfidenceScore with detailed breakdown
        """
        factors = {}
        
        # Base AI confidence
        factors['ai_model'] = ai_confidence
        
        # Content safety filter
        filter_result = self.content_filter.filter_content(content)
        factors['content_safety'] = filter_result.confidence
        
        # Context quality factor
        factors['context_quality'] = context_quality
        
        # Response length factor (penalize very short or very long responses)
        if response_length is not None:
            if response_length < 10:
                factors['response_length'] = 0.3  # Too short
            elif response_length > 2000:
                factors['response_length'] = 0.7  # Too long
            else:
                factors['response_length'] = 1.0  # Good length
        else:
            factors['response_length'] = 1.0
        
        # Calculate weighted average
        weights = {
            'ai_model': 0.4,
            'content_safety': 0.3,
            'context_quality': 0.2,
            'response_length': 0.1
        }
        
        overall_score = sum(factors[key] * weights[key] for key in factors.keys())
        meets_threshold = overall_score >= self.confidence_threshold
        
        explanation = self._generate_confidence_explanation(factors, overall_score, meets_threshold)
        
        return ConfidenceScore(
            score=overall_score,
            factors=factors,
            meets_threshold=meets_threshold,
            explanation=explanation
        )
    
    def get_fallback_response(
        self,
        original_query: str,
        confidence_score: ConfidenceScore,
        context: Dict[str, Any] = None
    ) -> FallbackResponse:
        """Get appropriate fallback response for low confidence.
        
        Args:
            original_query: Original user query
            confidence_score: Calculated confidence score
            context: Additional context for fallback selection
            
        Returns:
            FallbackResponse with appropriate fallback content
        """
        # Determine best fallback type based on query and confidence
        fallback_type = self._select_fallback_type(original_query, confidence_score, context)
        
        # Generate fallback content
        content = self._generate_fallback_content(fallback_type, original_query, context)
        
        return FallbackResponse(
            content=content,
            fallback_type=fallback_type,
            confidence=0.6,  # Fallback responses have moderate confidence
            source=f"fallback_{fallback_type.value}"
        )
    
    def _select_fallback_type(
        self,
        query: str,
        confidence_score: ConfidenceScore,
        context: Dict[str, Any] = None
    ) -> FallbackType:
        """Select appropriate fallback type based on query and confidence."""
        query_lower = query.lower()
        
        # If content safety is the issue, use error message
        if confidence_score.factors.get('content_safety', 1.0) < 0.5:
            return FallbackType.ERROR_MESSAGE
        
        # For how-to or process questions, use checklists
        if any(word in query_lower for word in ['how to', 'steps', 'process', 'guide']):
            return FallbackType.CHECKLIST
        
        # For planning or strategy questions, use interactive scripts
        if any(word in query_lower for word in ['plan', 'strategy', 'approach', 'what should']):
            return FallbackType.INTERACTIVE_SCRIPT
        
        # For specific information requests, use templates
        if any(word in query_lower for word in ['what is', 'define', 'explain', 'tell me about']):
            return FallbackType.TEMPLATE
        
        # Default to interactive script
        return FallbackType.INTERACTIVE_SCRIPT
    
    def _generate_fallback_content(
        self,
        fallback_type: FallbackType,
        query: str,
        context: Dict[str, Any] = None
    ) -> str:
        """Generate content for specific fallback type."""
        if fallback_type == FallbackType.ERROR_MESSAGE:
            return self._get_error_message()
        elif fallback_type == FallbackType.CHECKLIST:
            return self._get_checklist_fallback(query)
        elif fallback_type == FallbackType.INTERACTIVE_SCRIPT:
            return self._get_interactive_script_fallback(query)
        elif fallback_type == FallbackType.TEMPLATE:
            return self._get_template_fallback(query)
        else:
            return self._get_default_fallback()
    
    def _get_error_message(self) -> str:
        """Get error message for unsafe content."""
        return """I apologize, but I cannot provide a confident response to your request due to content safety concerns. 

Please try rephrasing your question or contact support if you believe this is an error."""
    
    def _get_checklist_fallback(self, query: str) -> str:
        """Get checklist-based fallback response."""
        return f"""I don't have enough confidence to provide a complete answer to: "{query}"

Here's a general checklist approach you can follow:

□ Break down your question into smaller, specific parts
□ Research each component individually
□ Consult relevant documentation or experts
□ Test your approach with a small pilot
□ Iterate based on results
□ Document your findings for future reference

Would you like to rephrase your question more specifically?"""
    
    def _get_interactive_script_fallback(self, query: str) -> str:
        """Get interactive script fallback response."""
        return f"""I need more information to provide a confident answer to: "{query}"

Let's work through this step by step:

1. What is your main goal or objective?
2. What constraints or limitations do you have?
3. What resources are available to you?
4. What is your timeline?
5. What have you already tried?

Please provide answers to these questions, and I'll be better able to help you."""
    
    def _get_template_fallback(self, query: str) -> str:
        """Get template-based fallback response."""
        return f"""I cannot provide a confident answer to: "{query}"

Here's a template approach to help you find the information you need:

**Research Steps:**
- Identify key terms and concepts
- Check authoritative sources
- Look for recent updates or changes
- Verify information from multiple sources

**Questions to Consider:**
- What specific aspect interests you most?
- How does this relate to your current situation?
- What level of detail do you need?

Please provide more specific details about what you're looking for."""
    
    def _get_default_fallback(self) -> str:
        """Get default fallback response."""
        return """I don't have enough confidence to provide a reliable answer to your question right now.

This could be due to:
- Insufficient context
- Ambiguous phrasing
- Complex topic requiring more research

Please try rephrasing your question with more specific details, or break it down into smaller parts."""
    
    def _generate_confidence_explanation(
        self,
        factors: Dict[str, float],
        overall_score: float,
        meets_threshold: bool
    ) -> str:
        """Generate explanation for confidence score."""
        explanation = f"Overall confidence: {overall_score:.2f} (threshold: {self.confidence_threshold})\n"
        explanation += "Factors:\n"
        
        for factor, score in factors.items():
            explanation += f"  - {factor.replace('_', ' ').title()}: {score:.2f}\n"
        
        if meets_threshold:
            explanation += "✓ Confidence threshold met"
        else:
            explanation += "✗ Confidence below threshold - fallback recommended"
        
        return explanation
    
    def _initialize_fallback_resources(self) -> None:
        """Initialize fallback resource files."""
        # Create basic fallback templates if they don't exist
        templates = {
            'checklists.json': {
                'general_process': [
                    'Define the problem clearly',
                    'Research available options',
                    'Evaluate pros and cons',
                    'Make a decision',
                    'Implement the solution',
                    'Monitor and adjust'
                ],
                'business_planning': [
                    'Define your business idea',
                    'Research your market',
                    'Identify your target customers',
                    'Develop your business model',
                    'Create financial projections',
                    'Write your business plan'
                ]
            },
            'scripts.json': {
                'problem_solving': [
                    'What exactly is the problem you\'re trying to solve?',
                    'Who is affected by this problem?',
                    'What would success look like?',
                    'What resources do you have available?',
                    'What are the potential risks?'
                ]
            }
        }
        
        for filename, content in templates.items():
            filepath = self.fallback_dir / filename
            if not filepath.exists():
                import json
                with open(filepath, 'w') as f:
                    json.dump(content, f, indent=2)
    
    def update_confidence_threshold(self, new_threshold: float) -> None:
        """Update confidence threshold."""
        if not 0.0 <= new_threshold <= 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
        
        self.confidence_threshold = new_threshold
        settings.update_feature_flag('confidence_threshold', new_threshold)
        self.logger.info(f"Confidence threshold updated to {new_threshold}")