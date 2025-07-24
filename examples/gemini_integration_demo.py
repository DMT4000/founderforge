#!/usr/bin/env python3
"""
Demo script showing Gemini API integration with confidence scoring and fallback mechanisms.

This script demonstrates:
1. Gemini API client with retry logic
2. Confidence scoring based on AI response and content safety
3. Fallback mechanisms for low confidence responses
4. Content filtering for PII and toxicity

Usage:
    python examples/gemini_integration_demo.py
"""

import os
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.gemini_client import GeminiClient, MockMode
from src.confidence_manager import ConfidenceManager


def demo_basic_api_call():
    """Demonstrate basic Gemini API call with mock mode."""
    print("=== Basic Gemini API Call Demo ===")
    
    # Use mock mode for demo (no API key required)
    client = GeminiClient(mock_mode=MockMode.SUCCESS)
    
    try:
        response = client.generate_content("What are the key steps to start a business?")
        print(f"✓ API Response: {response.content}")
        print(f"✓ Confidence: {response.confidence:.2f}")
        print(f"✓ Model: {response.model_used}")
        print(f"✓ Token Usage: {response.token_usage}")
        print(f"✓ Response Time: {response.response_time:.2f}s")
        print(f"✓ Is Mocked: {response.is_mocked}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print()


def demo_retry_logic():
    """Demonstrate retry logic with rate limiting."""
    print("=== Retry Logic Demo ===")
    
    # Simulate rate limiting
    client = GeminiClient(mock_mode=MockMode.RATE_LIMIT)
    
    try:
        response = client.generate_content("Tell me about funding options.")
        print(f"✓ Unexpected success: {response.content}")
        
    except Exception as e:
        print(f"✓ Expected error (rate limit): {e}")
    
    print()


def demo_confidence_scoring():
    """Demonstrate confidence scoring and fallback mechanisms."""
    print("=== Confidence Scoring Demo ===")
    
    confidence_manager = ConfidenceManager(confidence_threshold=0.8)
    
    # Test cases with different confidence levels
    test_cases = [
        {
            "name": "High Confidence Response",
            "ai_confidence": 0.95,
            "content": "To start a business, you should first validate your idea through market research.",
            "context_quality": 0.9,
            "response_length": 150
        },
        {
            "name": "Low Confidence Response",
            "ai_confidence": 0.4,
            "content": "I'm not sure about this.",
            "context_quality": 0.3,
            "response_length": 20
        },
        {
            "name": "Unsafe Content Response",
            "ai_confidence": 0.9,
            "content": "Contact me at john.doe@example.com, you stupid person!",
            "context_quality": 0.8,
            "response_length": 100
        }
    ]
    
    for test_case in test_cases:
        print(f"\n--- {test_case['name']} ---")
        
        confidence_score = confidence_manager.calculate_confidence(
            ai_confidence=test_case['ai_confidence'],
            content=test_case['content'],
            context_quality=test_case['context_quality'],
            response_length=test_case['response_length']
        )
        
        print(f"Overall Score: {confidence_score.score:.2f}")
        print(f"Meets Threshold: {confidence_score.meets_threshold}")
        print(f"Factors: {confidence_score.factors}")
        
        if not confidence_score.meets_threshold:
            fallback = confidence_manager.get_fallback_response(
                "How do I start a business?",
                confidence_score
            )
            print(f"Fallback Type: {fallback.fallback_type.value}")
            print(f"Fallback Content: {fallback.content[:100]}...")
    
    print()


def demo_content_filtering():
    """Demonstrate content filtering for PII and toxicity."""
    print("=== Content Filtering Demo ===")
    
    confidence_manager = ConfidenceManager()
    content_filter = confidence_manager.content_filter
    
    test_contents = [
        "This is a safe business advice message.",
        "Contact me at john.doe@example.com for more information.",
        "Call me at (555) 123-4567 tomorrow.",
        "That's a stupid idea and you're an idiot.",
        "I will destroy your business plan.",
        "Email me at bad@example.com, you moron!"
    ]
    
    for content in test_contents:
        result = content_filter.filter_content(content)
        print(f"\nContent: '{content[:50]}...'")
        print(f"Safe: {result.is_safe}")
        print(f"Filter Type: {result.filter_type.value}")
        print(f"Detected Patterns: {result.detected_patterns}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Explanation: {result.explanation}")
    
    print()


def demo_integrated_workflow():
    """Demonstrate integrated workflow with API call, confidence scoring, and fallbacks."""
    print("=== Integrated Workflow Demo ===")
    
    # Initialize components
    client = GeminiClient(mock_mode=MockMode.SUCCESS)
    confidence_manager = ConfidenceManager(confidence_threshold=0.8)
    
    query = "What are the best funding options for a tech startup?"
    
    try:
        # Step 1: Get AI response
        print(f"Query: {query}")
        ai_response = client.generate_content(query)
        print(f"✓ AI Response received: {ai_response.content[:100]}...")
        
        # Step 2: Calculate confidence
        confidence_score = confidence_manager.calculate_confidence(
            ai_confidence=ai_response.confidence,
            content=ai_response.content,
            context_quality=0.8,  # Assume good context
            response_length=len(ai_response.content)
        )
        
        print(f"✓ Confidence Score: {confidence_score.score:.2f}")
        print(f"✓ Meets Threshold: {confidence_score.meets_threshold}")
        
        # Step 3: Use response or fallback
        if confidence_score.meets_threshold:
            print("✓ Using AI response (high confidence)")
            final_response = ai_response.content
        else:
            print("⚠ Using fallback response (low confidence)")
            fallback = confidence_manager.get_fallback_response(query, confidence_score)
            final_response = fallback.content
        
        print(f"✓ Final Response: {final_response[:200]}...")
        
    except Exception as e:
        print(f"✗ Error in workflow: {e}")
    
    print()


def main():
    """Run all demo functions."""
    print("FounderForge Gemini Integration Demo")
    print("=" * 50)
    
    demo_basic_api_call()
    demo_retry_logic()
    demo_confidence_scoring()
    demo_content_filtering()
    demo_integrated_workflow()
    
    print("Demo completed!")


if __name__ == "__main__":
    main()