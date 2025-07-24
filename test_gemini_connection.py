#!/usr/bin/env python3
"""
Test script to verify Gemini API connection and configuration.
"""

import os
import sys
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path for imports
sys.path.append('src')

try:
    from gemini_client import GeminiClient, MockMode
    from config.settings import settings
except ImportError as e:
    print(f"Failed to import required modules: {e}")
    sys.exit(1)


def test_env_configuration():
    """Test environment configuration."""
    print("🔧 Testing Environment Configuration")
    print("=" * 50)
    
    # Check if API key is set
    api_key = os.getenv('GEMINI_API_KEY')
    if api_key:
        print(f"✅ GEMINI_API_KEY is set (length: {len(api_key)})")
        # Mask the key for security
        masked_key = api_key[:8] + "*" * (len(api_key) - 12) + api_key[-4:] if len(api_key) > 12 else "*" * len(api_key)
        print(f"   Key preview: {masked_key}")
    else:
        print("❌ GEMINI_API_KEY is not set")
        return False
    
    # Check other environment variables
    debug = os.getenv('DEBUG', 'false')
    log_level = os.getenv('LOG_LEVEL', 'INFO')
    
    print(f"✅ DEBUG: {debug}")
    print(f"✅ LOG_LEVEL: {log_level}")
    
    # Check settings module
    try:
        print(f"✅ Settings module loaded")
        print(f"   Gemini API key available: {'Yes' if settings.gemini_api_key else 'No'}")
    except Exception as e:
        print(f"❌ Settings module error: {e}")
        return False
    
    return True


def test_gemini_client_initialization():
    """Test Gemini client initialization."""
    print("\n🚀 Testing Gemini Client Initialization")
    print("=" * 50)
    
    try:
        # Test with mock mode first
        print("Testing with mock mode...")
        mock_client = GeminiClient(mock_mode=MockMode.SUCCESS)
        print("✅ Mock client initialized successfully")
        
        # Test actual client initialization
        print("Testing actual client initialization...")
        real_client = GeminiClient(mock_mode=MockMode.DISABLED)
        print("✅ Real client initialized successfully")
        
        return real_client
        
    except Exception as e:
        print(f"❌ Client initialization failed: {e}")
        return None


def test_mock_response():
    """Test mock response generation."""
    print("\n🎭 Testing Mock Response Generation")
    print("=" * 50)
    
    try:
        client = GeminiClient(mock_mode=MockMode.SUCCESS)
        
        test_prompts = [
            "Hello, how are you?",
            "Create a daily action plan for my startup",
            "Provide coaching advice for entrepreneurs",
            "Validate this business plan"
        ]
        
        for prompt in test_prompts:
            print(f"\nTesting prompt: '{prompt[:30]}...'")
            start_time = time.time()
            
            response = client.generate_content(prompt, max_output_tokens=20000)
            
            print(f"✅ Response generated in {response.response_time:.2f}s")
            print(f"   Content length: {len(response.content)} chars")
            print(f"   Confidence: {response.confidence:.2f}")
            print(f"   Model: {response.model_used}")
            print(f"   Is mocked: {response.is_mocked}")
            
        return True
        
    except Exception as e:
        print(f"❌ Mock response test failed: {e}")
        return False


def test_real_api_connection():
    """Test real API connection."""
    print("\n🌐 Testing Real Gemini API Connection")
    print("=" * 50)
    
    try:
        client = GeminiClient(mock_mode=MockMode.DISABLED)
        
        # Simple test prompt
        test_prompt = "Hello! Please respond with a brief greeting and confirm you are Gemini 2.5 Flash."
        
        print(f"Sending test prompt: '{test_prompt}'")
        print("Waiting for response...")
        
        start_time = time.time()
        response = client.generate_content(
            test_prompt,
            temperature=0.7,
            max_output_tokens=20000
        )
        total_time = time.time() - start_time
        
        print(f"✅ API connection successful!")
        print(f"   Response time: {total_time:.2f}s")
        print(f"   Model response time: {response.response_time:.2f}s")
        print(f"   Content: {response.content[:200]}...")
        print(f"   Confidence: {response.confidence:.2f}")
        print(f"   Model: {response.model_used}")
        print(f"   Token usage: {response.token_usage}")
        
        return True
        
    except Exception as e:
        print(f"❌ Real API connection failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        return False


def test_health_check():
    """Test health check functionality."""
    print("\n🏥 Testing Health Check")
    print("=" * 50)
    
    try:
        client = GeminiClient(mock_mode=MockMode.DISABLED)
        
        print("Performing health check...")
        is_healthy = client.health_check()
        
        if is_healthy:
            print("✅ Health check passed - API is working correctly")
        else:
            print("❌ Health check failed - API may have issues")
        
        return is_healthy
        
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False


def test_error_handling():
    """Test error handling with different mock modes."""
    print("\n⚠️  Testing Error Handling")
    print("=" * 50)
    
    # Test rate limit handling
    try:
        print("Testing rate limit handling...")
        client = GeminiClient(mock_mode=MockMode.RATE_LIMIT)
        response = client.generate_content("Test prompt")
        print("❌ Expected rate limit error but got response")
    except Exception as e:
        print(f"✅ Rate limit error handled correctly: {type(e).__name__}")
    
    # Test failure handling
    try:
        print("Testing failure handling...")
        client = GeminiClient(mock_mode=MockMode.FAILURE)
        response = client.generate_content("Test prompt")
        print("❌ Expected failure error but got response")
    except Exception as e:
        print(f"✅ Failure error handled correctly: {type(e).__name__}")
    
    return True


def main():
    """Run all tests."""
    print("🧪 FounderForge Gemini API Connection Test")
    print("=" * 60)
    
    tests = [
        ("Environment Configuration", test_env_configuration),
        ("Client Initialization", test_gemini_client_initialization),
        ("Mock Response Generation", test_mock_response),
        ("Real API Connection", test_real_api_connection),
        ("Health Check", test_health_check),
        ("Error Handling", test_error_handling),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! Gemini API is properly configured and working.")
    else:
        print("⚠️  Some tests failed. Please check the configuration and API key.")
        sys.exit(1)


if __name__ == "__main__":
    main()