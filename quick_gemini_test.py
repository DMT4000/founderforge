#!/usr/bin/env python3
"""
Quick test to verify Gemini 2.5 Flash is working correctly.
"""

import sys
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path for imports
sys.path.append('src')

try:
    from gemini_client import GeminiClient, MockMode
except ImportError as e:
    print(f"Failed to import: {e}")
    sys.exit(1)


def main():
    """Test Gemini connection with a real business query."""
    print("🚀 Testing Gemini 2.5 Flash Connection")
    print("=" * 50)
    
    try:
        # Initialize client
        client = GeminiClient(mock_mode=MockMode.DISABLED)
        print("✅ Client initialized")
        
        # Test business-related query
        query = """
        I'm a startup founder working on an AI-powered customer service platform. 
        My current MRR is $50K with 200 enterprise customers. 
        What should be my top 3 priorities for the next quarter to prepare for Series A funding?
        """
        
        print(f"📝 Sending query: {query[:100]}...")
        
        start_time = time.time()
        response = client.generate_content(
            query.strip(),
            temperature=0.7,
            max_output_tokens=20000
        )
        total_time = time.time() - start_time
        
        print(f"✅ Response received in {total_time:.2f}s")
        print(f"📊 Confidence: {response.confidence:.2f}")
        print(f"🤖 Model: {response.model_used}")
        print(f"🔢 Tokens: {response.token_usage}")
        print(f"\n📄 Response:")
        print("-" * 50)
        print(response.content)
        print("-" * 50)
        
        # Test health check with fixed token limit
        print("\n🏥 Testing health check...")
        is_healthy = client.health_check()
        print(f"Health status: {'✅ Healthy' if is_healthy else '❌ Unhealthy'}")
        
        print("\n🎉 Gemini 2.5 Flash is working correctly!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print(f"Error type: {type(e).__name__}")
        sys.exit(1)


if __name__ == "__main__":
    main()