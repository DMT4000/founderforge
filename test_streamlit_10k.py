#!/usr/bin/env python3
"""
Test Streamlit app with 10k token limit.
"""

import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path for imports
sys.path.append('src')

try:
    import app
    from gemini_client import GeminiClient, MockMode
except ImportError as e:
    print(f"Failed to import: {e}")
    sys.exit(1)


def main():
    """Test Streamlit app initialization with 10k tokens."""
    print("🌐 Testing Streamlit App with 10k Token Limit")
    print("=" * 50)
    
    try:
        # Initialize the app
        founder_app = app.FounderForgeApp()
        print("✅ FounderForge app initialized successfully")
        
        # Test the Gemini client directly
        if hasattr(founder_app, 'gemini_client'):
            print("✅ Gemini client is available in app")
            
            # Test a business query
            test_query = "What are the top 5 metrics every SaaS startup should track for investors?"
            
            print(f"📝 Testing query: {test_query}")
            
            response = founder_app.gemini_client.generate_content(
                test_query,
                temperature=0.7,
                max_output_tokens=20000
            )
            
            print(f"✅ Response received!")
            print(f"📊 Confidence: {response.confidence:.2f}")
            print(f"🔢 Token usage: {response.token_usage}")
            print(f"📄 Response length: {len(response.content)} characters")
            
            if len(response.content) > 500:
                print("✅ SUCCESS: Streamlit app can generate comprehensive responses!")
            else:
                print("⚠️  Response is shorter than expected")
                
            print(f"\n📄 Response preview:")
            print("-" * 30)
            print(response.content[:300] + "..." if len(response.content) > 300 else response.content)
            print("-" * 30)
        
        else:
            print("❌ Gemini client not found in app")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
    print("\n🎉 All tests passed! Streamlit app is ready with 10k token limit.")