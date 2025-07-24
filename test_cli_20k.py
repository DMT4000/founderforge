#!/usr/bin/env python3
"""
Test CLI with 20k token limit directly.
"""

import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path for imports
sys.path.append('src')

try:
    from cli import FounderForgeCLI
except ImportError as e:
    print(f"Failed to import: {e}")
    sys.exit(1)


def main():
    """Test CLI with 20k tokens."""
    print("🖥️  Testing CLI with 20k Token Limit")
    print("=" * 50)
    
    try:
        cli = FounderForgeCLI()
        
        # Test direct message processing (bypassing agent workflow)
        user_id = "test_user"
        message = "What are the key components of a successful SaaS business model?"
        
        print(f"📝 Testing message: {message}")
        
        response = cli.process_message(user_id, message)
        
        print(f"✅ Response received!")
        print(f"📊 Confidence: {response.get('confidence', 'N/A'):.2f}")
        print(f"⏱️  Processing time: {response.get('processing_time', 'N/A'):.2f}s")
        print(f"🔢 Token usage: {response.get('token_usage', 'N/A')}")
        print(f"📄 Response length: {len(response.get('content', ''))} characters")
        
        content = response.get('content', '')
        if len(content) > 1000:
            print("✅ SUCCESS: CLI generated comprehensive response with 20k limit!")
        else:
            print("⚠️  Response is shorter than expected")
        
        print(f"\n📄 Response preview:")
        print("-" * 40)
        print(content[:500] + "..." if len(content) > 500 else content)
        print("-" * 40)
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print(f"Error type: {type(e).__name__}")
        sys.exit(1)


if __name__ == "__main__":
    main()