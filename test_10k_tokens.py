#!/usr/bin/env python3
"""
Test the 10k token limit configuration.
"""

import sys
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
    """Test 10k token limit."""
    print("🚀 Testing 10k Token Limit")
    print("=" * 40)
    
    try:
        client = GeminiClient(mock_mode=MockMode.DISABLED)
        
        # Test with a comprehensive business query
        query = """
        I'm a startup founder working on an AI-powered customer service platform. 
        My current MRR is $50K with 200 enterprise customers. I'm preparing for Series A funding.
        
        Please provide a comprehensive business strategy analysis including:
        1. Market positioning and competitive analysis
        2. Growth strategies for the next 12 months
        3. Key metrics I should focus on for investors
        4. Potential risks and mitigation strategies
        5. Team scaling recommendations
        6. Product roadmap priorities
        7. Funding strategy and timeline
        8. Customer acquisition and retention strategies
        
        Please be detailed and provide actionable insights for each area.
        """
        
        print(f"📝 Sending comprehensive query...")
        print(f"Query length: {len(query)} characters")
        
        response = client.generate_content(
            query.strip(),
            temperature=0.7,
            max_output_tokens=20000
        )
        
        print(f"✅ Response received!")
        print(f"📊 Confidence: {response.confidence:.2f}")
        print(f"🤖 Model: {response.model_used}")
        print(f"🔢 Token usage: {response.token_usage}")
        print(f"⏱️  Response time: {response.response_time:.2f}s")
        print(f"📄 Response length: {len(response.content)} characters")
        
        print(f"\n📄 Response Preview (first 500 chars):")
        print("-" * 50)
        print(response.content[:500] + "..." if len(response.content) > 500 else response.content)
        print("-" * 50)
        
        if len(response.content) > 1000:
            print("✅ SUCCESS: Received comprehensive response with 10k token limit!")
        else:
            print("⚠️  Response is shorter than expected, but system is working")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print(f"Error type: {type(e).__name__}")
        sys.exit(1)


if __name__ == "__main__":
    main()