#!/usr/bin/env python3
"""
Test the 20k token limit configuration.
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
    """Test 20k token limit."""
    print("🚀 Testing 20k Token Limit")
    print("=" * 40)
    
    try:
        client = GeminiClient(mock_mode=MockMode.DISABLED)
        
        # Test with a very comprehensive business query
        query = """
        I'm a startup founder working on an AI-powered customer service platform. 
        My current MRR is $50K with 200 enterprise customers. I'm preparing for Series A funding.
        
        Please provide an extremely comprehensive business strategy analysis including:
        
        1. MARKET ANALYSIS:
        - Complete competitive landscape analysis
        - Market size and growth projections
        - Customer segmentation and personas
        - Industry trends and disruptions
        - Regulatory considerations
        
        2. BUSINESS MODEL OPTIMIZATION:
        - Revenue stream analysis and optimization
        - Pricing strategy recommendations
        - Customer acquisition cost analysis
        - Lifetime value calculations
        - Churn reduction strategies
        
        3. GROWTH STRATEGIES:
        - 12-month growth roadmap
        - Geographic expansion opportunities
        - Product line extensions
        - Partnership and integration opportunities
        - Marketing channel optimization
        
        4. OPERATIONAL EXCELLENCE:
        - Team scaling plan and organizational structure
        - Technology infrastructure requirements
        - Process optimization recommendations
        - Quality assurance frameworks
        - Customer success strategies
        
        5. FINANCIAL PLANNING:
        - Detailed financial projections (3-year)
        - Key metrics and KPI framework
        - Funding requirements and use of funds
        - Investor presentation strategy
        - Risk assessment and mitigation
        
        6. PRODUCT DEVELOPMENT:
        - Feature prioritization framework
        - Technical roadmap and architecture
        - User experience optimization
        - Integration and API strategy
        - Innovation pipeline
        
        7. MARKET POSITIONING:
        - Brand positioning and messaging
        - Competitive differentiation
        - Thought leadership strategy
        - Public relations approach
        - Community building initiatives
        
        8. RISK MANAGEMENT:
        - Business continuity planning
        - Cybersecurity considerations
        - Legal and compliance framework
        - Insurance and protection strategies
        - Crisis management protocols
        
        Please provide detailed, actionable insights for each area with specific recommendations, 
        timelines, metrics, and implementation strategies. Include real-world examples and 
        case studies where relevant.
        """
        
        print(f"📝 Sending extremely comprehensive query...")
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
        print(f"📝 Word count: ~{len(response.content.split())} words")
        
        print(f"\n📄 Response Preview (first 1000 chars):")
        print("-" * 60)
        print(response.content[:1000] + "..." if len(response.content) > 1000 else response.content)
        print("-" * 60)
        
        # Check if we got a substantial response
        if len(response.content) > 5000:
            print("✅ SUCCESS: Received extremely comprehensive response with 20k token limit!")
            print(f"   📈 Response is {len(response.content)} characters long")
            print(f"   🎯 Used {response.token_usage.get('completion_tokens', 'N/A')} completion tokens")
        elif len(response.content) > 2000:
            print("✅ GOOD: Received comprehensive response, though not at maximum capacity")
        else:
            print("⚠️  Response is shorter than expected, but system is working")
        
        # Test health check with 20k limit
        print(f"\n🏥 Testing health check with 20k limit...")
        is_healthy = client.health_check()
        print(f"Health status: {'✅ Healthy' if is_healthy else '❌ Unhealthy'}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print(f"Error type: {type(e).__name__}")
        sys.exit(1)


if __name__ == "__main__":
    main()