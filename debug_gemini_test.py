#!/usr/bin/env python3
"""
Debug test to understand Gemini response structure.
"""

import sys
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path for imports
sys.path.append('src')

import google.generativeai as genai
import os

def main():
    """Debug Gemini API response."""
    print("🔍 Debugging Gemini API Response")
    print("=" * 50)
    
    # Configure API
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ No API key found")
        return
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    # Simple test
    query = "Hello! Please provide 3 business tips for startups."
    
    print(f"📝 Query: {query}")
    
    try:
        response = model.generate_content(
            query,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=20000,
            )
        )
        
        print(f"✅ Response received")
        print(f"📊 Candidates: {len(response.candidates) if response.candidates else 0}")
        
        if response.candidates:
            candidate = response.candidates[0]
            print(f"🏁 Finish reason: {candidate.finish_reason}")
            print(f"📝 Content parts: {len(candidate.content.parts) if candidate.content.parts else 0}")
            
            if candidate.content.parts:
                for i, part in enumerate(candidate.content.parts):
                    print(f"   Part {i}: {part.text[:100]}...")
            
            # Safety ratings
            if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                print(f"🛡️ Safety ratings:")
                for rating in candidate.safety_ratings:
                    print(f"   {rating.category.name}: {rating.probability.name}")
        
        # Usage metadata
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            print(f"🔢 Token usage:")
            print(f"   Prompt: {response.usage_metadata.prompt_token_count}")
            print(f"   Completion: {response.usage_metadata.candidates_token_count}")
            print(f"   Total: {response.usage_metadata.total_token_count}")
        
        # Full response text
        if response.text:
            print(f"\n📄 Full response:")
            print("-" * 30)
            print(response.text)
            print("-" * 30)
        else:
            print("❌ No response text available")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"Error type: {type(e).__name__}")


if __name__ == "__main__":
    main()