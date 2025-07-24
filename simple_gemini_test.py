#!/usr/bin/env python3
"""
Very simple test with minimal tokens.
"""

import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path for imports
sys.path.append('src')

import google.generativeai as genai
import os

def main():
    """Simple test."""
    print("🔍 Simple Gemini Test")
    print("=" * 30)
    
    # Configure API
    api_key = os.getenv('GEMINI_API_KEY')
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    # Very simple query
    query = "Hi"
    
    print(f"📝 Query: '{query}'")
    
    try:
        response = model.generate_content(
            query,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=20000,
            )
        )
        
        print(f"✅ Response received")
        print(f"🏁 Finish reason: {response.candidates[0].finish_reason}")
        
        if response.candidates[0].content.parts:
            print(f"📄 Content: '{response.candidates[0].content.parts[0].text}'")
        else:
            print("❌ No content parts")
            
        # Try with more tokens
        print("\n🔄 Trying with more tokens...")
        response2 = model.generate_content(
            "Hello",
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=20000,
            )
        )
        
        print(f"🏁 Finish reason: {response2.candidates[0].finish_reason}")
        if response2.candidates[0].content.parts:
            print(f"📄 Content: '{response2.candidates[0].content.parts[0].text}'")
        else:
            print("❌ No content parts")
            
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()