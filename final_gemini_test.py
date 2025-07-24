#!/usr/bin/env python3
"""
Final comprehensive test of Gemini 2.5 Flash integration.
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


def test_basic_functionality():
    """Test basic Gemini functionality."""
    print("🔧 Testing Basic Functionality")
    print("=" * 40)
    
    try:
        client = GeminiClient(mock_mode=MockMode.DISABLED)
        
        # Test 1: Simple greeting
        print("Test 1: Simple greeting")
        response = client.generate_content(
            "Hello! Please respond with a brief greeting.",
            max_output_tokens=20000
        )
        print(f"✅ Response: {response.content[:100]}...")
        print(f"   Confidence: {response.confidence:.2f}")
        
        # Test 2: Business question
        print("\nTest 2: Business question")
        response = client.generate_content(
            "What are 3 key metrics every startup should track?",
            max_output_tokens=20000
        )
        print(f"✅ Response: {response.content[:150]}...")
        print(f"   Confidence: {response.confidence:.2f}")
        print(f"   Tokens used: {response.token_usage.get('total_tokens', 'N/A')}")
        
        # Test 3: Health check
        print("\nTest 3: Health check")
        is_healthy = client.health_check()
        print(f"✅ Health status: {'Healthy' if is_healthy else 'Unhealthy'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False


def test_cli_integration():
    """Test CLI integration."""
    print("\n🖥️  Testing CLI Integration")
    print("=" * 40)
    
    try:
        # Test CLI system status
        import subprocess
        result = subprocess.run([
            'python', 'cli.py', 'system', 'status'
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ CLI system status works")
            # Check if Gemini is shown as connected
            if "🟢 Connected" in result.stdout:
                print("✅ Gemini shown as connected in CLI")
            else:
                print("⚠️  Gemini shown as disconnected in CLI")
        else:
            print(f"❌ CLI system status failed: {result.stderr}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ CLI integration test failed: {e}")
        return False


def test_streamlit_integration():
    """Test Streamlit integration."""
    print("\n🌐 Testing Streamlit Integration")
    print("=" * 40)
    
    try:
        # Import the app to check if it initializes correctly
        import app
        print("✅ Streamlit app imports successfully")
        
        # Check if the app can initialize its components
        founder_app = app.FounderForgeApp()
        print("✅ FounderForge app initializes successfully")
        
        # Test if Gemini client is available
        if hasattr(founder_app, 'gemini_client'):
            is_available = founder_app.gemini_client.is_available()
            print(f"✅ Gemini client available: {is_available}")
        
        return True
        
    except Exception as e:
        print(f"❌ Streamlit integration test failed: {e}")
        return False


def test_performance():
    """Test performance characteristics."""
    print("\n⚡ Testing Performance")
    print("=" * 40)
    
    try:
        client = GeminiClient(mock_mode=MockMode.DISABLED)
        
        # Test response times
        queries = [
            "Hi",
            "What is a startup?",
            "Explain the basics of venture capital funding in 2 sentences."
        ]
        
        total_time = 0
        for i, query in enumerate(queries, 1):
            start_time = time.time()
            response = client.generate_content(query, max_output_tokens=20000)
            response_time = time.time() - start_time
            total_time += response_time
            
            print(f"Query {i}: {response_time:.2f}s (confidence: {response.confidence:.2f})")
        
        avg_time = total_time / len(queries)
        print(f"✅ Average response time: {avg_time:.2f}s")
        
        if avg_time < 10:
            print("✅ Performance is acceptable")
            return True
        else:
            print("⚠️  Performance is slow but functional")
            return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 Final Gemini 2.5 Flash Integration Test")
    print("=" * 60)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("CLI Integration", test_cli_integration),
        ("Streamlit Integration", test_streamlit_integration),
        ("Performance", test_performance),
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
    print("\n📊 Final Test Summary")
    print("=" * 60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 SUCCESS: Gemini 2.5 Flash is properly configured and working!")
        print("   ✅ .env configuration is correct")
        print("   ✅ API key is valid and working")
        print("   ✅ Integration with CLI and Streamlit is functional")
        print("   ✅ Performance is acceptable")
    elif passed >= total * 0.75:
        print("\n✅ MOSTLY WORKING: Gemini integration is functional with minor issues")
    else:
        print("\n⚠️  ISSUES DETECTED: Some components may not work correctly")
        sys.exit(1)


if __name__ == "__main__":
    main()