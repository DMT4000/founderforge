#!/usr/bin/env python3
"""Security capabilities demonstration for FounderForge AI Cofounder.

This script demonstrates the security testing and filtering capabilities including:
- PII detection and filtering
- Injection attack prevention
- Data leak detection
- Security event logging
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.security_filter import EnhancedSecurityFilter
from src.confidence_manager import ContentFilter
from tests.test_security import SecurityTestSuite


def demo_pii_detection():
    """Demonstrate PII detection capabilities."""
    print("=== PII Detection Demo ===")
    
    filter = EnhancedSecurityFilter()
    
    test_cases = [
        "Contact me at john.doe@example.com for more information.",
        "Call me at (555) 123-4567 tomorrow.",
        "My SSN is 123-45-6789 for verification.",
        "Credit card: 4532 1234 5678 9012",
        "Server IP: 192.168.1.100",
        "Visit https://internal.company.com/admin",
        "This is a normal business message without PII."
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test}")
        result = filter.filter_content(test)
        
        print(f"  Safe: {result.is_safe}")
        print(f"  Risk Score: {result.overall_risk_score:.2f}")
        print(f"  Threats Detected: {len(result.threats)}")
        
        if result.threats:
            for threat in result.threats:
                print(f"    - {threat.pattern_name} ({threat.severity.value})")
        
        if result.sanitized_content and result.sanitized_content != test:
            print(f"  Sanitized: {result.sanitized_content}")


def demo_injection_detection():
    """Demonstrate injection attack detection."""
    print("\n=== Injection Attack Detection Demo ===")
    
    filter = EnhancedSecurityFilter()
    
    test_cases = [
        "'; DROP TABLE users; --",
        "1' UNION SELECT * FROM passwords --",
        "test; rm -rf /",
        "<script>alert('XSS')</script>",
        "{{config.__class__.__init__.__globals__['os'].popen('ls').read()}}",
        "../../../etc/passwd",
        "This is a normal query about business strategy."
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test}")
        result = filter.filter_content(test)
        
        print(f"  Safe: {result.is_safe}")
        print(f"  Blocked: {result.blocked_content}")
        print(f"  Risk Score: {result.overall_risk_score:.2f}")
        
        if result.threats:
            for threat in result.threats:
                print(f"    - {threat.threat_type.value}: {threat.pattern_name} ({threat.severity.value})")


def demo_data_leak_detection():
    """Demonstrate data leak detection."""
    print("\n=== Data Leak Detection Demo ===")
    
    filter = EnhancedSecurityFilter()
    
    test_cases = [
        "My API key is sk-1234567890abcdefghijklmnopqrstuvwxyz123456",
        "Database: postgresql://user:password@localhost:5432/mydb",
        "Remember that my password is secret123",
        "Our revenue last quarter was $2.5M",
        "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE",
        "DEBUG=True, SECRET_KEY=django-insecure-abc123",
        "This is normal business discussion."
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test}")
        result = filter.filter_content(test)
        
        print(f"  Safe: {result.is_safe}")
        print(f"  Risk Score: {result.overall_risk_score:.2f}")
        
        if result.threats:
            for threat in result.threats:
                print(f"    - Data leak: {threat.pattern_name} ({threat.severity.value})")


def demo_integration_with_confidence_manager():
    """Demonstrate integration with existing confidence manager."""
    print("\n=== Integration with Confidence Manager Demo ===")
    
    from src.confidence_manager import ConfidenceManager
    
    confidence_manager = ConfidenceManager()
    
    test_cases = [
        ("Normal business question", "What are the best practices for startup funding?"),
        ("PII content", "Contact me at sensitive@company.com with your SSN 123-45-6789"),
        ("Injection attempt", "'; DROP TABLE users; --"),
        ("Data leak", "My API key is sk-1234567890abcdef")
    ]
    
    for name, content in test_cases:
        print(f"\n{name}: {content}")
        
        # Calculate confidence with security filtering
        confidence_score = confidence_manager.calculate_confidence(
            ai_confidence=0.9,
            content=content,
            context_quality=1.0,
            response_length=len(content)
        )
        
        print(f"  Overall Confidence: {confidence_score.score:.2f}")
        print(f"  Meets Threshold: {confidence_score.meets_threshold}")
        print(f"  Content Safety Factor: {confidence_score.factors.get('content_safety', 'N/A'):.2f}")
        
        if not confidence_score.meets_threshold:
            fallback = confidence_manager.get_fallback_response(content, confidence_score)
            print(f"  Fallback Type: {fallback.fallback_type.value}")


def demo_security_test_suite():
    """Demonstrate the security test suite."""
    print("\n=== Security Test Suite Demo ===")
    
    suite = SecurityTestSuite()
    
    print("Running comprehensive security tests...")
    results = suite.run_all_tests()
    
    print(f"\nTest Results Summary:")
    print(f"  Overall Security Score: {results['overall']['security_score']:.2%}")
    print(f"  Total Tests: {results['overall']['total_tests']}")
    print(f"  Passed Tests: {results['overall']['passed_tests']}")
    print(f"  Meets 90% Threshold: {results['overall']['meets_threshold']}")


def main():
    """Run all security demonstrations."""
    print("FounderForge Security Capabilities Demonstration")
    print("=" * 60)
    
    try:
        demo_pii_detection()
        demo_injection_detection()
        demo_data_leak_detection()
        demo_integration_with_confidence_manager()
        demo_security_test_suite()
        
        print("\n" + "=" * 60)
        print("Security demonstration completed successfully!")
        print("\nKey Features Demonstrated:")
        print("✓ PII Detection and Sanitization")
        print("✓ Injection Attack Prevention")
        print("✓ Data Leak Detection")
        print("✓ Integration with Confidence Manager")
        print("✓ Comprehensive Security Testing")
        print("✓ Configurable Security Policies")
        print("✓ Security Event Logging")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()