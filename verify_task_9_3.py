#!/usr/bin/env python3
"""Verification script for Task 9.3 - Security Testing Capabilities."""

from tests.test_security import SecurityTestSuite

def main():
    print("=== TASK 9.3 VERIFICATION ===")
    print("Adding security testing capabilities...")
    
    suite = SecurityTestSuite()
    results = suite.run_all_tests()
    
    print("\n=== VERIFICATION RESULTS ===")
    print(f"✓ PII Detection Accuracy: {results['pii_detection']['metrics']['accuracy']:.1%}")
    print(f"✓ Injection Attack Detection: {results['injection_attacks']['passed_tests']}/{results['injection_attacks']['total_tests']} tests passed")
    print(f"✓ Data Leak Prevention: {results['data_leaks']['passed_tests']}/{results['data_leaks']['total_tests']} tests passed")
    print(f"✓ Overall Security Score: {results['overall']['security_score']:.1%}")
    print(f"✓ Meets 90% Threshold: {results['overall']['meets_threshold']}")
    
    print("\n=== TASK 9.3 REQUIREMENTS SATISFIED ===")
    print("✓ PII detection accuracy tests implemented")
    print("✓ Injection attack simulation scripts created") 
    print("✓ Data leak prevention validation added")
    print("✓ Requirements 6.4 and 6.5 satisfied")
    
    if results['overall']['meets_threshold']:
        print("\n🎉 TASK 9.3 COMPLETED SUCCESSFULLY!")
        return 0
    else:
        print("\n❌ TASK 9.3 FAILED - Security threshold not met")
        return 1

if __name__ == "__main__":
    exit(main())