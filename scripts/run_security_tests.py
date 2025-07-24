#!/usr/bin/env python3
"""Security testing script for FounderForge AI Cofounder.

This script runs comprehensive security tests including:
- PII detection accuracy validation
- Injection attack simulation
- Data leak prevention testing

Usage:
    python scripts/run_security_tests.py [--category pii|injection|leak|all] [--verbose] [--save-report]
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_security import SecurityTestSuite, PIIDetectionTester, InjectionAttackTester, DataLeakTester


def run_pii_tests(verbose: bool = False) -> Dict[str, Any]:
    """Run PII detection tests."""
    print("Running PII Detection Tests...")
    print("-" * 40)
    
    tester = PIIDetectionTester()
    results = tester.run_pii_detection_tests()
    metrics = tester.calculate_overall_accuracy(results)
    
    if verbose:
        for result in results:
            status = "✓ PASS" if result.passed else "✗ FAIL"
            print(f"{status} {result.test_name}")
            if not result.passed:
                print(f"    Expected: {result.expected_patterns}")
                print(f"    Detected: {result.detected_patterns}")
                print(f"    False Positives: {result.false_positives}")
                print(f"    False Negatives: {result.false_negatives}")
    
    print(f"\nPII Detection Results:")
    print(f"  Accuracy: {metrics['accuracy']:.2%}")
    print(f"  Precision: {metrics['precision']:.2%}")
    print(f"  Recall: {metrics['recall']:.2%}")
    print(f"  F1 Score: {metrics['f1']:.2%}")
    print(f"  Tests Passed: {sum(1 for r in results if r.passed)}/{len(results)}")
    
    return {
        'results': results,
        'metrics': metrics,
        'category': 'pii_detection'
    }


def run_injection_tests(verbose: bool = False) -> Dict[str, Any]:
    """Run injection attack tests."""
    print("\nRunning Injection Attack Tests...")
    print("-" * 40)
    
    tester = InjectionAttackTester()
    results = tester.run_injection_tests()
    
    if verbose:
        for result in results:
            status = "✓ PASS" if result.passed else "✗ FAIL"
            print(f"{status} {result.test_name}")
            if not result.passed:
                print(f"    Expected: {result.expected_patterns}")
                print(f"    Detected: {result.detected_patterns}")
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r.passed)
    accuracy = passed_tests / total_tests if total_tests > 0 else 0.0
    
    print(f"\nInjection Attack Detection Results:")
    print(f"  Accuracy: {accuracy:.2%}")
    print(f"  Tests Passed: {passed_tests}/{total_tests}")
    
    # Show results by severity
    severity_stats = {}
    for result in results:
        # Extract severity from test case (would need to modify to store this)
        severity = "unknown"
        if "critical" in result.test_name or "sql" in result.test_name or "command" in result.test_name:
            severity = "critical"
        elif "high" in result.test_name or "script" in result.test_name:
            severity = "high"
        else:
            severity = "medium"
        
        if severity not in severity_stats:
            severity_stats[severity] = {'total': 0, 'passed': 0}
        severity_stats[severity]['total'] += 1
        if result.passed:
            severity_stats[severity]['passed'] += 1
    
    for severity, stats in severity_stats.items():
        accuracy = stats['passed'] / stats['total'] if stats['total'] > 0 else 0.0
        print(f"  {severity.title()} Severity: {accuracy:.2%} ({stats['passed']}/{stats['total']})")
    
    return {
        'results': results,
        'accuracy': accuracy,
        'category': 'injection_attacks'
    }


def run_leak_tests(verbose: bool = False) -> Dict[str, Any]:
    """Run data leak prevention tests."""
    print("\nRunning Data Leak Prevention Tests...")
    print("-" * 40)
    
    tester = DataLeakTester()
    results = tester.run_data_leak_tests()
    
    if verbose:
        for result in results:
            status = "✓ PASS" if result.passed else "✗ FAIL"
            print(f"{status} {result.test_name}")
            if not result.passed:
                print(f"    Expected: {result.expected_patterns}")
                print(f"    Detected: {result.detected_patterns}")
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r.passed)
    accuracy = passed_tests / total_tests if total_tests > 0 else 0.0
    
    print(f"\nData Leak Prevention Results:")
    print(f"  Accuracy: {accuracy:.2%}")
    print(f"  Tests Passed: {passed_tests}/{total_tests}")
    
    return {
        'results': results,
        'accuracy': accuracy,
        'category': 'data_leak_prevention'
    }


def generate_security_report(results: List[Dict[str, Any]], output_file: str = None) -> str:
    """Generate a comprehensive security report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Calculate overall statistics
    total_tests = sum(len(r['results']) for r in results)
    total_passed = sum(sum(1 for test in r['results'] if test.passed) for r in results)
    overall_accuracy = total_passed / total_tests if total_tests > 0 else 0.0
    
    report = f"""
# FounderForge Security Test Report
Generated: {timestamp}

## Executive Summary
- **Overall Security Score**: {overall_accuracy:.2%}
- **Total Tests Executed**: {total_tests}
- **Tests Passed**: {total_passed}
- **Meets 90% Threshold**: {'PASS' if overall_accuracy >= 0.9 else 'FAIL'}

## Test Categories

"""
    
    for result_set in results:
        category = result_set['category'].replace('_', ' ').title()
        results_list = result_set['results']
        passed = sum(1 for r in results_list if r.passed)
        total = len(results_list)
        accuracy = passed / total if total > 0 else 0.0
        
        report += f"### {category}\n"
        report += f"- **Accuracy**: {accuracy:.2%}\n"
        report += f"- **Tests**: {passed}/{total}\n"
        
        if 'metrics' in result_set:
            metrics = result_set['metrics']
            report += f"- **Precision**: {metrics['precision']:.2%}\n"
            report += f"- **Recall**: {metrics['recall']:.2%}\n"
            report += f"- **F1 Score**: {metrics['f1']:.2%}\n"
        
        # List failed tests
        failed_tests = [r for r in results_list if not r.passed]
        if failed_tests:
            report += f"- **Failed Tests**: {len(failed_tests)}\n"
            for test in failed_tests[:5]:  # Show first 5 failures
                report += f"  - {test.test_name}: {test.details}\n"
            if len(failed_tests) > 5:
                report += f"  - ... and {len(failed_tests) - 5} more\n"
        
        report += "\n"
    
    report += """
## Recommendations

### High Priority
- Address any critical security test failures immediately
- Ensure PII detection accuracy remains above 90%
- Implement additional injection attack patterns if needed

### Medium Priority
- Review false positive patterns to reduce noise
- Enhance data leak detection for new patterns
- Add more comprehensive test cases

### Low Priority
- Optimize performance of security filters
- Add more detailed logging for security events
- Consider implementing machine learning-based detection

## Security Compliance
- **PII Detection**: Required for privacy compliance
- **Injection Prevention**: Critical for system security
- **Data Leak Prevention**: Essential for data protection

---
*This report was generated automatically by the FounderForge security testing suite.*
"""
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\nSecurity report saved to: {output_file}")
    
    return report


def main():
    """Main function to run security tests."""
    parser = argparse.ArgumentParser(description="Run FounderForge security tests")
    parser.add_argument(
        '--category',
        choices=['pii', 'injection', 'leak', 'all'],
        default='all',
        help='Category of tests to run (default: all)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed test results'
    )
    parser.add_argument(
        '--save-report',
        action='store_true',
        help='Save detailed report to file'
    )
    parser.add_argument(
        '--output-dir',
        default='data/security_test_results',
        help='Directory to save results (default: data/security_test_results)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("FounderForge Security Test Suite")
    print("=" * 50)
    
    results = []
    
    try:
        if args.category in ['pii', 'all']:
            pii_results = run_pii_tests(args.verbose)
            results.append(pii_results)
        
        if args.category in ['injection', 'all']:
            injection_results = run_injection_tests(args.verbose)
            results.append(injection_results)
        
        if args.category in ['leak', 'all']:
            leak_results = run_leak_tests(args.verbose)
            results.append(leak_results)
        
        # Calculate overall results
        total_tests = sum(len(r['results']) for r in results)
        total_passed = sum(sum(1 for test in r['results'] if test.passed) for r in results)
        overall_accuracy = total_passed / total_tests if total_tests > 0 else 0.0
        
        print("\n" + "=" * 50)
        print("OVERALL SECURITY TEST RESULTS")
        print("=" * 50)
        print(f"Overall Accuracy: {overall_accuracy:.2%}")
        print(f"Total Tests: {total_tests}")
        print(f"Passed Tests: {total_passed}")
        print(f"Meets 90% Threshold: {'PASS' if overall_accuracy >= 0.9 else 'FAIL'}")
        
        # Save results if requested
        if args.save_report:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save JSON results
            json_file = output_dir / f"security_results_{timestamp}.json"
            json_data = {
                'timestamp': timestamp,
                'overall_accuracy': overall_accuracy,
                'total_tests': total_tests,
                'passed_tests': total_passed,
                'meets_threshold': overall_accuracy >= 0.9,
                'categories': {}
            }
            
            for result_set in results:
                category = result_set['category']
                json_data['categories'][category] = {
                    'results': [
                        {
                            'test_name': r.test_name,
                            'passed': r.passed,
                            'accuracy': r.accuracy,
                            'detected_patterns': r.detected_patterns,
                            'expected_patterns': r.expected_patterns,
                            'details': r.details
                        }
                        for r in result_set['results']
                    ]
                }
                if 'metrics' in result_set:
                    json_data['categories'][category]['metrics'] = result_set['metrics']
            
            with open(json_file, 'w') as f:
                json.dump(json_data, f, indent=2)
            
            # Save markdown report
            report_file = output_dir / f"security_report_{timestamp}.md"
            generate_security_report(results, str(report_file))
        
        # Exit with appropriate code
        if overall_accuracy >= 0.9:
            print("\nSecurity tests PASSED - meets 90% threshold!")
            return 0
        else:
            print(f"\nSecurity tests FAILED - {overall_accuracy:.2%} below 90% threshold!")
            return 1
    
    except Exception as e:
        print(f"\nError running security tests: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)