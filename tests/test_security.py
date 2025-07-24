"""Security testing capabilities for FounderForge AI Cofounder.

This module implements comprehensive security testing including:
- PII detection accuracy tests
- Injection attack simulation
- Data leak prevention validation
"""

import pytest
import json
import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

from src.confidence_manager import ContentFilter, FilterType, FilterResult
from src.context_manager import ContextAssembler
from src.memory_repository import MemoryRepository
from src.agents import AgentOrchestrator


@dataclass
class SecurityTestCase:
    """Security test case definition."""
    name: str
    input_text: str
    expected_detection: bool
    expected_patterns: List[str]
    severity: str  # 'low', 'medium', 'high', 'critical'
    category: str  # 'pii', 'injection', 'leak'


@dataclass
class SecurityTestResult:
    """Security test result."""
    test_name: str
    passed: bool
    detected_patterns: List[str]
    expected_patterns: List[str]
    false_positives: List[str]
    false_negatives: List[str]
    accuracy: float
    details: str


class PIIDetectionTester:
    """Tests for PII detection accuracy."""
    
    def __init__(self):
        self.content_filter = ContentFilter()
        self.test_cases = self._load_pii_test_cases()
    
    def _load_pii_test_cases(self) -> List[SecurityTestCase]:
        """Load PII detection test cases."""
        return [
            # Email addresses
            SecurityTestCase(
                name="email_simple",
                input_text="Contact me at john.doe@example.com for more info.",
                expected_detection=True,
                expected_patterns=["pii_email"],
                severity="medium",
                category="pii"
            ),
            SecurityTestCase(
                name="email_multiple",
                input_text="Send reports to admin@company.com and backup@company.org",
                expected_detection=True,
                expected_patterns=["pii_email"],
                severity="medium",
                category="pii"
            ),
            SecurityTestCase(
                name="email_obfuscated",
                input_text="Email me at john dot doe at example dot com",
                expected_detection=False,
                expected_patterns=[],
                severity="low",
                category="pii"
            ),
            
            # Phone numbers
            SecurityTestCase(
                name="phone_formatted",
                input_text="Call me at (555) 123-4567 tomorrow.",
                expected_detection=True,
                expected_patterns=["pii_phone"],
                severity="medium",
                category="pii"
            ),
            SecurityTestCase(
                name="phone_unformatted",
                input_text="My number is 5551234567",
                expected_detection=True,
                expected_patterns=["pii_phone"],
                severity="medium",
                category="pii"
            ),
            SecurityTestCase(
                name="phone_international",
                input_text="International: +1-555-123-4567",
                expected_detection=True,
                expected_patterns=["pii_phone"],
                severity="medium",
                category="pii"
            ),
            
            # SSN
            SecurityTestCase(
                name="ssn_formatted",
                input_text="My SSN is 123-45-6789.",
                expected_detection=True,
                expected_patterns=["pii_ssn"],
                severity="high",
                category="pii"
            ),
            SecurityTestCase(
                name="ssn_unformatted",
                input_text="SSN: 123456789",
                expected_detection=True,
                expected_patterns=["pii_ssn"],
                severity="high",
                category="pii"
            ),
            
            # Credit cards
            SecurityTestCase(
                name="credit_card_visa",
                input_text="Card number: 4532 1234 5678 9012",
                expected_detection=True,
                expected_patterns=["pii_credit_card"],
                severity="critical",
                category="pii"
            ),
            SecurityTestCase(
                name="credit_card_mastercard",
                input_text="MC: 5555-4444-3333-2222",
                expected_detection=True,
                expected_patterns=["pii_credit_card"],
                severity="critical",
                category="pii"
            ),
            
            # IP addresses
            SecurityTestCase(
                name="ip_address_v4",
                input_text="Server IP: 192.168.1.100",
                expected_detection=True,
                expected_patterns=["pii_ip_address"],
                severity="medium",
                category="pii"
            ),
            SecurityTestCase(
                name="ip_address_public",
                input_text="External IP is 203.0.113.42",
                expected_detection=True,
                expected_patterns=["pii_ip_address"],
                severity="medium",
                category="pii"
            ),
            
            # URLs
            SecurityTestCase(
                name="url_https",
                input_text="Check out https://www.example.com/private/data",
                expected_detection=True,
                expected_patterns=["pii_url"],
                severity="low",
                category="pii"
            ),
            SecurityTestCase(
                name="url_http",
                input_text="Visit http://internal.company.com/admin",
                expected_detection=True,
                expected_patterns=["pii_url"],
                severity="medium",
                category="pii"
            ),
            
            # Mixed PII
            SecurityTestCase(
                name="mixed_pii",
                input_text="Contact John at john@company.com or call (555) 123-4567. Server: 192.168.1.1",
                expected_detection=True,
                expected_patterns=["pii_email", "pii_phone", "pii_ip_address"],
                severity="high",
                category="pii"
            ),
            
            # False positives (should NOT detect)
            SecurityTestCase(
                name="false_positive_numbers",
                input_text="The year 2023 was great, and 123-456 is not a phone number.",
                expected_detection=False,
                expected_patterns=[],
                severity="low",
                category="pii"
            ),
            SecurityTestCase(
                name="false_positive_email_like",
                input_text="The format is name@domain.extension but this is just an example.",
                expected_detection=False,
                expected_patterns=[],
                severity="low",
                category="pii"
            ),
        ]
    
    def run_pii_detection_tests(self) -> List[SecurityTestResult]:
        """Run all PII detection tests."""
        results = []
        
        for test_case in self.test_cases:
            if test_case.category != "pii":
                continue
                
            filter_result = self.content_filter.filter_content(test_case.input_text)
            
            # Analyze results
            detected_patterns = filter_result.detected_patterns
            expected_patterns = test_case.expected_patterns
            
            # Calculate accuracy metrics
            true_positives = set(detected_patterns) & set(expected_patterns)
            false_positives = set(detected_patterns) - set(expected_patterns)
            false_negatives = set(expected_patterns) - set(detected_patterns)
            
            # Determine if test passed
            detection_correct = (filter_result.filter_type == FilterType.PII_DETECTED) == test_case.expected_detection
            patterns_correct = set(detected_patterns) == set(expected_patterns)
            passed = detection_correct and patterns_correct
            
            # Calculate accuracy
            if len(expected_patterns) > 0:
                accuracy = len(true_positives) / len(expected_patterns)
            else:
                accuracy = 1.0 if len(detected_patterns) == 0 else 0.0
            
            result = SecurityTestResult(
                test_name=test_case.name,
                passed=passed,
                detected_patterns=detected_patterns,
                expected_patterns=expected_patterns,
                false_positives=list(false_positives),
                false_negatives=list(false_negatives),
                accuracy=accuracy,
                details=f"Input: '{test_case.input_text}' | Filter: {filter_result.filter_type.value}"
            )
            
            results.append(result)
        
        return results
    
    def calculate_overall_accuracy(self, results: List[SecurityTestResult]) -> Dict[str, float]:
        """Calculate overall PII detection accuracy metrics."""
        if not results:
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.passed)
        
        # Calculate precision and recall
        all_true_positives = 0
        all_false_positives = 0
        all_false_negatives = 0
        
        for result in results:
            all_true_positives += len(set(result.detected_patterns) & set(result.expected_patterns))
            all_false_positives += len(result.false_positives)
            all_false_negatives += len(result.false_negatives)
        
        precision = all_true_positives / (all_true_positives + all_false_positives) if (all_true_positives + all_false_positives) > 0 else 0.0
        recall = all_true_positives / (all_true_positives + all_false_negatives) if (all_true_positives + all_false_negatives) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "accuracy": passed_tests / total_tests,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }


class InjectionAttackTester:
    """Tests for injection attack detection and prevention."""
    
    def __init__(self):
        self.test_cases = self._load_injection_test_cases()
    
    def _load_injection_test_cases(self) -> List[SecurityTestCase]:
        """Load injection attack test cases."""
        return [
            # SQL Injection attempts
            SecurityTestCase(
                name="sql_injection_basic",
                input_text="'; DROP TABLE users; --",
                expected_detection=True,
                expected_patterns=["sql_injection"],
                severity="critical",
                category="injection"
            ),
            SecurityTestCase(
                name="sql_injection_union",
                input_text="1' UNION SELECT * FROM passwords --",
                expected_detection=True,
                expected_patterns=["sql_injection"],
                severity="critical",
                category="injection"
            ),
            SecurityTestCase(
                name="sql_injection_comment",
                input_text="admin'/**/OR/**/1=1#",
                expected_detection=True,
                expected_patterns=["sql_injection"],
                severity="critical",
                category="injection"
            ),
            
            # Command injection
            SecurityTestCase(
                name="command_injection_basic",
                input_text="test; rm -rf /",
                expected_detection=True,
                expected_patterns=["command_injection"],
                severity="critical",
                category="injection"
            ),
            SecurityTestCase(
                name="command_injection_pipe",
                input_text="input | cat /etc/passwd",
                expected_detection=True,
                expected_patterns=["command_injection"],
                severity="critical",
                category="injection"
            ),
            SecurityTestCase(
                name="command_injection_backtick",
                input_text="test `whoami`",
                expected_detection=True,
                expected_patterns=["command_injection"],
                severity="critical",
                category="injection"
            ),
            
            # Script injection
            SecurityTestCase(
                name="script_injection_javascript",
                input_text="<script>alert('XSS')</script>",
                expected_detection=True,
                expected_patterns=["script_injection"],
                severity="high",
                category="injection"
            ),
            SecurityTestCase(
                name="script_injection_event",
                input_text="<img src=x onerror=alert('XSS')>",
                expected_detection=True,
                expected_patterns=["script_injection"],
                severity="high",
                category="injection"
            ),
            
            # Template injection
            SecurityTestCase(
                name="template_injection_jinja",
                input_text="{{config.__class__.__init__.__globals__['os'].popen('ls').read()}}",
                expected_detection=True,
                expected_patterns=["template_injection"],
                severity="critical",
                category="injection"
            ),
            SecurityTestCase(
                name="template_injection_simple",
                input_text="{{7*7}}",
                expected_detection=True,
                expected_patterns=["template_injection"],
                severity="medium",
                category="injection"
            ),
            
            # Path traversal
            SecurityTestCase(
                name="path_traversal_basic",
                input_text="../../../etc/passwd",
                expected_detection=True,
                expected_patterns=["path_traversal"],
                severity="high",
                category="injection"
            ),
            SecurityTestCase(
                name="path_traversal_encoded",
                input_text="..%2F..%2F..%2Fetc%2Fpasswd",
                expected_detection=True,
                expected_patterns=["path_traversal"],
                severity="high",
                category="injection"
            ),
            
            # LDAP injection
            SecurityTestCase(
                name="ldap_injection_basic",
                input_text="*)(uid=*))(|(uid=*",
                expected_detection=True,
                expected_patterns=["ldap_injection"],
                severity="high",
                category="injection"
            ),
            
            # NoSQL injection
            SecurityTestCase(
                name="nosql_injection_mongo",
                input_text="'; return db.users.find(); var dummy='",
                expected_detection=True,
                expected_patterns=["nosql_injection"],
                severity="critical",
                category="injection"
            ),
        ]
    
    def _detect_injection_patterns(self, text: str) -> List[str]:
        """Detect injection attack patterns in text."""
        patterns = {
            'sql_injection': [
                r"('|(\\'))+.*(;|--|#)",  # SQL injection with quotes and comments
                r"\b(union|select|insert|update|delete|drop|create|alter)\b.*\b(from|where|table)\b",
                r"1\s*=\s*1|1\s*=\s*'1'",  # Always true conditions
                r"or\s+1\s*=\s*1|and\s+1\s*=\s*1",
            ],
            'command_injection': [
                r"[;&|`$(){}[\]\\]",  # Command separators and special chars
                r"\b(rm|cat|ls|ps|kill|chmod|chown|sudo|su)\b",  # Common commands
                r"(\\x[0-9a-f]{2})+",  # Hex encoding
            ],
            'script_injection': [
                r"<script[^>]*>.*</script>",
                r"javascript:",
                r"on\w+\s*=",  # Event handlers
                r"<iframe|<object|<embed",
            ],
            'template_injection': [
                r"\{\{.*\}\}",  # Jinja2/Django templates
                r"\$\{.*\}",   # Various template engines
                r"<%.*%>",     # ASP/JSP
            ],
            'path_traversal': [
                r"\.\./",
                r"\.\.\\",
                r"%2e%2e%2f",  # URL encoded
                r"%2e%2e%5c",
            ],
            'ldap_injection': [
                r"\*\)\(",
                r"\|\(",
                r"&\(",
            ],
            'nosql_injection': [
                r"\$where",
                r"\$ne",
                r"return\s+db\.",
            ]
        }
        
        detected = []
        for injection_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
                    detected.append(injection_type)
                    break  # Only add each type once
        
        return detected
    
    def run_injection_tests(self) -> List[SecurityTestResult]:
        """Run all injection attack tests."""
        results = []
        
        for test_case in self.test_cases:
            if test_case.category != "injection":
                continue
            
            detected_patterns = self._detect_injection_patterns(test_case.input_text)
            
            # Analyze results
            expected_patterns = test_case.expected_patterns
            
            # Calculate accuracy metrics
            true_positives = set(detected_patterns) & set(expected_patterns)
            false_positives = set(detected_patterns) - set(expected_patterns)
            false_negatives = set(expected_patterns) - set(detected_patterns)
            
            # Determine if test passed
            detection_correct = (len(detected_patterns) > 0) == test_case.expected_detection
            patterns_correct = len(false_negatives) == 0  # All expected patterns should be detected
            passed = detection_correct and patterns_correct
            
            # Calculate accuracy
            if len(expected_patterns) > 0:
                accuracy = len(true_positives) / len(expected_patterns)
            else:
                accuracy = 1.0 if len(detected_patterns) == 0 else 0.0
            
            result = SecurityTestResult(
                test_name=test_case.name,
                passed=passed,
                detected_patterns=detected_patterns,
                expected_patterns=expected_patterns,
                false_positives=list(false_positives),
                false_negatives=list(false_negatives),
                accuracy=accuracy,
                details=f"Input: '{test_case.input_text}' | Detected: {detected_patterns}"
            )
            
            results.append(result)
        
        return results


class DataLeakTester:
    """Tests for data leak prevention."""
    
    def __init__(self):
        self.test_cases = self._load_data_leak_test_cases()
    
    def _load_data_leak_test_cases(self) -> List[SecurityTestCase]:
        """Load data leak prevention test cases."""
        return [
            # Memory leaks
            SecurityTestCase(
                name="memory_leak_user_data",
                input_text="Remember that my password is secret123 and my API key is sk-1234567890abcdef",
                expected_detection=True,
                expected_patterns=["sensitive_data_leak"],
                severity="critical",
                category="leak"
            ),
            SecurityTestCase(
                name="memory_leak_business_data",
                input_text="Our revenue last quarter was $2.5M and we have 50,000 users",
                expected_detection=True,
                expected_patterns=["business_data_leak"],
                severity="high",
                category="leak"
            ),
            
            # API key leaks
            SecurityTestCase(
                name="api_key_leak_openai",
                input_text="My OpenAI key is sk-1234567890abcdefghijklmnopqrstuvwxyz123456",
                expected_detection=True,
                expected_patterns=["api_key_leak"],
                severity="critical",
                category="leak"
            ),
            SecurityTestCase(
                name="api_key_leak_aws",
                input_text="AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE",
                expected_detection=True,
                expected_patterns=["api_key_leak"],
                severity="critical",
                category="leak"
            ),
            
            # Database connection strings
            SecurityTestCase(
                name="db_connection_leak",
                input_text="Database: postgresql://user:password@localhost:5432/mydb",
                expected_detection=True,
                expected_patterns=["db_connection_leak"],
                severity="critical",
                category="leak"
            ),
            
            # Internal system info
            SecurityTestCase(
                name="system_info_leak",
                input_text="Server hostname: prod-server-01.internal.company.com",
                expected_detection=True,
                expected_patterns=["system_info_leak"],
                severity="medium",
                category="leak"
            ),
            
            # Configuration leaks
            SecurityTestCase(
                name="config_leak_env",
                input_text="DEBUG=True, SECRET_KEY=django-insecure-abc123",
                expected_detection=True,
                expected_patterns=["config_leak"],
                severity="high",
                category="leak"
            ),
        ]
    
    def _detect_data_leak_patterns(self, text: str) -> List[str]:
        """Detect data leak patterns in text."""
        patterns = {
            'sensitive_data_leak': [
                r"\b(password|passwd|pwd)\s*[:=]\s*\S+",
                r"\b(token|key)\s*[:=]\s*\S+",
                r"\b(secret|private)\s*[:=]\s*\S+",
            ],
            'business_data_leak': [
                r"\$[\d,]+\.?\d*[MKB]?",  # Money amounts
                r"\b\d+,?\d*\s*(users|customers|clients)",
                r"\b(revenue|profit|loss|earnings)\b.*\$",
            ],
            'api_key_leak': [
                r"sk-[a-zA-Z0-9]{32,}",  # OpenAI style
                r"AKIA[0-9A-Z]{16}",     # AWS Access Key
                r"AIza[0-9A-Za-z\\-_]{35}",  # Google API Key
                r"ya29\.[0-9A-Za-z\\-_]+",   # Google OAuth
            ],
            'db_connection_leak': [
                r"(postgresql|mysql|mongodb)://[^/\s]+/",
                r"(host|server)\s*[:=]\s*[^\s;]+",
                r"(database|db)\s*[:=]\s*[^\s;]+",
            ],
            'system_info_leak': [
                r"\b[a-zA-Z0-9-]+\.internal\.[a-zA-Z0-9.-]+",
                r"\b(hostname|server)\s*[:=]\s*[^\s;]+",
                r"192\.168\.\d+\.\d+",  # Internal IPs
            ],
            'config_leak': [
                r"DEBUG\s*=\s*True",
                r"SECRET_KEY\s*=\s*['\"][^'\"]+['\"]",
                r"(API_KEY|ACCESS_TOKEN)\s*=\s*['\"][^'\"]+['\"]",
            ]
        }
        
        detected = []
        for leak_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                if re.search(pattern, text, re.IGNORECASE):
                    detected.append(leak_type)
                    break  # Only add each type once
        
        return detected
    
    def run_data_leak_tests(self) -> List[SecurityTestResult]:
        """Run all data leak prevention tests."""
        results = []
        
        for test_case in self.test_cases:
            if test_case.category != "leak":
                continue
            
            detected_patterns = self._detect_data_leak_patterns(test_case.input_text)
            
            # Analyze results
            expected_patterns = test_case.expected_patterns
            
            # Calculate accuracy metrics
            true_positives = set(detected_patterns) & set(expected_patterns)
            false_positives = set(detected_patterns) - set(expected_patterns)
            false_negatives = set(expected_patterns) - set(detected_patterns)
            
            # Determine if test passed
            detection_correct = (len(detected_patterns) > 0) == test_case.expected_detection
            patterns_correct = len(false_negatives) == 0
            passed = detection_correct and patterns_correct
            
            # Calculate accuracy
            if len(expected_patterns) > 0:
                accuracy = len(true_positives) / len(expected_patterns)
            else:
                accuracy = 1.0 if len(detected_patterns) == 0 else 0.0
            
            result = SecurityTestResult(
                test_name=test_case.name,
                passed=passed,
                detected_patterns=detected_patterns,
                expected_patterns=expected_patterns,
                false_positives=list(false_positives),
                false_negatives=list(false_negatives),
                accuracy=accuracy,
                details=f"Input: '{test_case.input_text}' | Detected: {detected_patterns}"
            )
            
            results.append(result)
        
        return results


class SecurityTestSuite:
    """Comprehensive security testing suite."""
    
    def __init__(self):
        self.pii_tester = PIIDetectionTester()
        self.injection_tester = InjectionAttackTester()
        self.leak_tester = DataLeakTester()
        self.results_dir = Path("data/security_test_results")
        self.results_dir.mkdir(exist_ok=True)
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all security tests and return comprehensive results."""
        print("Running comprehensive security test suite...")
        
        # Run PII detection tests
        print("Testing PII detection accuracy...")
        pii_results = self.pii_tester.run_pii_detection_tests()
        pii_metrics = self.pii_tester.calculate_overall_accuracy(pii_results)
        
        # Run injection attack tests
        print("Testing injection attack detection...")
        injection_results = self.injection_tester.run_injection_tests()
        
        # Run data leak tests
        print("Testing data leak prevention...")
        leak_results = self.leak_tester.run_data_leak_tests()
        
        # Compile overall results
        all_results = {
            'pii_detection': {
                'results': pii_results,
                'metrics': pii_metrics,
                'total_tests': len(pii_results),
                'passed_tests': sum(1 for r in pii_results if r.passed)
            },
            'injection_attacks': {
                'results': injection_results,
                'total_tests': len(injection_results),
                'passed_tests': sum(1 for r in injection_results if r.passed)
            },
            'data_leaks': {
                'results': leak_results,
                'total_tests': len(leak_results),
                'passed_tests': sum(1 for r in leak_results if r.passed)
            }
        }
        
        # Calculate overall security score
        total_tests = sum(cat['total_tests'] for cat in all_results.values())
        total_passed = sum(cat['passed_tests'] for cat in all_results.values())
        overall_score = total_passed / total_tests if total_tests > 0 else 0.0
        
        all_results['overall'] = {
            'total_tests': total_tests,
            'passed_tests': total_passed,
            'security_score': overall_score,
            'meets_threshold': overall_score >= 0.9  # 90% threshold from requirements
        }
        
        # Save results
        self._save_results(all_results)
        
        # Print summary
        self._print_summary(all_results)
        
        return all_results
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save test results to file."""
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"security_test_results_{timestamp}.json"
        
        # Convert results to JSON-serializable format
        json_results = self._convert_results_to_json(results)
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"Security test results saved to: {results_file}")
    
    def _convert_results_to_json(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Convert results to JSON-serializable format."""
        json_results = {}
        
        for category, data in results.items():
            if category == 'overall':
                json_results[category] = data
            else:
                json_results[category] = {
                    'total_tests': data['total_tests'],
                    'passed_tests': data['passed_tests'],
                    'results': []
                }
                
                if 'metrics' in data:
                    json_results[category]['metrics'] = data['metrics']
                
                for result in data['results']:
                    json_results[category]['results'].append({
                        'test_name': result.test_name,
                        'passed': result.passed,
                        'detected_patterns': result.detected_patterns,
                        'expected_patterns': result.expected_patterns,
                        'false_positives': result.false_positives,
                        'false_negatives': result.false_negatives,
                        'accuracy': result.accuracy,
                        'details': result.details
                    })
        
        return json_results
    
    def _print_summary(self, results: Dict[str, Any]) -> None:
        """Print test results summary."""
        print("\n" + "="*60)
        print("SECURITY TEST SUITE RESULTS")
        print("="*60)
        
        # Overall results
        overall = results['overall']
        print(f"Overall Security Score: {overall['security_score']:.2%}")
        print(f"Total Tests: {overall['total_tests']}")
        print(f"Passed Tests: {overall['passed_tests']}")
        print(f"Meets 90% Threshold: {'✓' if overall['meets_threshold'] else '✗'}")
        
        print("\n" + "-"*40)
        
        # Category breakdown
        for category, data in results.items():
            if category == 'overall':
                continue
            
            category_name = category.replace('_', ' ').title()
            score = data['passed_tests'] / data['total_tests'] if data['total_tests'] > 0 else 0.0
            
            print(f"{category_name}: {score:.2%} ({data['passed_tests']}/{data['total_tests']})")
            
            if 'metrics' in data:
                metrics = data['metrics']
                print(f"  Precision: {metrics['precision']:.2%}")
                print(f"  Recall: {metrics['recall']:.2%}")
                print(f"  F1 Score: {metrics['f1']:.2%}")
            
            # Show failed tests
            failed_tests = [r for r in data['results'] if not r.passed]
            if failed_tests:
                print(f"  Failed Tests: {[t.test_name for t in failed_tests[:3]]}")
                if len(failed_tests) > 3:
                    print(f"    ... and {len(failed_tests) - 3} more")
        
        print("\n" + "="*60)


# Pytest test functions
class TestSecurityCapabilities:
    """Pytest test class for security capabilities."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.security_suite = SecurityTestSuite()
    
    def test_pii_detection_accuracy(self):
        """Test PII detection accuracy meets requirements."""
        pii_results = self.security_suite.pii_tester.run_pii_detection_tests()
        metrics = self.security_suite.pii_tester.calculate_overall_accuracy(pii_results)
        
        # Should meet 90% accuracy threshold
        assert metrics['accuracy'] >= 0.9, f"PII detection accuracy {metrics['accuracy']:.2%} below 90% threshold"
        assert metrics['precision'] >= 0.8, f"PII detection precision {metrics['precision']:.2%} below 80%"
        assert metrics['recall'] >= 0.8, f"PII detection recall {metrics['recall']:.2%} below 80%"
    
    def test_injection_attack_detection(self):
        """Test injection attack detection capabilities."""
        injection_results = self.security_suite.injection_tester.run_injection_tests()
        
        total_tests = len(injection_results)
        passed_tests = sum(1 for r in injection_results if r.passed)
        accuracy = passed_tests / total_tests if total_tests > 0 else 0.0
        
        # Should detect most injection attempts
        assert accuracy >= 0.8, f"Injection detection accuracy {accuracy:.2%} below 80% threshold"
        
        # Critical severity tests should all pass
        critical_tests = [r for r in injection_results if 'critical' in r.details.lower()]
        if critical_tests:
            critical_passed = sum(1 for r in critical_tests if r.passed)
            critical_accuracy = critical_passed / len(critical_tests)
            assert critical_accuracy >= 0.9, f"Critical injection tests accuracy {critical_accuracy:.2%} below 90%"
    
    def test_data_leak_prevention(self):
        """Test data leak prevention capabilities."""
        leak_results = self.security_suite.leak_tester.run_data_leak_tests()
        
        total_tests = len(leak_results)
        passed_tests = sum(1 for r in leak_results if r.passed)
        accuracy = passed_tests / total_tests if total_tests > 0 else 0.0
        
        # Should prevent most data leaks
        assert accuracy >= 0.8, f"Data leak prevention accuracy {accuracy:.2%} below 80% threshold"
    
    def test_overall_security_score(self):
        """Test overall security score meets requirements."""
        results = self.security_suite.run_all_tests()
        
        overall_score = results['overall']['security_score']
        meets_threshold = results['overall']['meets_threshold']
        
        # Should meet 90% overall threshold as per requirements
        assert overall_score >= 0.9, f"Overall security score {overall_score:.2%} below 90% threshold"
        assert meets_threshold, "Security tests do not meet the 90% threshold requirement"
    
    def test_content_filter_integration(self):
        """Test integration with existing content filter."""
        content_filter = ContentFilter()
        
        # Test with PII content
        pii_result = content_filter.filter_content("Contact me at test@example.com")
        assert pii_result.filter_type == FilterType.PII_DETECTED
        assert not pii_result.is_safe
        
        # Test with safe content
        safe_result = content_filter.filter_content("This is a normal business question")
        assert safe_result.filter_type == FilterType.SAFE
        assert safe_result.is_safe


if __name__ == "__main__":
    # Run security test suite when executed directly
    suite = SecurityTestSuite()
    results = suite.run_all_tests()
    
    # Exit with appropriate code
    if results['overall']['meets_threshold']:
        print("\n✓ All security tests passed!")
        exit(0)
    else:
        print("\n✗ Security tests failed to meet threshold!")
        exit(1)