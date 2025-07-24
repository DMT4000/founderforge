"""Enhanced security filtering system for FounderForge AI Cofounder.

This module provides comprehensive security filtering including:
- Configurable PII detection
- Injection attack prevention
- Data leak detection
- Security event logging
"""

import re
import json
import logging
from logging_manager import get_logging_manager, LogLevel, LogCategory
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from datetime import datetime

from config.settings import settings


class SecurityLevel(Enum):
    """Security severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatType(Enum):
    """Types of security threats."""
    PII_DETECTION = "pii_detection"
    INJECTION_ATTACK = "injection_attack"
    DATA_LEAK = "data_leak"
    TOXICITY = "toxicity"


@dataclass
class SecurityThreat:
    """Detected security threat."""
    threat_type: ThreatType
    severity: SecurityLevel
    pattern_name: str
    matched_text: str
    confidence: float
    position: Tuple[int, int]  # Start and end positions
    description: str


@dataclass
class SecurityFilterResult:
    """Result of security filtering."""
    is_safe: bool
    threats: List[SecurityThreat]
    overall_risk_score: float
    blocked_content: bool
    sanitized_content: Optional[str]
    explanation: str


class SecurityConfig:
    """Security configuration manager."""
    
    def __init__(self, config_path: str = "config/security_config.json"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.logger = get_logging_manager().get_logger(__name__.split(".")[-1], LogCategory.SYSTEM)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load security configuration from file."""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.warning(f"Security config not found at {self.config_path}, using defaults")
            return self._get_default_config()
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in security config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default security configuration."""
        return {
            "security_policies": {
                "pii_detection": {"enabled": True, "accuracy_threshold": 0.9},
                "injection_prevention": {"enabled": True, "accuracy_threshold": 0.8},
                "data_leak_prevention": {"enabled": True, "accuracy_threshold": 0.8}
            },
            "testing_config": {
                "overall_threshold": 0.9
            }
        }
    
    def get_policy(self, policy_name: str) -> Dict[str, Any]:
        """Get security policy configuration."""
        return self.config.get("security_policies", {}).get(policy_name, {})
    
    def is_enabled(self, policy_name: str) -> bool:
        """Check if a security policy is enabled."""
        policy = self.get_policy(policy_name)
        return policy.get("enabled", False)
    
    def get_threshold(self, policy_name: str) -> float:
        """Get accuracy threshold for a policy."""
        policy = self.get_policy(policy_name)
        return policy.get("accuracy_threshold", 0.8)
    
    def get_patterns(self, policy_name: str) -> Dict[str, Any]:
        """Get patterns for a security policy."""
        policy = self.get_policy(policy_name)
        return policy.get("patterns", {})


class EnhancedSecurityFilter:
    """Enhanced security filter with configurable policies."""
    
    def __init__(self, config_path: str = "config/security_config.json"):
        self.config = SecurityConfig(config_path)
        self.logger = get_logging_manager().get_logger(__name__.split(".")[-1], LogCategory.SYSTEM)
        
        # Initialize pattern matchers
        self.pii_matcher = PIIPatternMatcher(self.config)
        self.injection_matcher = InjectionPatternMatcher(self.config)
        self.leak_matcher = DataLeakPatternMatcher(self.config)
        
        # Security event logging
        self.events_dir = Path("data/security_events")
        self.events_dir.mkdir(exist_ok=True)
    
    def filter_content(self, content: str, user_id: str = None) -> SecurityFilterResult:
        """Perform comprehensive security filtering on content."""
        threats = []
        
        # Run all enabled security checks
        if self.config.is_enabled("pii_detection"):
            pii_threats = self.pii_matcher.detect_threats(content)
            threats.extend(pii_threats)
        
        if self.config.is_enabled("injection_prevention"):
            injection_threats = self.injection_matcher.detect_threats(content)
            threats.extend(injection_threats)
        
        if self.config.is_enabled("data_leak_prevention"):
            leak_threats = self.leak_matcher.detect_threats(content)
            threats.extend(leak_threats)
        
        # Calculate overall risk score
        risk_score = self._calculate_risk_score(threats)
        
        # Determine if content should be blocked
        blocked = self._should_block_content(threats, risk_score)
        
        # Generate sanitized content if needed
        sanitized_content = self._sanitize_content(content, threats) if not blocked else None
        
        # Create result
        result = SecurityFilterResult(
            is_safe=not blocked and risk_score < 0.5,
            threats=threats,
            overall_risk_score=risk_score,
            blocked_content=blocked,
            sanitized_content=sanitized_content,
            explanation=self._generate_explanation(threats, risk_score, blocked)
        )
        
        # Log security event if threats detected
        if threats:
            self._log_security_event(content, result, user_id)
        
        return result
    
    def _calculate_risk_score(self, threats: List[SecurityThreat]) -> float:
        """Calculate overall risk score based on detected threats."""
        if not threats:
            return 0.0
        
        # Weight threats by severity
        severity_weights = {
            SecurityLevel.LOW: 0.1,
            SecurityLevel.MEDIUM: 0.3,
            SecurityLevel.HIGH: 0.7,
            SecurityLevel.CRITICAL: 1.0
        }
        
        total_weight = 0.0
        weighted_score = 0.0
        
        for threat in threats:
            weight = severity_weights.get(threat.severity, 0.5)
            total_weight += weight
            weighted_score += weight * threat.confidence
        
        return min(weighted_score / total_weight if total_weight > 0 else 0.0, 1.0)
    
    def _should_block_content(self, threats: List[SecurityThreat], risk_score: float) -> bool:
        """Determine if content should be blocked based on threats."""
        # Block if any critical threats detected
        critical_threats = [t for t in threats if t.severity == SecurityLevel.CRITICAL]
        if critical_threats:
            return True
        
        # Block if risk score is too high
        if risk_score > 0.8:
            return True
        
        # Block if multiple high-severity threats
        high_threats = [t for t in threats if t.severity == SecurityLevel.HIGH]
        if len(high_threats) >= 2:
            return True
        
        return False
    
    def _sanitize_content(self, content: str, threats: List[SecurityThreat]) -> str:
        """Sanitize content by removing or masking detected threats."""
        sanitized = content
        
        # Sort threats by position (reverse order to maintain positions)
        sorted_threats = sorted(threats, key=lambda t: t.position[0], reverse=True)
        
        for threat in sorted_threats:
            start, end = threat.position
            if threat.severity in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
                # Replace with placeholder
                placeholder = f"[{threat.threat_type.value.upper()}_REMOVED]"
                sanitized = sanitized[:start] + placeholder + sanitized[end:]
            elif threat.severity == SecurityLevel.MEDIUM:
                # Mask partially
                original = sanitized[start:end]
                masked = original[:2] + "*" * (len(original) - 4) + original[-2:] if len(original) > 4 else "*" * len(original)
                sanitized = sanitized[:start] + masked + sanitized[end:]
        
        return sanitized
    
    def _generate_explanation(self, threats: List[SecurityThreat], risk_score: float, blocked: bool) -> str:
        """Generate explanation for security filtering result."""
        if not threats:
            return "Content passed all security checks"
        
        explanation = f"Security analysis (risk score: {risk_score:.2f}):\n"
        
        # Group threats by type
        threat_groups = {}
        for threat in threats:
            threat_type = threat.threat_type.value
            if threat_type not in threat_groups:
                threat_groups[threat_type] = []
            threat_groups[threat_type].append(threat)
        
        for threat_type, threat_list in threat_groups.items():
            explanation += f"- {threat_type.replace('_', ' ').title()}: {len(threat_list)} detected\n"
            for threat in threat_list[:3]:  # Show first 3
                explanation += f"  • {threat.pattern_name} ({threat.severity.value})\n"
            if len(threat_list) > 3:
                explanation += f"  • ... and {len(threat_list) - 3} more\n"
        
        if blocked:
            explanation += "\nContent blocked due to security policy violations."
        
        return explanation
    
    def _log_security_event(self, content: str, result: SecurityFilterResult, user_id: str = None) -> None:
        """Log security event for monitoring and analysis."""
        if not self.config.config.get("monitoring", {}).get("log_security_events", True):
            return
        
        timestamp = datetime.now().isoformat()
        event = {
            "timestamp": timestamp,
            "user_id": user_id,
            "content_length": len(content),
            "content_preview": content[:100] + "..." if len(content) > 100 else content,
            "threats_detected": len(result.threats),
            "risk_score": result.overall_risk_score,
            "blocked": result.blocked_content,
            "threats": [
                {
                    "type": threat.threat_type.value,
                    "severity": threat.severity.value,
                    "pattern": threat.pattern_name,
                    "confidence": threat.confidence
                }
                for threat in result.threats
            ]
        }
        
        # Save to daily log file
        date_str = datetime.now().strftime("%Y%m%d")
        log_file = self.events_dir / f"security_events_{date_str}.jsonl"
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(event) + "\n")


class PIIPatternMatcher:
    """PII detection pattern matcher."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.patterns = self._load_patterns()
    
    def _load_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load PII detection patterns from config."""
        return self.config.get_patterns("pii_detection")
    
    def detect_threats(self, content: str) -> List[SecurityThreat]:
        """Detect PII threats in content."""
        threats = []
        
        for pattern_name, pattern_config in self.patterns.items():
            if not pattern_config.get("enabled", True):
                continue
            
            regex = pattern_config.get("regex", "")
            severity = SecurityLevel(pattern_config.get("severity", "medium"))
            
            matches = re.finditer(regex, content, re.IGNORECASE)
            for match in matches:
                threat = SecurityThreat(
                    threat_type=ThreatType.PII_DETECTION,
                    severity=severity,
                    pattern_name=pattern_name,
                    matched_text=match.group(),
                    confidence=0.9,  # High confidence for regex matches
                    position=(match.start(), match.end()),
                    description=f"PII detected: {pattern_name}"
                )
                threats.append(threat)
        
        return threats


class InjectionPatternMatcher:
    """Injection attack pattern matcher."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.patterns = self._load_patterns()
    
    def _load_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load injection attack patterns from config."""
        return self.config.get_patterns("injection_prevention")
    
    def detect_threats(self, content: str) -> List[SecurityThreat]:
        """Detect injection attack threats in content."""
        threats = []
        
        for pattern_name, pattern_config in self.patterns.items():
            if not pattern_config.get("enabled", True):
                continue
            
            patterns = pattern_config.get("patterns", [])
            severity = SecurityLevel(pattern_config.get("severity", "high"))
            
            for pattern in patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE | re.DOTALL)
                for match in matches:
                    threat = SecurityThreat(
                        threat_type=ThreatType.INJECTION_ATTACK,
                        severity=severity,
                        pattern_name=pattern_name,
                        matched_text=match.group(),
                        confidence=0.8,  # Good confidence for injection patterns
                        position=(match.start(), match.end()),
                        description=f"Injection attack detected: {pattern_name}"
                    )
                    threats.append(threat)
                    break  # Only report first match per pattern type
        
        return threats


class DataLeakPatternMatcher:
    """Data leak detection pattern matcher."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.patterns = self._load_patterns()
    
    def _load_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load data leak detection patterns from config."""
        return self.config.get_patterns("data_leak_prevention")
    
    def detect_threats(self, content: str) -> List[SecurityThreat]:
        """Detect data leak threats in content."""
        threats = []
        
        for pattern_name, pattern_config in self.patterns.items():
            if not pattern_config.get("enabled", True):
                continue
            
            patterns = pattern_config.get("patterns", [])
            severity = SecurityLevel(pattern_config.get("severity", "medium"))
            
            for pattern in patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    threat = SecurityThreat(
                        threat_type=ThreatType.DATA_LEAK,
                        severity=severity,
                        pattern_name=pattern_name,
                        matched_text=match.group(),
                        confidence=0.7,  # Moderate confidence for leak detection
                        position=(match.start(), match.end()),
                        description=f"Data leak detected: {pattern_name}"
                    )
                    threats.append(threat)
        
        return threats


# Integration with existing confidence manager
def integrate_with_confidence_manager():
    """Integration function to update existing confidence manager."""
    try:
        from src.confidence_manager import ContentFilter, FilterResult, FilterType
        
        # Monkey patch the existing ContentFilter to use enhanced security
        original_filter_content = ContentFilter.filter_content
        
        def enhanced_filter_content(self, content: str) -> FilterResult:
            # Use enhanced security filter
            enhanced_filter = EnhancedSecurityFilter()
            security_result = enhanced_filter.filter_content(content)
            
            # Convert to original format for compatibility
            detected_patterns = []
            filter_type = FilterType.SAFE
            confidence = 0.9
            
            if security_result.threats:
                # Determine filter type based on threats
                pii_threats = [t for t in security_result.threats if t.threat_type == ThreatType.PII_DETECTION]
                if pii_threats:
                    filter_type = FilterType.PII_DETECTED
                    detected_patterns.extend([f"pii_{t.pattern_name}" for t in pii_threats])
                
                # Adjust confidence based on risk score
                confidence = max(0.1, 1.0 - security_result.overall_risk_score)
            
            return FilterResult(
                is_safe=security_result.is_safe,
                filter_type=filter_type,
                detected_patterns=detected_patterns,
                confidence=confidence,
                explanation=security_result.explanation
            )
        
        ContentFilter.filter_content = enhanced_filter_content
        
    except ImportError:
        # If confidence manager not available, skip integration
        pass


if __name__ == "__main__":
    # Test the enhanced security filter
    filter = EnhancedSecurityFilter()
    
    test_cases = [
        "Contact me at john@example.com",
        "'; DROP TABLE users; --",
        "My API key is sk-1234567890abcdef",
        "This is a normal message"
    ]
    
    for test in test_cases:
        result = filter.filter_content(test)
        print(f"Input: {test}")
        print(f"Safe: {result.is_safe}")
        print(f"Risk Score: {result.overall_risk_score:.2f}")
        print(f"Threats: {len(result.threats)}")
        print(f"Explanation: {result.explanation}")
        print("-" * 50)