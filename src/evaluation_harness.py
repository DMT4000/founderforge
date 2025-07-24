"""
Evaluation harness for AI quality assessment and testing.

This module provides comprehensive testing capabilities for the FounderForge AI system,
including accuracy measurement, confidence threshold validation, and fallback testing.
"""

import json
import time
import asyncio
import logging
from logging_manager import get_logging_manager, LogLevel, LogCategory
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from datetime import datetime

from src.agents import AgentOrchestrator
from src.context_manager import ContextAssembler
from src.confidence_manager import ConfidenceManager, ConfidenceScore
from src.gemini_client import GeminiClient, MockMode
from src.models import WorkflowResult


class TestScenarioType(Enum):
    """Types of test scenarios."""
    FUNDING_ADVICE = "funding_advice"
    BUSINESS_PLANNING = "business_planning"
    STRATEGY_GUIDANCE = "strategy_guidance"
    OPERATIONAL_SUPPORT = "operational_support"
    GENERAL_QUERY = "general_query"


class TestResult(Enum):
    """Test result outcomes."""
    PASS = "pass"
    FAIL = "fail"
    PARTIAL = "partial"
    ERROR = "error"


@dataclass
class TestScenario:
    """Individual test scenario definition."""
    id: str
    name: str
    scenario_type: TestScenarioType
    query: str
    expected_response_type: str
    expected_keywords: List[str]
    context: Dict[str, Any]
    success_criteria: Dict[str, Any]
    confidence_threshold: float = 0.8


@dataclass
class EvaluationResult:
    """Result of evaluating a single test scenario."""
    scenario_id: str
    test_result: TestResult
    actual_response: str
    confidence_score: float
    execution_time: float
    accuracy_score: float
    keyword_matches: List[str]
    missing_keywords: List[str]
    fallback_used: bool
    error_message: Optional[str] = None
    detailed_metrics: Dict[str, Any] = None


@dataclass
class EvaluationSummary:
    """Summary of evaluation run results."""
    total_scenarios: int
    passed: int
    failed: int
    partial: int
    errors: int
    overall_accuracy: float
    average_confidence: float
    average_execution_time: float
    fallback_usage_rate: float
    timestamp: datetime
    detailed_results: List[EvaluationResult]


class EvaluationHarness:
    """Main evaluation harness for AI quality assessment."""
    
    def __init__(
        self,
        gemini_client: GeminiClient,
        context_manager: ContextAssembler,
        confidence_manager: ConfidenceManager,
        agent_orchestrator: AgentOrchestrator,
        test_data_path: str = "data/evaluation"
    ):
        """Initialize evaluation harness.
        
        Args:
            gemini_client: Gemini API client
            context_manager: Context assembly manager
            confidence_manager: Confidence scoring manager
            agent_orchestrator: Multi-agent orchestrator
            test_data_path: Path to test data directory
        """
        self.gemini_client = gemini_client
        self.context_manager = context_manager
        self.confidence_manager = confidence_manager
        self.agent_orchestrator = agent_orchestrator
        
        self.logger = get_logging_manager().get_logger(__name__.split(".")[-1], LogCategory.SYSTEM)
        self.test_data_path = Path(test_data_path)
        self.test_data_path.mkdir(exist_ok=True)
        
        # Initialize test scenarios
        self.test_scenarios = self._load_test_scenarios()
        
        # Performance targets
        self.target_accuracy = 0.9  # 90% pass rate target
        self.target_confidence = 0.8  # 80% confidence threshold
        self.max_execution_time = 30.0  # 30 seconds max per scenario
    
    def _load_test_scenarios(self) -> List[TestScenario]:
        """Load predefined test scenarios."""
        scenarios_file = self.test_data_path / "test_scenarios.json"
        
        if scenarios_file.exists():
            with open(scenarios_file, 'r') as f:
                scenarios_data = json.load(f)
            return [TestScenario(**scenario) for scenario in scenarios_data]
        else:
            # Create default test scenarios
            default_scenarios = self._create_default_scenarios()
            self._save_test_scenarios(default_scenarios)
            return default_scenarios
    
    def _create_default_scenarios(self) -> List[TestScenario]:
        """Create default test scenarios for evaluation."""
        scenarios = [
            TestScenario(
                id="funding_001",
                name="Basic Funding Advice",
                scenario_type=TestScenarioType.FUNDING_ADVICE,
                query="How should I prepare for a Series A funding round?",
                expected_response_type="structured_advice",
                expected_keywords=["series a", "funding", "preparation", "investors", "pitch deck"],
                context={"business_stage": "growth", "funding_history": "seed"},
                success_criteria={"min_keywords": 3, "response_length": 100}
            ),
            TestScenario(
                id="planning_001",
                name="Business Plan Creation",
                scenario_type=TestScenarioType.BUSINESS_PLANNING,
                query="What should be included in a business plan for a tech startup?",
                expected_response_type="structured_list",
                expected_keywords=["business plan", "executive summary", "market analysis", "financial projections"],
                context={"industry": "technology", "stage": "startup"},
                success_criteria={"min_keywords": 3, "response_length": 150}
            ),
            TestScenario(
                id="strategy_001",
                name="Market Entry Strategy",
                scenario_type=TestScenarioType.STRATEGY_GUIDANCE,
                query="How should I approach entering a competitive market?",
                expected_response_type="strategic_advice",
                expected_keywords=["market entry", "competition", "differentiation", "strategy"],
                context={"market_type": "competitive", "resources": "limited"},
                success_criteria={"min_keywords": 2, "response_length": 120}
            ),
            TestScenario(
                id="operations_001",
                name="Team Building Advice",
                scenario_type=TestScenarioType.OPERATIONAL_SUPPORT,
                query="When should I hire my first employees and what roles should I prioritize?",
                expected_response_type="operational_guidance",
                expected_keywords=["hiring", "employees", "roles", "team", "priorities"],
                context={"company_size": "solo_founder", "stage": "early"},
                success_criteria={"min_keywords": 3, "response_length": 100}
            ),
            TestScenario(
                id="general_001",
                name="Startup Terminology",
                scenario_type=TestScenarioType.GENERAL_QUERY,
                query="What is the difference between angel investors and venture capitalists?",
                expected_response_type="explanation",
                expected_keywords=["angel investors", "venture capitalists", "difference", "funding"],
                context={},
                success_criteria={"min_keywords": 3, "response_length": 80}
            ),
            TestScenario(
                id="funding_002",
                name="Valuation Guidance",
                scenario_type=TestScenarioType.FUNDING_ADVICE,
                query="How do I determine the valuation of my startup for fundraising?",
                expected_response_type="structured_advice",
                expected_keywords=["valuation", "startup", "fundraising", "methods", "metrics"],
                context={"revenue": "pre_revenue", "stage": "seed"},
                success_criteria={"min_keywords": 3, "response_length": 120}
            ),
            TestScenario(
                id="planning_002",
                name="Go-to-Market Strategy",
                scenario_type=TestScenarioType.BUSINESS_PLANNING,
                query="Help me create a go-to-market strategy for my SaaS product",
                expected_response_type="strategic_plan",
                expected_keywords=["go-to-market", "saas", "strategy", "customers", "channels"],
                context={"product_type": "saas", "target_market": "b2b"},
                success_criteria={"min_keywords": 3, "response_length": 150}
            ),
            TestScenario(
                id="strategy_002",
                name="Product-Market Fit",
                scenario_type=TestScenarioType.STRATEGY_GUIDANCE,
                query="How do I know if I've achieved product-market fit?",
                expected_response_type="diagnostic_guidance",
                expected_keywords=["product-market fit", "metrics", "indicators", "validation"],
                context={"product_stage": "mvp", "user_feedback": "mixed"},
                success_criteria={"min_keywords": 2, "response_length": 100}
            ),
            TestScenario(
                id="operations_002",
                name="Scaling Operations",
                scenario_type=TestScenarioType.OPERATIONAL_SUPPORT,
                query="What operational challenges should I prepare for when scaling from 10 to 50 employees?",
                expected_response_type="operational_guidance",
                expected_keywords=["scaling", "operations", "employees", "challenges", "growth"],
                context={"current_size": "10", "target_size": "50"},
                success_criteria={"min_keywords": 3, "response_length": 130}
            ),
            TestScenario(
                id="general_002",
                name="Equity Distribution",
                scenario_type=TestScenarioType.GENERAL_QUERY,
                query="How should I distribute equity among co-founders?",
                expected_response_type="guidance",
                expected_keywords=["equity", "co-founders", "distribution", "factors", "vesting"],
                context={"founders": "2", "stage": "early"},
                success_criteria={"min_keywords": 3, "response_length": 100}
            )
        ]
        return scenarios
    
    def _save_test_scenarios(self, scenarios: List[TestScenario]) -> None:
        """Save test scenarios to file."""
        scenarios_file = self.test_data_path / "test_scenarios.json"
        scenarios_data = [asdict(scenario) for scenario in scenarios]
        
        # Convert enums to strings for JSON serialization
        for scenario_data in scenarios_data:
            scenario_data['scenario_type'] = scenario_data['scenario_type'].value
        
        with open(scenarios_file, 'w') as f:
            json.dump(scenarios_data, f, indent=2)
    
    async def run_evaluation(
        self,
        scenario_ids: Optional[List[str]] = None,
        mock_mode: bool = False
    ) -> EvaluationSummary:
        """Run comprehensive evaluation on test scenarios.
        
        Args:
            scenario_ids: Specific scenario IDs to test (None for all)
            mock_mode: Use mock responses for testing
            
        Returns:
            EvaluationSummary with detailed results
        """
        self.logger.info("Starting evaluation run")
        start_time = time.time()
        
        # Filter scenarios if specific IDs provided
        scenarios_to_test = self.test_scenarios
        if scenario_ids:
            scenarios_to_test = [s for s in self.test_scenarios if s.id in scenario_ids]
        
        # Set mock mode if requested
        if mock_mode:
            self.gemini_client.mock_mode = MockMode.SUCCESS
        
        # Run evaluation on each scenario
        results = []
        for scenario in scenarios_to_test:
            try:
                result = await self._evaluate_scenario(scenario)
                results.append(result)
                self.logger.info(f"Scenario {scenario.id}: {result.test_result.value}")
            except Exception as e:
                self.logger.error(f"Error evaluating scenario {scenario.id}: {e}")
                results.append(EvaluationResult(
                    scenario_id=scenario.id,
                    test_result=TestResult.ERROR,
                    actual_response="",
                    confidence_score=0.0,
                    execution_time=0.0,
                    accuracy_score=0.0,
                    keyword_matches=[],
                    missing_keywords=scenario.expected_keywords,
                    fallback_used=False,
                    error_message=str(e)
                ))
        
        # Calculate summary statistics
        summary = self._calculate_summary(results, time.time() - start_time)
        
        # Save results
        self._save_evaluation_results(summary)
        
        self.logger.info(f"Evaluation completed: {summary.passed}/{summary.total_scenarios} passed")
        return summary
    
    async def _evaluate_scenario(self, scenario: TestScenario) -> EvaluationResult:
        """Evaluate a single test scenario.
        
        Args:
            scenario: Test scenario to evaluate
            
        Returns:
            EvaluationResult with detailed metrics
        """
        start_time = time.time()
        
        try:
            # Execute the scenario through the agent orchestrator
            workflow_result = await self.agent_orchestrator.execute_workflow(
                workflow_type="general",
                user_id=f"test_user_{scenario.id}",
                task_data={"query": scenario.query, "context": scenario.context},
                user_context=scenario.context
            )
            
            execution_time = time.time() - start_time
            
            # Extract response content
            actual_response = self._extract_response_content(workflow_result)
            
            # Calculate confidence score
            confidence_score = self._calculate_scenario_confidence(workflow_result, scenario)
            
            # Check if fallback was used
            fallback_used = self._check_fallback_usage(workflow_result)
            
            # Evaluate accuracy
            accuracy_metrics = self._evaluate_accuracy(scenario, actual_response)
            
            # Determine overall test result
            test_result = self._determine_test_result(
                scenario, accuracy_metrics, confidence_score, execution_time, fallback_used
            )
            
            return EvaluationResult(
                scenario_id=scenario.id,
                test_result=test_result,
                actual_response=actual_response,
                confidence_score=confidence_score,
                execution_time=execution_time,
                accuracy_score=accuracy_metrics['accuracy_score'],
                keyword_matches=accuracy_metrics['keyword_matches'],
                missing_keywords=accuracy_metrics['missing_keywords'],
                fallback_used=fallback_used,
                detailed_metrics=accuracy_metrics
            )
            
        except Exception as e:
            return EvaluationResult(
                scenario_id=scenario.id,
                test_result=TestResult.ERROR,
                actual_response="",
                confidence_score=0.0,
                execution_time=time.time() - start_time,
                accuracy_score=0.0,
                keyword_matches=[],
                missing_keywords=scenario.expected_keywords,
                fallback_used=False,
                error_message=str(e)
            )
    
    def _extract_response_content(self, workflow_result: WorkflowResult) -> str:
        """Extract response content from workflow result."""
        if not workflow_result.success:
            return "Workflow execution failed"
        
        # Try to extract from different agent outputs
        result_data = workflow_result.result_data
        
        if 'coach_output' in result_data:
            return result_data['coach_output'].get('message', '')
        elif 'planner_output' in result_data:
            planner_output = result_data['planner_output']
            if isinstance(planner_output, dict) and 'executive_summary' in planner_output:
                return planner_output['executive_summary']
        elif 'orchestrator_output' in result_data:
            return str(result_data['orchestrator_output'])
        
        return str(result_data)
    
    def _calculate_scenario_confidence(
        self, 
        workflow_result: WorkflowResult, 
        scenario: TestScenario
    ) -> float:
        """Calculate confidence score for scenario result."""
        if not workflow_result.success:
            return workflow_result.confidence_score  # Return actual confidence even if failed
        
        return workflow_result.confidence_score
    
    def _check_fallback_usage(self, workflow_result: WorkflowResult) -> bool:
        """Check if fallback mechanisms were used."""
        # Check agent logs for fallback indicators
        for log in workflow_result.agent_logs:
            # Check action for fallback indicators
            if 'fallback' in log.action.lower():
                return True
            # Check output data for fallback messages
            if isinstance(log.output_data, dict):
                message = log.output_data.get('message', '')
                if isinstance(message, str) and ('fallback' in message.lower() or 'low confidence' in message.lower()):
                    return True
        
        # Check if confidence score is below threshold
        return workflow_result.confidence_score < self.target_confidence
    
    def _evaluate_accuracy(self, scenario: TestScenario, response: str) -> Dict[str, Any]:
        """Evaluate accuracy of response against expected criteria.
        
        Args:
            scenario: Test scenario with expected criteria
            response: Actual response to evaluate
            
        Returns:
            Dictionary with accuracy metrics
        """
        response_lower = response.lower()
        
        # Check keyword matches
        keyword_matches = []
        missing_keywords = []
        
        for keyword in scenario.expected_keywords:
            if keyword.lower() in response_lower:
                keyword_matches.append(keyword)
            else:
                missing_keywords.append(keyword)
        
        # Calculate keyword accuracy
        keyword_accuracy = len(keyword_matches) / len(scenario.expected_keywords) if scenario.expected_keywords else 1.0
        
        # Check response length criteria
        min_length = scenario.success_criteria.get('response_length', 50)
        length_score = 1.0 if len(response) >= min_length else len(response) / min_length
        
        # Check minimum keyword requirement
        min_keywords = scenario.success_criteria.get('min_keywords', 1)
        keyword_requirement_met = len(keyword_matches) >= min_keywords
        
        # Calculate overall accuracy score
        accuracy_score = (keyword_accuracy * 0.7) + (length_score * 0.3)
        
        return {
            'accuracy_score': accuracy_score,
            'keyword_matches': keyword_matches,
            'missing_keywords': missing_keywords,
            'keyword_accuracy': keyword_accuracy,
            'length_score': length_score,
            'keyword_requirement_met': keyword_requirement_met,
            'response_length': len(response)
        }
    
    def _determine_test_result(
        self,
        scenario: TestScenario,
        accuracy_metrics: Dict[str, Any],
        confidence_score: float,
        execution_time: float,
        fallback_used: bool
    ) -> TestResult:
        """Determine overall test result based on all metrics."""
        # Check for hard failures
        if execution_time > self.max_execution_time:
            return TestResult.FAIL
        
        if confidence_score < scenario.confidence_threshold:
            return TestResult.FAIL
        
        # Check accuracy requirements
        accuracy_score = accuracy_metrics['accuracy_score']
        keyword_requirement_met = accuracy_metrics['keyword_requirement_met']
        
        if accuracy_score >= 0.8 and keyword_requirement_met:
            return TestResult.PASS
        elif accuracy_score >= 0.6 or keyword_requirement_met:
            return TestResult.PARTIAL
        else:
            return TestResult.FAIL
    
    def _calculate_summary(self, results: List[EvaluationResult], total_time: float) -> EvaluationSummary:
        """Calculate summary statistics from evaluation results."""
        total_scenarios = len(results)
        passed = sum(1 for r in results if r.test_result == TestResult.PASS)
        failed = sum(1 for r in results if r.test_result == TestResult.FAIL)
        partial = sum(1 for r in results if r.test_result == TestResult.PARTIAL)
        errors = sum(1 for r in results if r.test_result == TestResult.ERROR)
        
        # Calculate averages (excluding errors)
        valid_results = [r for r in results if r.test_result != TestResult.ERROR]
        
        if valid_results:
            overall_accuracy = sum(r.accuracy_score for r in valid_results) / len(valid_results)
            average_confidence = sum(r.confidence_score for r in valid_results) / len(valid_results)
            average_execution_time = sum(r.execution_time for r in valid_results) / len(valid_results)
            fallback_usage_rate = sum(1 for r in valid_results if r.fallback_used) / len(valid_results)
        else:
            overall_accuracy = 0.0
            average_confidence = 0.0
            average_execution_time = 0.0
            fallback_usage_rate = 0.0
        
        return EvaluationSummary(
            total_scenarios=total_scenarios,
            passed=passed,
            failed=failed,
            partial=partial,
            errors=errors,
            overall_accuracy=overall_accuracy,
            average_confidence=average_confidence,
            average_execution_time=average_execution_time,
            fallback_usage_rate=fallback_usage_rate,
            timestamp=datetime.now(),
            detailed_results=results
        )
    
    def _save_evaluation_results(self, summary: EvaluationSummary) -> None:
        """Save evaluation results to file."""
        timestamp_str = summary.timestamp.strftime("%Y%m%d_%H%M%S")
        results_file = self.test_data_path / f"evaluation_results_{timestamp_str}.json"
        
        # Convert summary to dict for JSON serialization
        summary_dict = asdict(summary)
        summary_dict['timestamp'] = summary.timestamp.isoformat()
        
        # Convert enums to strings
        for result in summary_dict['detailed_results']:
            result['test_result'] = result['test_result'].value
        
        with open(results_file, 'w') as f:
            json.dump(summary_dict, f, indent=2)
        
        self.logger.info(f"Evaluation results saved to {results_file}")
    
    def generate_report(self, summary: EvaluationSummary) -> str:
        """Generate human-readable evaluation report.
        
        Args:
            summary: Evaluation summary to report on
            
        Returns:
            Formatted report string
        """
        report = f"""
# FounderForge AI Evaluation Report
Generated: {summary.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- **Total Scenarios**: {summary.total_scenarios}
- **Passed**: {summary.passed} ({summary.passed/summary.total_scenarios*100:.1f}%)
- **Failed**: {summary.failed} ({summary.failed/summary.total_scenarios*100:.1f}%)
- **Partial**: {summary.partial} ({summary.partial/summary.total_scenarios*100:.1f}%)
- **Errors**: {summary.errors} ({summary.errors/summary.total_scenarios*100:.1f}%)

## Performance Metrics
- **Overall Accuracy**: {summary.overall_accuracy:.3f} (Target: {self.target_accuracy:.3f})
- **Average Confidence**: {summary.average_confidence:.3f} (Target: {self.target_confidence:.3f})
- **Average Execution Time**: {summary.average_execution_time:.2f}s (Max: {self.max_execution_time}s)
- **Fallback Usage Rate**: {summary.fallback_usage_rate:.3f}

## Target Achievement
- **Accuracy Target**: {'✓ ACHIEVED' if summary.overall_accuracy >= self.target_accuracy else '✗ MISSED'}
- **Confidence Target**: {'✓ ACHIEVED' if summary.average_confidence >= self.target_confidence else '✗ MISSED'}

## Detailed Results
"""
        
        for result in summary.detailed_results:
            status_icon = {
                TestResult.PASS: "✓",
                TestResult.FAIL: "✗",
                TestResult.PARTIAL: "~",
                TestResult.ERROR: "!"
            }[result.test_result]
            
            report += f"""
### {status_icon} {result.scenario_id}
- **Result**: {result.test_result.value.upper()}
- **Accuracy**: {result.accuracy_score:.3f}
- **Confidence**: {result.confidence_score:.3f}
- **Execution Time**: {result.execution_time:.2f}s
- **Fallback Used**: {'Yes' if result.fallback_used else 'No'}
- **Keywords Found**: {len(result.keyword_matches)}/{len(result.keyword_matches) + len(result.missing_keywords)}
"""
            
            if result.error_message:
                report += f"- **Error**: {result.error_message}\n"
        
        return report
    
    async def run_confidence_threshold_validation(self) -> Dict[str, Any]:
        """Run validation tests for confidence threshold effectiveness.
        
        Returns:
            Dictionary with validation results
        """
        self.logger.info("Running confidence threshold validation")
        
        # Test different confidence thresholds
        thresholds_to_test = [0.6, 0.7, 0.8, 0.9]
        results = {}
        
        original_threshold = self.confidence_manager.confidence_threshold
        
        try:
            for threshold in thresholds_to_test:
                self.confidence_manager.confidence_threshold = threshold
                
                # Run evaluation with current threshold
                summary = await self.run_evaluation(mock_mode=True)
                
                results[threshold] = {
                    'accuracy': summary.overall_accuracy,
                    'passed': summary.passed,
                    'fallback_rate': summary.fallback_usage_rate,
                    'avg_confidence': summary.average_confidence
                }
        
        finally:
            # Restore original threshold
            self.confidence_manager.confidence_threshold = original_threshold
        
        # Find optimal threshold
        optimal_threshold = max(
            results.keys(),
            key=lambda t: results[t]['accuracy'] * (1 - results[t]['fallback_rate'])
        )
        
        validation_result = {
            'threshold_results': results,
            'optimal_threshold': optimal_threshold,
            'current_threshold': original_threshold,
            'recommendation': self._generate_threshold_recommendation(results, optimal_threshold)
        }
        
        self.logger.info(f"Confidence threshold validation completed. Optimal: {optimal_threshold}")
        return validation_result
    
    def _generate_threshold_recommendation(
        self, 
        results: Dict[float, Dict[str, Any]], 
        optimal_threshold: float
    ) -> str:
        """Generate recommendation for confidence threshold."""
        current_results = results[self.confidence_manager.confidence_threshold]
        optimal_results = results[optimal_threshold]
        
        if optimal_threshold == self.confidence_manager.confidence_threshold:
            return "Current confidence threshold is optimal."
        
        accuracy_improvement = optimal_results['accuracy'] - current_results['accuracy']
        fallback_change = optimal_results['fallback_rate'] - current_results['fallback_rate']
        
        recommendation = f"Consider adjusting confidence threshold to {optimal_threshold}. "
        recommendation += f"This would improve accuracy by {accuracy_improvement:.3f} "
        
        if fallback_change > 0:
            recommendation += f"but increase fallback usage by {fallback_change:.3f}."
        else:
            recommendation += f"and reduce fallback usage by {abs(fallback_change):.3f}."
        
        return recommendation
    
    async def test_fallback_mechanisms(self) -> Dict[str, Any]:
        """Test effectiveness of fallback mechanisms.
        
        Returns:
            Dictionary with fallback test results
        """
        self.logger.info("Testing fallback mechanisms")
        
        # Create scenarios designed to trigger fallbacks
        fallback_scenarios = [
            {
                'query': 'What is the meaning of life?',  # Off-topic
                'expected_fallback': True,
                'reason': 'off_topic'
            },
            {
                'query': 'Tell me about my personal information at john@example.com',  # PII
                'expected_fallback': True,
                'reason': 'pii_detected'
            },
            {
                'query': 'How to start a business?',  # Valid but may trigger low confidence
                'expected_fallback': False,
                'reason': 'valid_query'
            }
        ]
        
        results = []
        
        for scenario in fallback_scenarios:
            try:
                # Execute query
                workflow_result = await self.agent_orchestrator.execute_workflow(
                    workflow_type="general",
                    user_id="fallback_test_user",
                    task_data={"query": scenario['query']},
                    user_context={}
                )
                
                # Check if fallback was used
                fallback_used = self._check_fallback_usage(workflow_result)
                
                # Evaluate result
                test_passed = fallback_used == scenario['expected_fallback']
                
                results.append({
                    'query': scenario['query'],
                    'expected_fallback': scenario['expected_fallback'],
                    'actual_fallback': fallback_used,
                    'test_passed': test_passed,
                    'reason': scenario['reason'],
                    'confidence': workflow_result.confidence_score
                })
                
            except Exception as e:
                results.append({
                    'query': scenario['query'],
                    'expected_fallback': scenario['expected_fallback'],
                    'actual_fallback': True,  # Error should trigger fallback
                    'test_passed': scenario['expected_fallback'],
                    'reason': 'error',
                    'error': str(e)
                })
        
        # Calculate summary
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r['test_passed'])
        
        fallback_test_summary = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'detailed_results': results
        }
        
        self.logger.info(f"Fallback mechanism testing completed: {passed_tests}/{total_tests} passed")
        return fallback_test_summary