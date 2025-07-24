"""
Tests for the evaluation harness and AI quality assessment system.
"""

import pytest
import asyncio
import json
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
from pathlib import Path

from src.evaluation_harness import (
    EvaluationHarness, TestScenario, TestScenarioType, TestResult,
    EvaluationResult, EvaluationSummary
)
from src.agents import AgentOrchestrator
from src.context_manager import ContextAssembler
from src.confidence_manager import ConfidenceManager
from src.gemini_client import GeminiClient, MockMode
from src.models import WorkflowResult, AgentLog


@pytest.fixture
def mock_gemini_client():
    """Create a mock Gemini client for testing."""
    client = GeminiClient(api_key="test_key", mock_mode=MockMode.SUCCESS)
    return client


@pytest.fixture
def mock_context_manager():
    """Create a mock context manager for testing."""
    return Mock(spec=ContextAssembler)


@pytest.fixture
def mock_confidence_manager():
    """Create a mock confidence manager for testing."""
    manager = Mock(spec=ConfidenceManager)
    manager.confidence_threshold = 0.8
    return manager


@pytest.fixture
def mock_agent_orchestrator():
    """Create a mock agent orchestrator for testing."""
    return Mock(spec=AgentOrchestrator)


@pytest.fixture
def temp_test_data_path():
    """Create temporary test data directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def sample_test_scenario():
    """Create a sample test scenario."""
    return TestScenario(
        id="test_001",
        name="Sample Test",
        scenario_type=TestScenarioType.FUNDING_ADVICE,
        query="How should I prepare for funding?",
        expected_response_type="advice",
        expected_keywords=["funding", "preparation", "investors"],
        context={"stage": "seed"},
        success_criteria={"min_keywords": 2, "response_length": 50}
    )


@pytest.fixture
def sample_workflow_result():
    """Create a sample workflow result."""
    return WorkflowResult(
        success=True,
        result_data={
            "coach_output": {
                "message": "Great question about funding preparation! You should focus on creating a solid pitch deck, financial projections, and building relationships with potential investors.",
                "confidence": 0.85
            }
        },
        execution_time=2.5,
        agent_logs=[
            AgentLog(
                agent_name="coach",
                action="process_query",
                input_data={"query": "funding advice"},
                output_data={"message": "advice provided"},
                execution_time=2.5,
                timestamp=datetime.now()
            )
        ],
        confidence_score=0.85
    )


class TestEvaluationHarness:
    """Test cases for EvaluationHarness."""
    
    def test_initialization(
        self,
        mock_gemini_client,
        mock_context_manager,
        mock_confidence_manager,
        mock_agent_orchestrator,
        temp_test_data_path
    ):
        """Test evaluation harness initialization."""
        harness = EvaluationHarness(
            mock_gemini_client,
            mock_context_manager,
            mock_confidence_manager,
            mock_agent_orchestrator,
            temp_test_data_path
        )
        
        assert harness.gemini_client == mock_gemini_client
        assert harness.context_manager == mock_context_manager
        assert harness.confidence_manager == mock_confidence_manager
        assert harness.agent_orchestrator == mock_agent_orchestrator
        assert harness.target_accuracy == 0.9
        assert harness.target_confidence == 0.8
        assert len(harness.test_scenarios) > 0
    
    def test_default_scenarios_creation(
        self,
        mock_gemini_client,
        mock_context_manager,
        mock_confidence_manager,
        mock_agent_orchestrator,
        temp_test_data_path
    ):
        """Test creation of default test scenarios."""
        harness = EvaluationHarness(
            mock_gemini_client,
            mock_context_manager,
            mock_confidence_manager,
            mock_agent_orchestrator,
            temp_test_data_path
        )
        
        # Check that scenarios file was created
        scenarios_file = Path(temp_test_data_path) / "test_scenarios.json"
        assert scenarios_file.exists()
        
        # Check scenario content
        scenarios = harness.test_scenarios
        assert len(scenarios) >= 10  # Should have at least 10 default scenarios
        
        # Check scenario types are covered
        scenario_types = {s.scenario_type for s in scenarios}
        assert TestScenarioType.FUNDING_ADVICE in scenario_types
        assert TestScenarioType.BUSINESS_PLANNING in scenario_types
        assert TestScenarioType.STRATEGY_GUIDANCE in scenario_types
        assert TestScenarioType.OPERATIONAL_SUPPORT in scenario_types
        assert TestScenarioType.GENERAL_QUERY in scenario_types
    
    def test_load_existing_scenarios(
        self,
        mock_gemini_client,
        mock_context_manager,
        mock_confidence_manager,
        mock_agent_orchestrator,
        temp_test_data_path
    ):
        """Test loading existing test scenarios from file."""
        # Create test scenarios file
        scenarios_data = [
            {
                "id": "custom_001",
                "name": "Custom Test",
                "scenario_type": "funding_advice",
                "query": "Custom query",
                "expected_response_type": "advice",
                "expected_keywords": ["custom", "test"],
                "context": {},
                "success_criteria": {"min_keywords": 1, "response_length": 20},
                "confidence_threshold": 0.8
            }
        ]
        
        scenarios_file = Path(temp_test_data_path) / "test_scenarios.json"
        with open(scenarios_file, 'w') as f:
            json.dump(scenarios_data, f)
        
        # Initialize harness
        harness = EvaluationHarness(
            mock_gemini_client,
            mock_context_manager,
            mock_confidence_manager,
            mock_agent_orchestrator,
            temp_test_data_path
        )
        
        # Check that custom scenario was loaded
        assert len(harness.test_scenarios) == 1
        assert harness.test_scenarios[0].id == "custom_001"
        assert harness.test_scenarios[0].name == "Custom Test"
    
    @pytest.mark.asyncio
    async def test_evaluate_scenario_success(
        self,
        mock_gemini_client,
        mock_context_manager,
        mock_confidence_manager,
        mock_agent_orchestrator,
        temp_test_data_path,
        sample_test_scenario,
        sample_workflow_result
    ):
        """Test successful scenario evaluation."""
        harness = EvaluationHarness(
            mock_gemini_client,
            mock_context_manager,
            mock_confidence_manager,
            mock_agent_orchestrator,
            temp_test_data_path
        )
        
        # Mock agent orchestrator response
        mock_agent_orchestrator.execute_workflow = AsyncMock(return_value=sample_workflow_result)
        
        # Evaluate scenario
        result = await harness._evaluate_scenario(sample_test_scenario)
        
        # Check result
        assert isinstance(result, EvaluationResult)
        assert result.scenario_id == "test_001"
        assert result.test_result in [TestResult.PASS, TestResult.PARTIAL]
        assert result.confidence_score == 0.85
        assert result.execution_time > 0
        assert len(result.keyword_matches) > 0
        assert not result.fallback_used
    
    @pytest.mark.asyncio
    async def test_evaluate_scenario_failure(
        self,
        mock_gemini_client,
        mock_context_manager,
        mock_confidence_manager,
        mock_agent_orchestrator,
        temp_test_data_path,
        sample_test_scenario
    ):
        """Test scenario evaluation with failure."""
        harness = EvaluationHarness(
            mock_gemini_client,
            mock_context_manager,
            mock_confidence_manager,
            mock_agent_orchestrator,
            temp_test_data_path
        )
        
        # Mock failed workflow result
        failed_result = WorkflowResult(
            success=False,
            result_data={"error": "Workflow failed"},
            execution_time=1.0,
            agent_logs=[],
            confidence_score=0.2
        )
        
        mock_agent_orchestrator.execute_workflow = AsyncMock(return_value=failed_result)
        
        # Evaluate scenario
        result = await harness._evaluate_scenario(sample_test_scenario)
        
        # Check result
        assert result.test_result == TestResult.FAIL
        assert result.confidence_score == 0.2
        assert result.accuracy_score < 0.5  # Should be low but not necessarily 0
    
    @pytest.mark.asyncio
    async def test_evaluate_scenario_error(
        self,
        mock_gemini_client,
        mock_context_manager,
        mock_confidence_manager,
        mock_agent_orchestrator,
        temp_test_data_path,
        sample_test_scenario
    ):
        """Test scenario evaluation with exception."""
        harness = EvaluationHarness(
            mock_gemini_client,
            mock_context_manager,
            mock_confidence_manager,
            mock_agent_orchestrator,
            temp_test_data_path
        )
        
        # Mock exception
        mock_agent_orchestrator.execute_workflow = AsyncMock(side_effect=Exception("Test error"))
        
        # Evaluate scenario
        result = await harness._evaluate_scenario(sample_test_scenario)
        
        # Check result
        assert result.test_result == TestResult.ERROR
        assert result.error_message == "Test error"
        assert result.confidence_score == 0.0
    
    def test_evaluate_accuracy(
        self,
        mock_gemini_client,
        mock_context_manager,
        mock_confidence_manager,
        mock_agent_orchestrator,
        temp_test_data_path,
        sample_test_scenario
    ):
        """Test accuracy evaluation logic."""
        harness = EvaluationHarness(
            mock_gemini_client,
            mock_context_manager,
            mock_confidence_manager,
            mock_agent_orchestrator,
            temp_test_data_path
        )
        
        # Test response with all keywords
        response = "For funding preparation, you need to research investors and create a solid plan."
        metrics = harness._evaluate_accuracy(sample_test_scenario, response)
        
        assert metrics['keyword_matches'] == ["funding", "preparation", "investors"]
        assert metrics['missing_keywords'] == []
        assert metrics['keyword_accuracy'] == 1.0
        assert metrics['keyword_requirement_met'] is True
        assert metrics['accuracy_score'] > 0.8
    
    def test_evaluate_accuracy_partial(
        self,
        mock_gemini_client,
        mock_context_manager,
        mock_confidence_manager,
        mock_agent_orchestrator,
        temp_test_data_path,
        sample_test_scenario
    ):
        """Test accuracy evaluation with partial keyword matches."""
        harness = EvaluationHarness(
            mock_gemini_client,
            mock_context_manager,
            mock_confidence_manager,
            mock_agent_orchestrator,
            temp_test_data_path
        )
        
        # Test response with some keywords missing
        response = "You should focus on funding and preparation for the process."
        metrics = harness._evaluate_accuracy(sample_test_scenario, response)
        
        assert "funding" in metrics['keyword_matches']
        assert "preparation" in metrics['keyword_matches']
        assert "investors" in metrics['missing_keywords']
        assert metrics['keyword_accuracy'] < 1.0
        assert metrics['keyword_requirement_met'] is True  # Has at least 2 keywords
    
    def test_determine_test_result(
        self,
        mock_gemini_client,
        mock_context_manager,
        mock_confidence_manager,
        mock_agent_orchestrator,
        temp_test_data_path,
        sample_test_scenario
    ):
        """Test test result determination logic."""
        harness = EvaluationHarness(
            mock_gemini_client,
            mock_context_manager,
            mock_confidence_manager,
            mock_agent_orchestrator,
            temp_test_data_path
        )
        
        # Test passing scenario
        accuracy_metrics = {
            'accuracy_score': 0.9,
            'keyword_requirement_met': True
        }
        result = harness._determine_test_result(
            sample_test_scenario, accuracy_metrics, 0.85, 2.0, False
        )
        assert result == TestResult.PASS
        
        # Test failing scenario (low confidence)
        result = harness._determine_test_result(
            sample_test_scenario, accuracy_metrics, 0.5, 2.0, False
        )
        assert result == TestResult.FAIL
        
        # Test failing scenario (timeout)
        result = harness._determine_test_result(
            sample_test_scenario, accuracy_metrics, 0.85, 35.0, False
        )
        assert result == TestResult.FAIL
        
        # Test partial scenario
        partial_metrics = {
            'accuracy_score': 0.7,
            'keyword_requirement_met': True
        }
        result = harness._determine_test_result(
            sample_test_scenario, partial_metrics, 0.85, 2.0, False
        )
        assert result == TestResult.PARTIAL
    
    @pytest.mark.asyncio
    async def test_run_evaluation_full(
        self,
        mock_gemini_client,
        mock_context_manager,
        mock_confidence_manager,
        mock_agent_orchestrator,
        temp_test_data_path,
        sample_workflow_result
    ):
        """Test full evaluation run."""
        harness = EvaluationHarness(
            mock_gemini_client,
            mock_context_manager,
            mock_confidence_manager,
            mock_agent_orchestrator,
            temp_test_data_path
        )
        
        # Mock successful responses for all scenarios
        mock_agent_orchestrator.execute_workflow = AsyncMock(return_value=sample_workflow_result)
        
        # Run evaluation on first 3 scenarios only
        scenario_ids = [s.id for s in harness.test_scenarios[:3]]
        summary = await harness.run_evaluation(scenario_ids=scenario_ids, mock_mode=True)
        
        # Check summary
        assert isinstance(summary, EvaluationSummary)
        assert summary.total_scenarios == 3
        assert summary.passed + summary.failed + summary.partial + summary.errors == 3
        assert summary.overall_accuracy >= 0.0
        assert summary.average_confidence >= 0.0
        assert len(summary.detailed_results) == 3
        
        # Check that results file was saved
        results_files = list(Path(temp_test_data_path).glob("evaluation_results_*.json"))
        assert len(results_files) > 0
    
    def test_generate_report(
        self,
        mock_gemini_client,
        mock_context_manager,
        mock_confidence_manager,
        mock_agent_orchestrator,
        temp_test_data_path
    ):
        """Test evaluation report generation."""
        harness = EvaluationHarness(
            mock_gemini_client,
            mock_context_manager,
            mock_confidence_manager,
            mock_agent_orchestrator,
            temp_test_data_path
        )
        
        # Create sample summary
        sample_results = [
            EvaluationResult(
                scenario_id="test_001",
                test_result=TestResult.PASS,
                actual_response="Good response",
                confidence_score=0.9,
                execution_time=2.0,
                accuracy_score=0.85,
                keyword_matches=["keyword1", "keyword2"],
                missing_keywords=[],
                fallback_used=False
            ),
            EvaluationResult(
                scenario_id="test_002",
                test_result=TestResult.FAIL,
                actual_response="Poor response",
                confidence_score=0.5,
                execution_time=1.5,
                accuracy_score=0.3,
                keyword_matches=["keyword1"],
                missing_keywords=["keyword2", "keyword3"],
                fallback_used=True
            )
        ]
        
        summary = EvaluationSummary(
            total_scenarios=2,
            passed=1,
            failed=1,
            partial=0,
            errors=0,
            overall_accuracy=0.575,
            average_confidence=0.7,
            average_execution_time=1.75,
            fallback_usage_rate=0.5,
            timestamp=datetime.now(),
            detailed_results=sample_results
        )
        
        # Generate report
        report = harness.generate_report(summary)
        
        # Check report content
        assert "FounderForge AI Evaluation Report" in report
        assert "**Total Scenarios**: 2" in report
        assert "**Passed**: 1 (50.0%)" in report
        assert "**Failed**: 1 (50.0%)" in report
        assert "**Overall Accuracy**: 0.575" in report
        assert "✓ test_001" in report
        assert "✗ test_002" in report
    
    @pytest.mark.asyncio
    async def test_confidence_threshold_validation(
        self,
        mock_gemini_client,
        mock_context_manager,
        mock_confidence_manager,
        mock_agent_orchestrator,
        temp_test_data_path,
        sample_workflow_result
    ):
        """Test confidence threshold validation."""
        harness = EvaluationHarness(
            mock_gemini_client,
            mock_context_manager,
            mock_confidence_manager,
            mock_agent_orchestrator,
            temp_test_data_path
        )
        
        # Mock successful responses
        mock_agent_orchestrator.execute_workflow = AsyncMock(return_value=sample_workflow_result)
        
        # Run confidence threshold validation
        validation_result = await harness.run_confidence_threshold_validation()
        
        # Check validation result
        assert 'threshold_results' in validation_result
        assert 'optimal_threshold' in validation_result
        assert 'current_threshold' in validation_result
        assert 'recommendation' in validation_result
        
        # Check that multiple thresholds were tested
        assert len(validation_result['threshold_results']) >= 4
        
        # Check that original threshold was restored
        assert mock_confidence_manager.confidence_threshold == 0.8
    
    @pytest.mark.asyncio
    async def test_fallback_mechanisms_testing(
        self,
        mock_gemini_client,
        mock_context_manager,
        mock_confidence_manager,
        mock_agent_orchestrator,
        temp_test_data_path
    ):
        """Test fallback mechanisms testing."""
        harness = EvaluationHarness(
            mock_gemini_client,
            mock_context_manager,
            mock_confidence_manager,
            mock_agent_orchestrator,
            temp_test_data_path
        )
        
        # Mock different responses for different queries
        def mock_execute_workflow(workflow_type, user_id, task_data, user_context):
            query = task_data.get('query', '')
            
            if 'meaning of life' in query.lower():
                # Should trigger fallback (off-topic)
                return WorkflowResult(
                    success=True,
                    result_data={"fallback": "This is off-topic"},
                    execution_time=1.0,
                    agent_logs=[AgentLog(
                        agent_name="orchestrator",
                        action="low_confidence_fallback",
                        input_data={},
                        output_data={"message": "Low confidence fallback triggered"},
                        execution_time=1.0,
                        timestamp=datetime.now()
                    )],
                    confidence_score=0.3
                )
            elif '@' in query:
                # Should trigger fallback (PII detected)
                return WorkflowResult(
                    success=True,
                    result_data={"error": "PII detected"},
                    execution_time=0.5,
                    agent_logs=[],
                    confidence_score=0.2
                )
            else:
                # Valid business query
                return WorkflowResult(
                    success=True,
                    result_data={"coach_output": {"message": "Good business advice"}},
                    execution_time=2.0,
                    agent_logs=[],
                    confidence_score=0.9
                )
        
        mock_agent_orchestrator.execute_workflow = AsyncMock(side_effect=mock_execute_workflow)
        
        # Run fallback testing
        fallback_results = await harness.test_fallback_mechanisms()
        
        # Check results
        assert 'total_tests' in fallback_results
        assert 'passed_tests' in fallback_results
        assert 'success_rate' in fallback_results
        assert 'detailed_results' in fallback_results
        
        assert fallback_results['total_tests'] == 3
        assert fallback_results['success_rate'] >= 0.0
        
        # Check individual test results
        detailed_results = fallback_results['detailed_results']
        assert len(detailed_results) == 3
        
        for result in detailed_results:
            assert 'query' in result
            assert 'expected_fallback' in result
            assert 'actual_fallback' in result
            assert 'test_passed' in result
    
    def test_extract_response_content(
        self,
        mock_gemini_client,
        mock_context_manager,
        mock_confidence_manager,
        mock_agent_orchestrator,
        temp_test_data_path
    ):
        """Test response content extraction from workflow results."""
        harness = EvaluationHarness(
            mock_gemini_client,
            mock_context_manager,
            mock_confidence_manager,
            mock_agent_orchestrator,
            temp_test_data_path
        )
        
        # Test coach output extraction
        coach_result = WorkflowResult(
            success=True,
            result_data={"coach_output": {"message": "Coach response"}},
            execution_time=1.0,
            agent_logs=[],
            confidence_score=0.8
        )
        content = harness._extract_response_content(coach_result)
        assert content == "Coach response"
        
        # Test planner output extraction
        planner_result = WorkflowResult(
            success=True,
            result_data={"planner_output": {"executive_summary": "Planner summary"}},
            execution_time=1.0,
            agent_logs=[],
            confidence_score=0.8
        )
        content = harness._extract_response_content(planner_result)
        assert content == "Planner summary"
        
        # Test orchestrator output extraction
        orchestrator_result = WorkflowResult(
            success=True,
            result_data={"orchestrator_output": "Orchestrator response"},
            execution_time=1.0,
            agent_logs=[],
            confidence_score=0.8
        )
        content = harness._extract_response_content(orchestrator_result)
        assert content == "Orchestrator response"
        
        # Test failed workflow
        failed_result = WorkflowResult(
            success=False,
            result_data={},
            execution_time=1.0,
            agent_logs=[],
            confidence_score=0.0
        )
        content = harness._extract_response_content(failed_result)
        assert content == "Workflow execution failed"
    
    def test_check_fallback_usage(
        self,
        mock_gemini_client,
        mock_context_manager,
        mock_confidence_manager,
        mock_agent_orchestrator,
        temp_test_data_path
    ):
        """Test fallback usage detection."""
        harness = EvaluationHarness(
            mock_gemini_client,
            mock_context_manager,
            mock_confidence_manager,
            mock_agent_orchestrator,
            temp_test_data_path
        )
        
        # Test fallback detected in logs
        fallback_result = WorkflowResult(
            success=True,
            result_data={},
            execution_time=1.0,
            agent_logs=[
                AgentLog(
                    agent_name="orchestrator",
                    action="fallback_triggered",
                    input_data={},
                    output_data={"message": "Fallback mechanism triggered due to low confidence"},
                    execution_time=1.0,
                    timestamp=datetime.now()
                )
            ],
            confidence_score=0.9
        )
        assert harness._check_fallback_usage(fallback_result) is True
        
        # Test fallback detected by low confidence
        low_confidence_result = WorkflowResult(
            success=True,
            result_data={},
            execution_time=1.0,
            agent_logs=[],
            confidence_score=0.5  # Below threshold
        )
        assert harness._check_fallback_usage(low_confidence_result) is True
        
        # Test no fallback
        normal_result = WorkflowResult(
            success=True,
            result_data={},
            execution_time=1.0,
            agent_logs=[],
            confidence_score=0.9
        )
        assert harness._check_fallback_usage(normal_result) is False


if __name__ == "__main__":
    pytest.main([__file__])