"""
Tests for the multi-agent system using LangGraph.
"""

import pytest
import asyncio
import json
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from src.agents import (
    AgentOrchestrator, OrchestratorAgent, ValidatorAgent, PlannerAgent,
    ToolCallerAgent, CoachAgent, AgentConfig, AgentType, WorkflowState
)
from src.models import WorkflowResult, AgentLog
from src.gemini_client import GeminiClient, GeminiResponse, MockMode
from src.context_manager import ContextAssembler
from src.confidence_manager import ConfidenceManager


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
    return Mock(spec=ConfidenceManager)


@pytest.fixture
def agent_config():
    """Create a test agent configuration."""
    return AgentConfig(
        name="test_agent",
        agent_type=AgentType.ORCHESTRATOR,
        max_retries=2,
        timeout_seconds=10.0,
        confidence_threshold=0.7
    )


@pytest.fixture
def sample_workflow_state():
    """Create a sample workflow state for testing."""
    return WorkflowState(
        user_id="test_user",
        user_context={"goals": ["test goal"], "business_info": {"stage": "mvp"}},
        current_task="test_task",
        task_data={"test": "data"},
        agent_outputs={},
        workflow_result={},
        confidence_scores={},
        execution_logs=[],
        error_messages=[],
        next_agent=None,
        completed=False
    )


class TestOrchestratorAgent:
    """Test cases for OrchestratorAgent."""
    
    @pytest.mark.asyncio
    async def test_orchestrator_execute_success(
        self,
        agent_config,
        mock_gemini_client,
        mock_context_manager,
        mock_confidence_manager,
        sample_workflow_state
    ):
        """Test successful orchestrator execution."""
        agent = OrchestratorAgent(
            agent_config, mock_gemini_client, mock_context_manager, mock_confidence_manager
        )
        
        # Mock Gemini response
        mock_response = GeminiResponse(
            content='{"complexity": "medium", "required_agents": ["validator"], "confidence": 0.8}',
            confidence=0.8,
            model_used="mock",
            token_usage={},
            response_time=1.0
        )
        mock_gemini_client.generate_content = Mock(return_value=mock_response)
        
        result = await agent.execute(sample_workflow_state)
        
        assert "orchestrator_output" in result
        assert "next_agent" in result
        assert "confidence_scores" in result
        assert "execution_logs" in result
        assert len(result["execution_logs"]) == 1
    
    @pytest.mark.asyncio
    async def test_orchestrator_execute_failure(
        self,
        agent_config,
        mock_gemini_client,
        mock_context_manager,
        mock_confidence_manager,
        sample_workflow_state
    ):
        """Test orchestrator execution failure handling."""
        agent = OrchestratorAgent(
            agent_config, mock_gemini_client, mock_context_manager, mock_confidence_manager
        )
        
        # Mock Gemini failure
        mock_gemini_client.generate_content = Mock(side_effect=Exception("API Error"))
        
        result = await agent.execute(sample_workflow_state)
        
        assert "orchestrator_output" in result
        assert "error" in result["orchestrator_output"]
        assert "error_messages" in result
        assert len(result["error_messages"]) > 0


class TestValidatorAgent:
    """Test cases for ValidatorAgent."""
    
    @pytest.mark.asyncio
    async def test_validator_execute_success(
        self,
        mock_gemini_client,
        mock_context_manager,
        mock_confidence_manager,
        sample_workflow_state
    ):
        """Test successful validator execution."""
        config = AgentConfig(
            name="validator",
            agent_type=AgentType.VALIDATOR,
            confidence_threshold=0.8
        )
        
        agent = ValidatorAgent(
            config, mock_gemini_client, mock_context_manager, mock_confidence_manager
        )
        
        # Set up test data for funding form validation
        sample_workflow_state["current_task"] = "funding_form"
        sample_workflow_state["task_data"] = {
            "company_name": "Test Company",
            "funding_amount": 100000,
            "business_plan": "A" * 150  # Meets minimum length requirement
        }
        
        result = await agent.execute(sample_workflow_state)
        
        assert "validator_output" in result
        assert "is_valid" in result["validator_output"]
        assert result["validator_output"]["is_valid"] is True
        assert "confidence_scores" in result
        assert "execution_logs" in result
    
    @pytest.mark.asyncio
    async def test_validator_validation_failure(
        self,
        mock_gemini_client,
        mock_context_manager,
        mock_confidence_manager,
        sample_workflow_state
    ):
        """Test validator with invalid data."""
        config = AgentConfig(
            name="validator",
            agent_type=AgentType.VALIDATOR,
            confidence_threshold=0.8
        )
        
        agent = ValidatorAgent(
            config, mock_gemini_client, mock_context_manager, mock_confidence_manager
        )
        
        # Set up invalid test data
        sample_workflow_state["current_task"] = "funding_form"
        sample_workflow_state["task_data"] = {
            "company_name": "",  # Invalid - empty
            "funding_amount": -1000,  # Invalid - negative
            "business_plan": "Short"  # Invalid - too short
        }
        
        result = await agent.execute(sample_workflow_state)
        
        assert "validator_output" in result
        assert result["validator_output"]["is_valid"] is False
        assert len(result["validator_output"]["errors"]) > 0


class TestPlannerAgent:
    """Test cases for PlannerAgent."""
    
    @pytest.mark.asyncio
    async def test_planner_execute_success(
        self,
        mock_gemini_client,
        mock_context_manager,
        mock_confidence_manager,
        sample_workflow_state
    ):
        """Test successful planner execution."""
        config = AgentConfig(
            name="planner",
            agent_type=AgentType.PLANNER,
            confidence_threshold=0.7
        )
        
        agent = PlannerAgent(
            config, mock_gemini_client, mock_context_manager, mock_confidence_manager
        )
        
        # Mock Gemini response with valid JSON plan
        plan_json = {
            "executive_summary": "Test plan summary",
            "action_items": [
                {
                    "title": "Test Action",
                    "description": "Test description",
                    "priority": "high",
                    "timeline": "1 week",
                    "resources_needed": ["resource1"]
                }
            ],
            "success_metrics": ["metric1"],
            "risks": [{"risk": "test risk", "mitigation": "test mitigation"}],
            "next_steps": ["step1"]
        }
        
        mock_response = GeminiResponse(
            content=json.dumps(plan_json),
            confidence=0.8,
            model_used="mock",
            token_usage={},
            response_time=1.0
        )
        mock_gemini_client.generate_content = Mock(return_value=mock_response)
        
        result = await agent.execute(sample_workflow_state)
        
        assert "planner_output" in result
        assert "executive_summary" in result["planner_output"]
        assert "action_items" in result["planner_output"]
        assert len(result["planner_output"]["action_items"]) > 0
        assert "confidence_scores" in result
    
    @pytest.mark.asyncio
    async def test_planner_text_parsing(
        self,
        mock_gemini_client,
        mock_context_manager,
        mock_confidence_manager,
        sample_workflow_state
    ):
        """Test planner with non-JSON text response."""
        config = AgentConfig(
            name="planner",
            agent_type=AgentType.PLANNER,
            confidence_threshold=0.7
        )
        
        agent = PlannerAgent(
            config, mock_gemini_client, mock_context_manager, mock_confidence_manager
        )
        
        # Mock Gemini response with text (not JSON)
        text_response = """
        Action Items:
        - Complete market research
        - Develop MVP prototype
        - Secure initial funding
        
        Success Metrics:
        - User acquisition rate
        - Revenue growth
        
        Next Steps:
        - Schedule team meeting
        - Review budget
        """
        
        mock_response = GeminiResponse(
            content=text_response,
            confidence=0.7,
            model_used="mock",
            token_usage={},
            response_time=1.0
        )
        mock_gemini_client.generate_content = Mock(return_value=mock_response)
        
        result = await agent.execute(sample_workflow_state)
        
        assert "planner_output" in result
        assert "action_items" in result["planner_output"]
        # Should have parsed some action items from text
        assert len(result["planner_output"]["action_items"]) > 0


class TestToolCallerAgent:
    """Test cases for ToolCallerAgent."""
    
    @pytest.mark.asyncio
    async def test_tool_caller_execute_success(
        self,
        mock_gemini_client,
        mock_context_manager,
        mock_confidence_manager,
        sample_workflow_state
    ):
        """Test successful tool caller execution."""
        config = AgentConfig(
            name="tool_caller",
            agent_type=AgentType.TOOL_CALLER,
            confidence_threshold=0.8
        )
        
        agent = ToolCallerAgent(
            config, mock_gemini_client, mock_context_manager, mock_confidence_manager
        )
        
        # Set up planner output that would trigger tool calls
        sample_workflow_state["agent_outputs"]["planner_output"] = {
            "action_items": [
                {
                    "title": "Analyze market data",
                    "description": "Perform comprehensive data analysis"
                },
                {
                    "title": "Generate report",
                    "description": "Create summary report"
                }
            ]
        }
        
        result = await agent.execute(sample_workflow_state)
        
        assert "tool_caller_output" in result
        assert "tool_results" in result["tool_caller_output"]
        assert len(result["tool_caller_output"]["tool_results"]) > 0
        assert "confidence_scores" in result


class TestCoachAgent:
    """Test cases for CoachAgent."""
    
    @pytest.mark.asyncio
    async def test_coach_execute_success(
        self,
        mock_gemini_client,
        mock_context_manager,
        mock_confidence_manager,
        sample_workflow_state
    ):
        """Test successful coach execution."""
        config = AgentConfig(
            name="coach",
            agent_type=AgentType.COACH,
            confidence_threshold=0.7
        )
        
        agent = CoachAgent(
            config, mock_gemini_client, mock_context_manager, mock_confidence_manager
        )
        
        # Mock Gemini response
        mock_response = GeminiResponse(
            content="Great work on your progress! Here's some encouraging feedback...",
            confidence=0.8,
            model_used="mock",
            token_usage={},
            response_time=1.0
        )
        mock_gemini_client.generate_content = Mock(return_value=mock_response)
        
        result = await agent.execute(sample_workflow_state)
        
        assert "coach_output" in result
        assert "message" in result["coach_output"]
        assert "confidence" in result["coach_output"]
        assert "coaching_type" in result["coach_output"]
        assert "confidence_scores" in result


class TestAgentOrchestrator:
    """Test cases for AgentOrchestrator."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            temp_path = f.name
        yield temp_path
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    @pytest.mark.asyncio
    async def test_orchestrator_initialization(
        self,
        mock_gemini_client,
        mock_context_manager,
        mock_confidence_manager,
        temp_db_path
    ):
        """Test orchestrator initialization."""
        orchestrator = AgentOrchestrator(
            mock_gemini_client,
            mock_context_manager,
            mock_confidence_manager,
            temp_db_path
        )
        
        assert len(orchestrator.agents) == 5
        assert "orchestrator" in orchestrator.agents
        assert "validator" in orchestrator.agents
        assert "planner" in orchestrator.agents
        assert "tool_caller" in orchestrator.agents
        assert "coach" in orchestrator.agents
    
    @pytest.mark.asyncio
    async def test_execute_workflow_simple(
        self,
        mock_gemini_client,
        mock_context_manager,
        mock_confidence_manager,
        temp_db_path
    ):
        """Test simple workflow execution."""
        orchestrator = AgentOrchestrator(
            mock_gemini_client,
            mock_context_manager,
            mock_confidence_manager,
            temp_db_path
        )
        
        # Mock successful responses for all agents
        mock_responses = [
            GeminiResponse(
                content='{"complexity": "low", "required_agents": ["validator"], "confidence": 0.8}',
                confidence=0.8,
                model_used="mock",
                token_usage={},
                response_time=1.0
            ),
            GeminiResponse(
                content='{"executive_summary": "Simple plan", "action_items": []}',
                confidence=0.7,
                model_used="mock",
                token_usage={},
                response_time=1.0
            )
        ]
        
        mock_gemini_client.generate_content = Mock(side_effect=mock_responses)
        
        result = await orchestrator.execute_workflow(
            workflow_type="general",
            user_id="test_user",
            task_data={"test": "data"},
            user_context={"goals": ["test goal"]}
        )
        
        assert isinstance(result, WorkflowResult)
        assert result.success is True
        assert len(result.agent_logs) > 0
        assert result.execution_time > 0
    
    def test_get_available_workflows(
        self,
        mock_gemini_client,
        mock_context_manager,
        mock_confidence_manager,
        temp_db_path
    ):
        """Test getting available workflows."""
        orchestrator = AgentOrchestrator(
            mock_gemini_client,
            mock_context_manager,
            mock_confidence_manager,
            temp_db_path
        )
        
        workflows = orchestrator.get_available_workflows()
        
        assert isinstance(workflows, list)
        assert "funding_form" in workflows
        assert "daily_planning" in workflows
        assert "general" in workflows
    
    @pytest.mark.asyncio
    async def test_health_check(
        self,
        mock_gemini_client,
        mock_context_manager,
        mock_confidence_manager,
        temp_db_path
    ):
        """Test health check functionality."""
        orchestrator = AgentOrchestrator(
            mock_gemini_client,
            mock_context_manager,
            mock_confidence_manager,
            temp_db_path
        )
        
        # Mock health check
        mock_gemini_client.health_check = Mock(return_value=True)
        
        health_status = await orchestrator.health_check()
        
        assert isinstance(health_status, dict)
        assert "orchestrator" in health_status
        assert "validator" in health_status
        assert "planner" in health_status
        assert "tool_caller" in health_status
        assert "coach" in health_status
        assert "gemini_client" in health_status
    
    def test_get_agent_performance_metrics(
        self,
        mock_gemini_client,
        mock_context_manager,
        mock_confidence_manager,
        temp_db_path
    ):
        """Test getting agent performance metrics."""
        orchestrator = AgentOrchestrator(
            mock_gemini_client,
            mock_context_manager,
            mock_confidence_manager,
            temp_db_path
        )
        
        metrics = orchestrator.get_agent_performance_metrics()
        
        assert isinstance(metrics, dict)
        assert len(metrics) == 5
        for agent_name, agent_metrics in metrics.items():
            assert "agent_name" in agent_metrics
            assert "execution_count" in agent_metrics
            assert "success_rate" in agent_metrics
            assert "average_execution_time" in agent_metrics


if __name__ == "__main__":
    pytest.main([__file__])