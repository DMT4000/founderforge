"""
Unit tests for core data models.
Tests validation, serialization, and business logic for all dataclasses.
"""

import pytest
from datetime import datetime, timedelta
from src.models import (
    UserContext, Memory, WorkflowResult, Response, TokenUsage,
    BusinessInfo, UserPreferences, Message, AgentLog, MemoryType,
    ValidationError, validate_email, validate_user_id,
    create_user_context_from_dict, create_memory_from_dict
)


class TestTokenUsage:
    """Test TokenUsage dataclass."""
    
    def test_valid_token_usage(self):
        """Test valid token usage creation."""
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150
    
    def test_auto_calculate_total(self):
        """Test automatic total calculation."""
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50)
        assert usage.total_tokens == 150
    
    def test_negative_tokens_validation(self):
        """Test validation of negative token counts."""
        with pytest.raises(ValidationError):
            TokenUsage(prompt_tokens=-1, completion_tokens=50, total_tokens=49)


class TestBusinessInfo:
    """Test BusinessInfo dataclass."""
    
    def test_valid_business_info(self):
        """Test valid business info creation."""
        info = BusinessInfo(
            company_name="TechCorp",
            industry="Software",
            stage="growth",
            team_size=25
        )
        assert info.company_name == "TechCorp"
        assert info.team_size == 25
    
    def test_negative_team_size_validation(self):
        """Test validation of negative team size."""
        with pytest.raises(ValidationError):
            BusinessInfo(team_size=-5)
    
    def test_default_values(self):
        """Test default values for business info."""
        info = BusinessInfo()
        assert info.company_name == ""
        assert info.team_size == 0


class TestUserPreferences:
    """Test UserPreferences dataclass."""
    
    def test_valid_preferences(self):
        """Test valid preferences creation."""
        prefs = UserPreferences(
            communication_style="professional",
            response_length="medium",
            focus_areas=["funding", "strategy"]
        )
        assert prefs.communication_style == "professional"
        assert "funding" in prefs.focus_areas
    
    def test_invalid_communication_style(self):
        """Test validation of invalid communication style."""
        with pytest.raises(ValidationError):
            UserPreferences(communication_style="invalid")
    
    def test_invalid_response_length(self):
        """Test validation of invalid response length."""
        with pytest.raises(ValidationError):
            UserPreferences(response_length="invalid")


class TestMessage:
    """Test Message dataclass."""
    
    def test_valid_message(self):
        """Test valid message creation."""
        msg = Message(content="Hello world", role="user")
        assert msg.content == "Hello world"
        assert msg.role == "user"
        assert msg.id is not None
    
    def test_empty_content_validation(self):
        """Test validation of empty content."""
        with pytest.raises(ValidationError):
            Message(content="", role="user")
        
        with pytest.raises(ValidationError):
            Message(content="   ", role="user")
    
    def test_invalid_role_validation(self):
        """Test validation of invalid role."""
        with pytest.raises(ValidationError):
            Message(content="Hello", role="invalid")
    
    def test_negative_token_count_validation(self):
        """Test validation of negative token count."""
        with pytest.raises(ValidationError):
            Message(content="Hello", role="user", token_count=-1)


class TestUserContext:
    """Test UserContext dataclass."""
    
    def test_valid_user_context(self):
        """Test valid user context creation."""
        context = UserContext(
            user_id="user123",
            goals=["Launch MVP", "Raise funding"],
            business_info=BusinessInfo(company_name="StartupCorp"),
            preferences=UserPreferences()
        )
        assert context.user_id == "user123"
        assert len(context.goals) == 2
        assert context.business_info.company_name == "StartupCorp"
    
    def test_empty_user_id_validation(self):
        """Test validation of empty user ID."""
        with pytest.raises(ValidationError):
            UserContext(user_id="")
        
        with pytest.raises(ValidationError):
            UserContext(user_id="   ")
    
    def test_negative_token_count_validation(self):
        """Test validation of negative token count."""
        with pytest.raises(ValidationError):
            UserContext(user_id="user123", token_count=-1)
    
    def test_add_message(self):
        """Test adding message to context."""
        context = UserContext(user_id="user123")
        context.add_message("Hello", "user")
        
        assert len(context.chat_history) == 1
        assert context.chat_history[0].content == "Hello"
        assert context.chat_history[0].role == "user"
    
    def test_get_recent_messages(self):
        """Test getting recent messages."""
        context = UserContext(user_id="user123")
        
        # Add multiple messages
        for i in range(15):
            context.add_message(f"Message {i}", "user")
        
        recent = context.get_recent_messages(limit=5)
        assert len(recent) == 5
        # Should be in reverse chronological order
        assert recent[0].content == "Message 14"
    
    def test_to_dict_serialization(self):
        """Test dictionary serialization."""
        context = UserContext(
            user_id="user123",
            goals=["goal1"],
            business_info=BusinessInfo(company_name="Test Corp"),
            preferences=UserPreferences(communication_style="casual")
        )
        context.add_message("Test message", "user")
        
        data = context.to_dict()
        assert data["user_id"] == "user123"
        assert data["goals"] == ["goal1"]
        assert data["business_info"]["company_name"] == "Test Corp"
        assert data["preferences"]["communication_style"] == "casual"
        assert len(data["chat_history"]) == 1


class TestMemory:
    """Test Memory dataclass."""
    
    def test_valid_memory(self):
        """Test valid memory creation."""
        memory = Memory(
            user_id="user123",
            content="User prefers technical explanations",
            memory_type=MemoryType.LONG_TERM,
            confidence=0.9
        )
        assert memory.user_id == "user123"
        assert memory.memory_type == MemoryType.LONG_TERM
        assert memory.confidence == 0.9
    
    def test_empty_user_id_validation(self):
        """Test validation of empty user ID."""
        with pytest.raises(ValidationError):
            Memory(user_id="", content="test", confidence=0.5)
    
    def test_empty_content_validation(self):
        """Test validation of empty content."""
        with pytest.raises(ValidationError):
            Memory(user_id="user123", content="", confidence=0.5)
    
    def test_confidence_range_validation(self):
        """Test validation of confidence range."""
        with pytest.raises(ValidationError):
            Memory(user_id="user123", content="test", confidence=-0.1)
        
        with pytest.raises(ValidationError):
            Memory(user_id="user123", content="test", confidence=1.1)
    
    def test_expiration_date_validation(self):
        """Test validation of expiration date."""
        now = datetime.now()
        past = now - timedelta(hours=1)
        
        with pytest.raises(ValidationError):
            Memory(
                user_id="user123",
                content="test",
                confidence=0.5,
                created_at=now,
                expires_at=past
            )
    
    def test_is_expired(self):
        """Test expiration checking."""
        now = datetime.now()
        future = now + timedelta(hours=1)
        past = now - timedelta(hours=1)
        
        # Not expired
        memory1 = Memory(
            user_id="user123",
            content="test",
            confidence=0.5,
            expires_at=future
        )
        assert not memory1.is_expired()
        
        # Expired
        memory2 = Memory(
            user_id="user123",
            content="test",
            confidence=0.5,
            created_at=past - timedelta(hours=1),
            expires_at=past
        )
        assert memory2.is_expired()
        
        # No expiration
        memory3 = Memory(
            user_id="user123",
            content="test",
            confidence=0.5
        )
        assert not memory3.is_expired()
    
    def test_to_dict_serialization(self):
        """Test dictionary serialization."""
        memory = Memory(
            user_id="user123",
            content="test content",
            memory_type=MemoryType.SHORT_TERM,
            confidence=0.8
        )
        
        data = memory.to_dict()
        assert data["user_id"] == "user123"
        assert data["content"] == "test content"
        assert data["memory_type"] == "SHORT_TERM"
        assert data["confidence"] == 0.8


class TestAgentLog:
    """Test AgentLog dataclass."""
    
    def test_valid_agent_log(self):
        """Test valid agent log creation."""
        log = AgentLog(
            agent_name="TestAgent",
            action="process_request",
            input_data={"query": "test"},
            output_data={"result": "success"},
            execution_time=0.5
        )
        assert log.agent_name == "TestAgent"
        assert log.execution_time == 0.5
        assert log.success is True
    
    def test_empty_agent_name_validation(self):
        """Test validation of empty agent name."""
        with pytest.raises(ValidationError):
            AgentLog(
                agent_name="",
                action="test",
                input_data={},
                output_data={},
                execution_time=0.1
            )
    
    def test_empty_action_validation(self):
        """Test validation of empty action."""
        with pytest.raises(ValidationError):
            AgentLog(
                agent_name="TestAgent",
                action="",
                input_data={},
                output_data={},
                execution_time=0.1
            )
    
    def test_negative_execution_time_validation(self):
        """Test validation of negative execution time."""
        with pytest.raises(ValidationError):
            AgentLog(
                agent_name="TestAgent",
                action="test",
                input_data={},
                output_data={},
                execution_time=-0.1
            )


class TestWorkflowResult:
    """Test WorkflowResult dataclass."""
    
    def test_valid_workflow_result(self):
        """Test valid workflow result creation."""
        result = WorkflowResult(
            success=True,
            result_data={"output": "test"},
            execution_time=1.5,
            confidence_score=0.9
        )
        assert result.success is True
        assert result.confidence_score == 0.9
        assert result.workflow_id is not None
    
    def test_negative_execution_time_validation(self):
        """Test validation of negative execution time."""
        with pytest.raises(ValidationError):
            WorkflowResult(success=True, execution_time=-1.0)
    
    def test_confidence_score_range_validation(self):
        """Test validation of confidence score range."""
        with pytest.raises(ValidationError):
            WorkflowResult(success=True, confidence_score=-0.1)
        
        with pytest.raises(ValidationError):
            WorkflowResult(success=True, confidence_score=1.1)
    
    def test_add_agent_log(self):
        """Test adding agent log."""
        result = WorkflowResult(success=True)
        log = AgentLog(
            agent_name="TestAgent",
            action="test",
            input_data={},
            output_data={},
            execution_time=0.5
        )
        
        result.add_agent_log(log)
        assert len(result.agent_logs) == 1
        assert result.agent_logs[0].agent_name == "TestAgent"
    
    def test_get_total_execution_time(self):
        """Test calculating total execution time."""
        result = WorkflowResult(success=True)
        
        log1 = AgentLog("Agent1", "action1", {}, {}, 0.5)
        log2 = AgentLog("Agent2", "action2", {}, {}, 0.3)
        
        result.add_agent_log(log1)
        result.add_agent_log(log2)
        
        assert result.get_total_execution_time() == 0.8
    
    def test_to_dict_serialization(self):
        """Test dictionary serialization."""
        result = WorkflowResult(
            success=True,
            result_data={"key": "value"},
            confidence_score=0.8
        )
        
        log = AgentLog("TestAgent", "test", {}, {}, 0.1)
        result.add_agent_log(log)
        
        data = result.to_dict()
        assert data["success"] is True
        assert data["result_data"]["key"] == "value"
        assert data["confidence_score"] == 0.8
        assert len(data["agent_logs"]) == 1


class TestResponse:
    """Test Response dataclass."""
    
    def test_valid_response(self):
        """Test valid response creation."""
        response = Response(
            content="This is a test response",
            confidence=0.9,
            sources=["memory", "context"],
            token_usage=TokenUsage(100, 50, 150)
        )
        assert response.content == "This is a test response"
        assert response.confidence == 0.9
        assert len(response.sources) == 2
    
    def test_empty_content_validation(self):
        """Test validation of empty content."""
        with pytest.raises(ValidationError):
            Response(content="")
        
        with pytest.raises(ValidationError):
            Response(content="   ")
    
    def test_confidence_range_validation(self):
        """Test validation of confidence range."""
        with pytest.raises(ValidationError):
            Response(content="test", confidence=-0.1)
        
        with pytest.raises(ValidationError):
            Response(content="test", confidence=1.1)
    
    def test_negative_processing_time_validation(self):
        """Test validation of negative processing time."""
        with pytest.raises(ValidationError):
            Response(content="test", processing_time=-0.1)
    
    def test_is_high_confidence(self):
        """Test high confidence checking."""
        response1 = Response(content="test", confidence=0.9)
        response2 = Response(content="test", confidence=0.7)
        
        assert response1.is_high_confidence(threshold=0.8)
        assert not response2.is_high_confidence(threshold=0.8)
    
    def test_add_source(self):
        """Test adding sources."""
        response = Response(content="test")
        response.add_source("memory")
        response.add_source("context")
        response.add_source("memory")  # Duplicate should be ignored
        
        assert len(response.sources) == 2
        assert "memory" in response.sources
        assert "context" in response.sources
    
    def test_to_dict_serialization(self):
        """Test dictionary serialization."""
        response = Response(
            content="test response",
            confidence=0.8,
            sources=["test"],
            token_usage=TokenUsage(50, 25, 75)
        )
        
        data = response.to_dict()
        assert data["content"] == "test response"
        assert data["confidence"] == 0.8
        assert data["sources"] == ["test"]
        assert data["token_usage"]["total_tokens"] == 75


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_validate_email(self):
        """Test email validation."""
        assert validate_email("test@example.com")
        assert validate_email("user.name+tag@domain.co.uk")
        assert not validate_email("invalid-email")
        assert not validate_email("@domain.com")
        assert not validate_email("user@")
    
    def test_validate_user_id(self):
        """Test user ID validation."""
        assert validate_user_id("user123")
        assert validate_user_id("user_123")
        assert validate_user_id("user-123")
        assert not validate_user_id("")
        assert not validate_user_id("   ")
        assert not validate_user_id("user@123")
        assert not validate_user_id("user 123")
    
    def test_create_user_context_from_dict(self):
        """Test creating UserContext from dictionary."""
        data = {
            "user_id": "user123",
            "goals": ["goal1", "goal2"],
            "business_info": {
                "company_name": "TestCorp",
                "team_size": 5
            },
            "preferences": {
                "communication_style": "professional"
            },
            "chat_history": [
                {
                    "id": "msg1",
                    "content": "Hello",
                    "role": "user",
                    "timestamp": datetime.now().isoformat(),
                    "token_count": 10
                }
            ],
            "token_count": 100
        }
        
        context = create_user_context_from_dict(data)
        assert context.user_id == "user123"
        assert len(context.goals) == 2
        assert context.business_info.company_name == "TestCorp"
        assert context.preferences.communication_style == "professional"
        assert len(context.chat_history) == 1
        assert context.token_count == 100
    
    def test_create_memory_from_dict(self):
        """Test creating Memory from dictionary."""
        data = {
            "id": "mem123",
            "user_id": "user123",
            "content": "Test memory",
            "memory_type": "LONG_TERM",
            "confidence": 0.8,
            "created_at": datetime.now().isoformat(),
            "expires_at": None
        }
        
        memory = create_memory_from_dict(data)
        assert memory.id == "mem123"
        assert memory.user_id == "user123"
        assert memory.content == "Test memory"
        assert memory.memory_type == MemoryType.LONG_TERM
        assert memory.confidence == 0.8
        assert memory.expires_at is None


if __name__ == "__main__":
    pytest.main([__file__])