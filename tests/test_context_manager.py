"""
Tests for context management system.
"""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch

from src.context_manager import ContextAssembler, Context, TokenManager, ContextSummarizer
from src.models import UserContext, Message, UserPreferences, TokenUsage
from src.database import DatabaseManager


class TestContextAssembler:
    """Test cases for ContextAssembler class."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.assembler = ContextAssembler()
        
        # Mock the data directories to use temp directory
        self.assembler.data_dir = self.temp_dir
        self.assembler.chat_history_dir = self.temp_dir / "chat_history"
        self.assembler.business_data_dir = self.temp_dir / "business_data"
        self.assembler.guard_rails_dir = self.temp_dir / "guard_rails"
        
        # Create test directories
        for directory in [self.assembler.chat_history_dir, 
                         self.assembler.business_data_dir, 
                         self.assembler.guard_rails_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def test_assemble_context_basic(self):
        """Test basic context assembly."""
        user_id = "test_user_123"
        
        # Mock database response
        with patch.object(self.assembler.db_manager, 'execute_query') as mock_query:
            mock_query.side_effect = [
                # User query result
                [{'id': user_id, 'name': 'Test User', 'email': 'test@example.com', 'preferences': '{}'}],
                # Memory query result
                [{'content': 'Build a successful startup'}, {'content': 'Focus on customer acquisition'}]
            ]
            
            context = self.assembler.assemble_context(user_id, "What should I focus on?")
            
            assert context.user_context.user_id == user_id
            assert len(context.goals) == 2
            assert "Build a successful startup" in context.goals
            assert context.token_count > 0
            assert user_id in str(context.sources)
    
    def test_load_chat_history(self):
        """Test loading chat history from JSON file."""
        user_id = "test_user_456"
        
        # Create test chat history file
        chat_data = {
            "messages": [
                {
                    "id": "msg1",
                    "content": "Hello, I need help with my startup",
                    "role": "user",
                    "timestamp": datetime.now().isoformat(),
                    "token_count": 10
                },
                {
                    "id": "msg2",
                    "content": "I'd be happy to help! What specific area?",
                    "role": "assistant",
                    "timestamp": datetime.now().isoformat(),
                    "token_count": 12
                }
            ]
        }
        
        chat_file = self.assembler.chat_history_dir / f"{user_id}_history.json"
        with open(chat_file, 'w') as f:
            json.dump(chat_data, f)
        
        messages = self.assembler._load_chat_history(user_id)
        
        assert len(messages) == 2
        assert messages[0].content == "Hello, I need help with my startup"
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"
    
    def test_load_business_data(self):
        """Test loading business data from text and JSON files."""
        user_id = "test_user_789"
        
        # Create test business data files
        user_business_file = self.assembler.business_data_dir / f"{user_id}_business.txt"
        with open(user_business_file, 'w') as f:
            f.write("SaaS startup in fintech space, Series A stage")
        
        general_file = self.assembler.business_data_dir / "market_trends.txt"
        with open(general_file, 'w') as f:
            f.write("AI and automation are key trends in 2024")
        
        json_file = self.assembler.business_data_dir / "industry_data.json"
        with open(json_file, 'w') as f:
            json.dump({"fintech": {"growth_rate": "15%", "key_players": ["Stripe", "Square"]}}, f)
        
        business_data = self.assembler._load_business_data(user_id)
        
        assert "user_business_info" in business_data
        assert "SaaS startup in fintech space" in business_data["user_business_info"]
        assert "Market Trends" in business_data
        assert "Industry Data" in business_data
        assert business_data["Industry Data"]["fintech"]["growth_rate"] == "15%"
    
    def test_load_guard_rails(self):
        """Test loading guard rails from files."""
        # Create test guard rails file
        rails_file = self.assembler.guard_rails_dir / "custom_rules.txt"
        with open(rails_file, 'w') as f:
            f.write("Always verify financial claims\nFocus on sustainable growth strategies")
        
        guard_rails = self.assembler._load_guard_rails()
        
        # Should include default rails plus custom ones
        assert len(guard_rails) > 5  # Default rails plus custom
        assert "Always verify financial claims" in guard_rails
        assert "Focus on sustainable growth strategies" in guard_rails
        assert "Provide helpful, accurate, and relevant information" in guard_rails
    
    def test_extract_priority_data(self):
        """Test extraction of priority data."""
        user_context = UserContext(
            user_id="test_user",
            preferences=UserPreferences(
                communication_style="technical",
                response_length="long",
                focus_areas=["funding", "product development"]
            )
        )
        
        business_data = {
            "user_business_info": "AI-powered analytics platform for healthcare providers"
        }
        
        priority_data = self.assembler._extract_priority_data(user_context, business_data)
        
        assert priority_data["Communication Style"] == "technical"
        assert priority_data["Response Length"] == "long"
        assert priority_data["Focus Areas"] == "funding, product development"
        assert "AI-powered analytics platform" in priority_data["Key Business Info"]
    
    def test_save_chat_message(self):
        """Test saving chat message to history file."""
        user_id = "test_user_save"
        message = Message(
            content="Test message content",
            role="user",
            timestamp=datetime.now()
        )
        
        success = self.assembler.save_chat_message(user_id, message)
        assert success
        
        # Verify file was created and contains message
        chat_file = self.assembler.chat_history_dir / f"{user_id}_history.json"
        assert chat_file.exists()
        
        with open(chat_file, 'r') as f:
            data = json.load(f)
        
        assert len(data["messages"]) == 1
        assert data["messages"][0]["content"] == "Test message content"
        assert data["messages"][0]["role"] == "user"
    
    def test_save_business_data(self):
        """Test saving business data to file."""
        user_id = "test_user_business"
        business_info = "Tech startup focused on sustainable energy solutions"
        
        success = self.assembler.save_business_data(user_id, business_info, "company_overview")
        assert success
        
        # Verify file was created
        business_file = self.assembler.business_data_dir / f"{user_id}_company_overview.txt"
        assert business_file.exists()
        
        with open(business_file, 'r') as f:
            content = f.read()
        
        assert content == business_info
    
    def test_context_to_prompt_string(self):
        """Test context conversion to prompt string."""
        user_context = UserContext(user_id="test_user")
        
        context = Context(
            user_context=user_context,
            goals=["Achieve product-market fit", "Raise Series A"],
            business_data={"Company Stage": "MVP", "Industry": "SaaS"},
            priority_data={"Communication Style": "professional"},
            guard_rails=["Be helpful", "Stay focused"],
            chat_history=[
                Message(content="What's my next step?", role="user"),
                Message(content="Focus on customer validation", role="assistant")
            ]
        )
        
        prompt_string = context.to_prompt_string()
        
        assert "=== CRITICAL INFORMATION ===" in prompt_string
        assert "Communication Style: professional" in prompt_string
        assert "=== USER GOALS ===" in prompt_string
        assert "Achieve product-market fit" in prompt_string
        assert "=== BUSINESS CONTEXT ===" in prompt_string
        assert "Company Stage: MVP" in prompt_string
        assert "=== RECENT CONVERSATION ===" in prompt_string
        assert "User: What's my next step?" in prompt_string
        assert "=== GUIDELINES ===" in prompt_string
        assert "Be helpful" in prompt_string
    
    def test_token_estimation(self):
        """Test token count estimation."""
        context = Context(
            user_context=UserContext(user_id="test_user"),
            goals=["Test goal"],
            business_data={"key": "value"},
            guard_rails=["Test rule"]
        )
        
        token_count = self.assembler._estimate_token_count(context)
        
        assert token_count > 0
        assert isinstance(token_count, int)
        
        # Token count should be reasonable (not too high or low)
        prompt_length = len(context.to_prompt_string())
        expected_range = (prompt_length // 6, prompt_length // 2)
        assert expected_range[0] <= token_count <= expected_range[1]


class TestContext:
    """Test cases for Context class."""
    
    def test_context_creation(self):
        """Test basic context creation."""
        user_context = UserContext(user_id="test_user")
        context = Context(user_context=user_context)
        
        assert context.user_context.user_id == "test_user"
        assert context.business_data == {}
        assert context.chat_history == []
        assert context.goals == []
        assert context.guard_rails == []
        assert context.token_count == 0
        assert context.sources == []
    
    def test_add_source(self):
        """Test adding sources to context."""
        context = Context(user_context=UserContext(user_id="test_user"))
        
        context.add_source("database")
        context.add_source("chat_history")
        context.add_source("database")  # Duplicate should be ignored
        
        assert len(context.sources) == 2
        assert "database" in context.sources
        assert "chat_history" in context.sources
    
    def test_empty_context_prompt(self):
        """Test prompt generation with empty context."""
        context = Context(user_context=UserContext(user_id="test_user"))
        prompt = context.to_prompt_string()
        
        # Should handle empty context gracefully
        assert isinstance(prompt, str)
        assert len(prompt.strip()) == 0  # Should be empty or minimal


if __name__ == "__main__":
    pytest.main([__file__])


class TestTokenManager:
    """Test cases for TokenManager class."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Mock settings to use temp directory
        with patch('src.context_manager.settings') as mock_settings:
            mock_settings.data_dir = self.temp_dir
            mock_settings.get_feature_flag.return_value = 50000
            self.token_manager = TokenManager()
    
    def test_log_token_usage(self):
        """Test token usage logging."""
        from src.context_manager import TokenManager
        from src.models import TokenUsage
        
        # Create token manager with temp directory
        token_manager = TokenManager()
        token_manager.data_dir = self.temp_dir
        token_manager.token_logs_dir = self.temp_dir / "token_logs"
        token_manager.token_logs_dir.mkdir(parents=True, exist_ok=True)
        
        user_id = "test_user"
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        
        token_manager.log_token_usage(user_id, usage, "test_operation")
        
        # Check session usage updated
        session_usage = token_manager.get_session_usage()
        assert session_usage.total_tokens == 150
        assert session_usage.prompt_tokens == 100
        assert session_usage.completion_tokens == 50
        
        # Check daily usage updated
        today = datetime.now().strftime("%Y-%m-%d")
        daily_usage = token_manager.get_daily_usage(user_id, today)
        assert daily_usage is not None
        assert daily_usage.total_tokens == 150
        
        # Check log files created
        daily_log_file = token_manager.token_logs_dir / f"tokens_{today}.jsonl"
        user_log_file = token_manager.token_logs_dir / f"user_{user_id}_tokens.jsonl"
        
        assert daily_log_file.exists()
        assert user_log_file.exists()
    
    def test_usage_summary(self):
        """Test usage summary generation."""
        from src.context_manager import TokenManager
        from src.models import TokenUsage
        
        token_manager = TokenManager()
        token_manager.data_dir = self.temp_dir
        token_manager.token_logs_dir = self.temp_dir / "token_logs"
        token_manager.token_logs_dir.mkdir(parents=True, exist_ok=True)
        
        user_id = "test_user"
        
        # Log some usage
        for i in range(3):
            usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
            token_manager.log_token_usage(user_id, usage, f"operation_{i}")
        
        summary = token_manager.get_usage_summary(user_id, days=7)
        
        assert summary["user_id"] == user_id
        assert summary["period_days"] == 7
        assert summary["total_usage"].total_tokens == 450  # 3 * 150
        assert len(summary["daily_usage"]) == 7  # 7 days of data
    
    def test_check_usage_limits(self):
        """Test usage limit checking."""
        from src.context_manager import TokenManager
        from src.models import TokenUsage
        
        token_manager = TokenManager()
        token_manager.data_dir = self.temp_dir
        token_manager.token_logs_dir = self.temp_dir / "token_logs"
        token_manager.token_logs_dir.mkdir(parents=True, exist_ok=True)
        
        user_id = "test_user"
        
        # Log some existing usage
        usage = TokenUsage(prompt_tokens=400, completion_tokens=200, total_tokens=600)
        token_manager.log_token_usage(user_id, usage, "existing")
        
        # Mock settings for the check_usage_limits call
        with patch('src.context_manager.settings') as mock_settings:
            mock_settings.get_feature_flag.side_effect = lambda key, default: {
                "daily_token_limit": 1000,
                "max_context_tokens": 500
            }.get(key, default)
            
            # Check limits for new proposed usage
            result = token_manager.check_usage_limits(user_id, 500)  # 600 + 500 = 1100 > 1000
            
            assert not result["within_daily_limit"]  # 600 + 500 > 1000
            assert result["within_context_limit"]    # 500 <= 500
            assert result["current_daily_usage"] == 600
            assert len(result["recommendations"]) > 0


class TestContextSummarizer:
    """Test cases for ContextSummarizer class."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        with patch('src.context_manager.settings') as mock_settings:
            mock_settings.get_feature_flag.return_value = 1000  # Small limit for testing
            mock_settings.data_dir = self.temp_dir
            
            from src.context_manager import TokenManager, ContextSummarizer
            self.token_manager = TokenManager()
            self.summarizer = ContextSummarizer(self.token_manager)
    
    def test_summarize_context_within_limits(self):
        """Test context that's already within limits."""
        from src.context_manager import Context
        from src.models import UserContext
        
        context = Context(
            user_context=UserContext(user_id="test_user"),
            goals=["Short goal"],
            business_data={"key": "value"},
            token_count=500  # Within 1000 limit
        )
        
        summarized = self.summarizer.summarize_context(context, max_tokens=1000)
        
        # Should return original context unchanged
        assert summarized.goals == context.goals
        assert summarized.business_data == context.business_data
        assert summarized.user_context.user_id == context.user_context.user_id
    
    def test_summarize_context_exceeds_limits(self):
        """Test context that exceeds token limits."""
        from src.context_manager import Context
        from src.models import UserContext, Message
        
        # Create context with lots of data
        long_goals = [f"This is a very long goal number {i} " * 20 for i in range(10)]
        long_messages = [
            Message(content=f"This is a very long message {i} " * 30, role="user")
            for i in range(20)
        ]
        
        context = Context(
            user_context=UserContext(user_id="test_user"),
            goals=long_goals,
            chat_history=long_messages,
            business_data={"long_description": "Very long business description " * 100},
            token_count=5000  # Exceeds 1000 limit
        )
        
        summarized = self.summarizer.summarize_context(context, max_tokens=1000)
        
        # Should be summarized
        assert len(summarized.goals) < len(context.goals)
        assert len(summarized.chat_history) < len(context.chat_history)
        assert summarized.token_count < context.token_count
        assert summarized.user_context.user_id == context.user_context.user_id
    
    def test_summarize_goals(self):
        """Test goal summarization."""
        goals = [f"Goal {i}: " + "Long description " * 10 for i in range(10)]
        
        summarized_goals = self.summarizer._summarize_goals(goals, max_tokens=100)
        
        assert len(summarized_goals) < len(goals)
        assert all(goal in goals for goal in summarized_goals)  # Should be subset
    
    def test_summarize_chat_history(self):
        """Test chat history summarization."""
        from src.models import Message
        from datetime import timedelta
        
        base_time = datetime.now()
        messages = [
            Message(
                content=f"Message {i}: " + "Content " * 20,
                role="user" if i % 2 == 0 else "assistant",
                timestamp=base_time + timedelta(minutes=i)
            )
            for i in range(20)
        ]
        
        summarized_history = self.summarizer._summarize_chat_history(messages, max_tokens=200)
        
        assert len(summarized_history) < len(messages)
        # Should keep most recent messages
        if summarized_history:
            latest_original = max(messages, key=lambda m: m.timestamp)
            latest_summarized = max(summarized_history, key=lambda m: m.timestamp)
            assert latest_summarized.timestamp == latest_original.timestamp
    
    def test_token_allocation(self):
        """Test token allocation to different sections."""
        allocation = self.summarizer._allocate_tokens(1000)
        
        assert isinstance(allocation, dict)
        assert "goals" in allocation
        assert "business_data" in allocation
        assert "chat_history" in allocation
        assert "guard_rails" in allocation
        
        # Should sum to approximately the total (allowing for rounding)
        total_allocated = sum(allocation.values())
        assert 950 <= total_allocated <= 1000  # Allow for rounding differences
    
    def test_estimate_section_tokens(self):
        """Test token estimation for different data types."""
        # String
        string_tokens = self.summarizer._estimate_section_tokens("Hello world")
        assert string_tokens > 0
        
        # List
        list_tokens = self.summarizer._estimate_section_tokens(["item1", "item2"])
        assert list_tokens > 0
        
        # Dict
        dict_tokens = self.summarizer._estimate_section_tokens({"key": "value"})
        assert dict_tokens > 0