"""
Context management system for FounderForge AI Cofounder.
Handles context assembly from multiple local sources, token management, and summarization.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import time

try:
    from .models import UserContext, Message, BusinessInfo, UserPreferences, TokenUsage
    from .database import get_db_manager
    from .logging_manager import get_logging_manager, LogLevel, LogCategory, log_performance, log_audit
    from config.settings import settings
except ImportError:
    from models import UserContext, Message, BusinessInfo, UserPreferences, TokenUsage
    from database import get_db_manager
    from logging_manager import get_logging_manager, LogLevel, LogCategory, log_performance, log_audit
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config.settings import settings


@dataclass
class Context:
    """Assembled context for AI interactions."""
    user_context: UserContext
    business_data: Dict[str, Any] = field(default_factory=dict)
    chat_history: List[Message] = field(default_factory=list)
    goals: List[str] = field(default_factory=list)
    guard_rails: List[str] = field(default_factory=list)
    token_count: int = 0
    priority_data: Dict[str, Any] = field(default_factory=dict)
    sources: List[str] = field(default_factory=list)
    
    def add_source(self, source: str) -> None:
        """Add a source reference to the context."""
        if source and source not in self.sources:
            self.sources.append(source)
    
    def to_prompt_string(self) -> str:
        """Convert context to formatted prompt string."""
        prompt_parts = []
        
        # Priority data first (critical information)
        if self.priority_data:
            prompt_parts.append("=== CRITICAL INFORMATION ===")
            for key, value in self.priority_data.items():
                prompt_parts.append(f"{key}: {value}")
            prompt_parts.append("")
        
        # User goals and preferences
        if self.goals:
            prompt_parts.append("=== USER GOALS ===")
            for i, goal in enumerate(self.goals, 1):
                prompt_parts.append(f"{i}. {goal}")
            prompt_parts.append("")
        
        # Business information
        if self.business_data:
            prompt_parts.append("=== BUSINESS CONTEXT ===")
            for key, value in self.business_data.items():
                if value:  # Only include non-empty values
                    prompt_parts.append(f"{key}: {value}")
            prompt_parts.append("")
        
        # Recent chat history
        if self.chat_history:
            prompt_parts.append("=== RECENT CONVERSATION ===")
            for msg in self.chat_history[-10:]:  # Last 10 messages
                role_prefix = "User" if msg.role == "user" else "Assistant"
                prompt_parts.append(f"{role_prefix}: {msg.content}")
            prompt_parts.append("")
        
        # Guard rails and constraints
        if self.guard_rails:
            prompt_parts.append("=== GUIDELINES ===")
            for rail in self.guard_rails:
                prompt_parts.append(f"- {rail}")
            prompt_parts.append("")
        
        return "\n".join(prompt_parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for serialization."""
        return {
            "user_id": self.user_context.user_id,
            "goals": self.goals,
            "business_data": self.business_data,
            "chat_history": [
                {"role": msg.role, "content": msg.content, "timestamp": msg.timestamp.isoformat() if hasattr(msg, 'timestamp') and msg.timestamp else None}
                for msg in self.chat_history
            ],
            "guard_rails": self.guard_rails,
            "priority_data": self.priority_data,
            "token_count": self.token_count,
            "sources": self.sources,
            "user_preferences": {
                "communication_style": self.user_context.preferences.communication_style,
                "response_length": self.user_context.preferences.response_length,
                "focus_areas": self.user_context.preferences.focus_areas
            } if self.user_context.preferences else {}
        }


class ContextAssembler:
    """Assembles context from multiple local sources."""
    
    def __init__(self):
        self.logging_manager = get_logging_manager()
        self.logger = self.logging_manager.get_logger("context_assembler", LogCategory.CONTEXT)
        self.db_manager = get_db_manager()
        self.data_dir = settings.data_dir
        
        # Ensure data directories exist
        self.chat_history_dir = self.data_dir / "chat_history"
        self.business_data_dir = self.data_dir / "business_data"
        self.guard_rails_dir = self.data_dir / "guard_rails"
        
        for directory in [self.chat_history_dir, self.business_data_dir, self.guard_rails_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def assemble_context(self, user_id: str, query: str = "") -> Context:
        """
        Assemble complete context from multiple local sources.
        
        Args:
            user_id: User identifier
            query: Current user query for context relevance
            
        Returns:
            Assembled context object
        """
        start_time = time.time()
        
        try:
            # Log audit event for data access
            log_audit(user_id, "context_assembly", "user_data", True, 
                     metadata={"query_length": len(query)})
            
            # Get user context from database
            user_context = self._get_user_context(user_id)
            
            # Load chat history from JSON files
            chat_history = self._load_chat_history(user_id)
            
            # Load business data from text files
            business_data = self._load_business_data(user_id)
            
            # Load guard rails and constraints
            guard_rails = self._load_guard_rails()
            
            # Extract priority data (critical information)
            priority_data = self._extract_priority_data(user_context, business_data)
            
            # Create assembled context
            context = Context(
                user_context=user_context,
                business_data=business_data,
                chat_history=chat_history,
                goals=user_context.goals,
                guard_rails=guard_rails,
                priority_data=priority_data
            )
            
            # Add source references
            context.add_source(f"user_profile:{user_id}")
            context.add_source(f"chat_history:{user_id}")
            context.add_source("business_data")
            context.add_source("guard_rails")
            
            # Calculate token count
            context.token_count = self._estimate_token_count(context)
            
            execution_time = time.time() - start_time
            
            # Log performance metrics
            log_performance("context_assembler", "assemble_context", execution_time, True,
                          user_id=user_id, metadata={
                              "token_count": context.token_count,
                              "sources_count": len(context.sources),
                              "chat_history_length": len(chat_history)
                          })
            
            self.logging_manager.log_structured(
                LogLevel.INFO, LogCategory.CONTEXT, "context_assembler",
                f"Context assembled for user {user_id}: {context.token_count} tokens",
                user_id=user_id, execution_time=execution_time,
                metadata={"token_count": context.token_count, "sources": context.sources}
            )
            
            return context
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Log error with performance data
            log_performance("context_assembler", "assemble_context", execution_time, False,
                          user_id=user_id, metadata={"error": str(e)})
            
            self.logging_manager.log_structured(
                LogLevel.ERROR, LogCategory.CONTEXT, "context_assembler",
                f"Context assembly failed for user {user_id}: {e}",
                user_id=user_id, execution_time=execution_time,
                error_details={"exception": str(e), "type": type(e).__name__}
            )
            
            # Return minimal context on error
            return Context(
                user_context=UserContext(user_id=user_id),
                guard_rails=["Respond helpfully and safely"]
            )
    
    def _get_user_context(self, user_id: str) -> UserContext:
        """Get user context from database."""
        try:
            # Query user information
            user_query = "SELECT * FROM users WHERE id = ?"
            user_result = self.db_manager.execute_query(user_query, (user_id,))
            
            if not user_result:
                # Create new user context
                return UserContext(user_id=user_id)
            
            user_row = user_result[0]
            
            # Parse preferences from JSON
            preferences_data = {}
            if user_row['preferences']:
                try:
                    preferences_data = json.loads(user_row['preferences'])
                except json.JSONDecodeError:
                    self.logger.warning(f"Invalid preferences JSON for user {user_id}")
            
            preferences = UserPreferences(**preferences_data)
            
            # Get recent memories for goals
            memory_query = """
                SELECT content FROM memories 
                WHERE user_id = ? AND memory_type = 'LONG_TERM'
                ORDER BY created_at DESC LIMIT 5
            """
            memory_results = self.db_manager.execute_query(memory_query, (user_id,))
            goals = [row['content'] for row in (memory_results or [])]
            
            return UserContext(
                user_id=user_id,
                goals=goals,
                preferences=preferences
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get user context for {user_id}: {e}")
            return UserContext(user_id=user_id)
    
    def _load_chat_history(self, user_id: str) -> List[Message]:
        """Load chat history from JSON files."""
        try:
            chat_file = self.chat_history_dir / f"{user_id}_history.json"
            
            if not chat_file.exists():
                return []
            
            with open(chat_file, 'r', encoding='utf-8') as f:
                history_data = json.load(f)
            
            messages = []
            for msg_data in history_data.get('messages', []):
                try:
                    message = Message(
                        id=msg_data.get('id', ''),
                        content=msg_data.get('content', ''),
                        role=msg_data.get('role', 'user'),
                        timestamp=datetime.fromisoformat(msg_data.get('timestamp', datetime.now().isoformat())),
                        token_count=msg_data.get('token_count', 0)
                    )
                    messages.append(message)
                except Exception as e:
                    self.logger.warning(f"Skipping invalid message in history: {e}")
                    continue
            
            # Sort by timestamp and return recent messages
            messages.sort(key=lambda m: m.timestamp)
            return messages[-20:]  # Last 20 messages
            
        except Exception as e:
            self.logger.error(f"Failed to load chat history for {user_id}: {e}")
            return []
    
    def _load_business_data(self, user_id: str) -> Dict[str, Any]:
        """Load business data from text files."""
        business_data = {}
        
        try:
            # Load user-specific business data
            user_business_file = self.business_data_dir / f"{user_id}_business.txt"
            if user_business_file.exists():
                with open(user_business_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        business_data['user_business_info'] = content
            
            # Load general business templates and data
            for data_file in self.business_data_dir.glob("*.txt"):
                if not data_file.name.startswith(user_id):
                    try:
                        with open(data_file, 'r', encoding='utf-8') as f:
                            content = f.read().strip()
                            if content:
                                key = data_file.stem.replace('_', ' ').title()
                                business_data[key] = content
                    except Exception as e:
                        self.logger.warning(f"Failed to load business data from {data_file}: {e}")
            
            # Load JSON business data files
            for json_file in self.business_data_dir.glob("*.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                        key = json_file.stem.replace('_', ' ').title()
                        business_data[key] = json_data
                except Exception as e:
                    self.logger.warning(f"Failed to load JSON business data from {json_file}: {e}")
            
        except Exception as e:
            self.logger.error(f"Failed to load business data: {e}")
        
        return business_data
    
    def _load_guard_rails(self) -> List[str]:
        """Load guard rails and constraints from files."""
        guard_rails = []
        
        try:
            # Load default guard rails
            default_rails = [
                "Provide helpful, accurate, and relevant information",
                "Maintain professional and supportive tone",
                "Respect user privacy and confidentiality",
                "Avoid providing financial or legal advice without disclaimers",
                "Focus on actionable insights and practical guidance"
            ]
            guard_rails.extend(default_rails)
            
            # Load custom guard rails from files
            for rail_file in self.guard_rails_dir.glob("*.txt"):
                try:
                    with open(rail_file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:
                            # Split by lines and add non-empty lines
                            lines = [line.strip() for line in content.split('\n') if line.strip()]
                            guard_rails.extend(lines)
                except Exception as e:
                    self.logger.warning(f"Failed to load guard rails from {rail_file}: {e}")
            
        except Exception as e:
            self.logger.error(f"Failed to load guard rails: {e}")
        
        return guard_rails
    
    def _extract_priority_data(self, user_context: UserContext, business_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract critical information that should be prioritized in context."""
        priority_data = {}
        
        # User preferences for communication style
        if user_context.preferences.communication_style:
            priority_data['Communication Style'] = user_context.preferences.communication_style
        
        if user_context.preferences.response_length:
            priority_data['Response Length'] = user_context.preferences.response_length
        
        # Critical business information
        if 'user_business_info' in business_data:
            # Extract key business details from user business info
            business_info = business_data['user_business_info']
            if len(business_info) > 200:  # If long, summarize key points
                priority_data['Key Business Info'] = business_info[:200] + "..."
            else:
                priority_data['Key Business Info'] = business_info
        
        # Current focus areas
        if user_context.preferences.focus_areas:
            priority_data['Focus Areas'] = ", ".join(user_context.preferences.focus_areas)
        
        return priority_data
    
    def _estimate_token_count(self, context: Context) -> int:
        """Estimate token count for the assembled context."""
        # Simple estimation: ~4 characters per token
        prompt_string = context.to_prompt_string()
        estimated_tokens = len(prompt_string) // 4
        
        # Add some buffer for formatting and structure
        return int(estimated_tokens * 1.2)
    
    def save_chat_message(self, user_id: str, message: Message) -> bool:
        """Save a chat message to the user's history file."""
        try:
            chat_file = self.chat_history_dir / f"{user_id}_history.json"
            
            # Load existing history
            history_data = {'messages': []}
            if chat_file.exists():
                with open(chat_file, 'r', encoding='utf-8') as f:
                    history_data = json.load(f)
            
            # Add new message
            message_data = {
                'id': message.id,
                'content': message.content,
                'role': message.role,
                'timestamp': message.timestamp.isoformat(),
                'token_count': message.token_count
            }
            history_data['messages'].append(message_data)
            
            # Keep only recent messages (last 100)
            history_data['messages'] = history_data['messages'][-100:]
            
            # Save updated history
            with open(chat_file, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save chat message for {user_id}: {e}")
            return False
    
    def save_business_data(self, user_id: str, data: str, data_type: str = "general") -> bool:
        """Save business data to a text file."""
        try:
            filename = f"{user_id}_{data_type}.txt"
            business_file = self.business_data_dir / filename
            
            with open(business_file, 'w', encoding='utf-8') as f:
                f.write(data)
            
            self.logger.info(f"Business data saved for user {user_id}: {data_type}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save business data for {user_id}: {e}")
            return False


class TokenManager:
    """Manages token usage monitoring and logging to local files."""
    
    def __init__(self):
        self.logging_manager = get_logging_manager()
        self.logger = self.logging_manager.get_logger("token_manager", LogCategory.TOKEN)
        self.data_dir = settings.data_dir
        self.token_logs_dir = self.data_dir / "token_logs"
        self.token_logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Current session token tracking
        self.session_usage = TokenUsage()
        self.daily_usage = {}  # Track daily usage by user
    
    def log_token_usage(self, user_id: str, usage: TokenUsage, operation: str = "general") -> None:
        """
        Log token usage to local files for monitoring.
        
        Args:
            user_id: User identifier
            usage: Token usage information
            operation: Type of operation (context_assembly, summarization, etc.)
        """
        try:
            # Update session usage
            self.session_usage.prompt_tokens += usage.prompt_tokens
            self.session_usage.completion_tokens += usage.completion_tokens
            self.session_usage.total_tokens += usage.total_tokens
            
            # Update daily usage
            today = datetime.now().strftime("%Y-%m-%d")
            if user_id not in self.daily_usage:
                self.daily_usage[user_id] = {}
            if today not in self.daily_usage[user_id]:
                self.daily_usage[user_id][today] = TokenUsage()
            
            daily_usage = self.daily_usage[user_id][today]
            daily_usage.prompt_tokens += usage.prompt_tokens
            daily_usage.completion_tokens += usage.completion_tokens
            daily_usage.total_tokens += usage.total_tokens
            
            # Log to file
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "user_id": user_id,
                "operation": operation,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens
            }
            
            # Daily log file
            daily_log_file = self.token_logs_dir / f"tokens_{today}.jsonl"
            with open(daily_log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + '\n')
            
            # User-specific log file
            user_log_file = self.token_logs_dir / f"user_{user_id}_tokens.jsonl"
            with open(user_log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + '\n')
            
            self.logger.debug(f"Token usage logged for {user_id}: {usage.total_tokens} tokens ({operation})")
            
        except Exception as e:
            self.logger.error(f"Failed to log token usage for {user_id}: {e}")
    
    def get_session_usage(self) -> TokenUsage:
        """Get current session token usage."""
        return self.session_usage
    
    def get_daily_usage(self, user_id: str, date: str = None) -> Optional[TokenUsage]:
        """
        Get daily token usage for a user.
        
        Args:
            user_id: User identifier
            date: Date in YYYY-MM-DD format (defaults to today)
            
        Returns:
            Token usage for the specified date or None if no data
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        return self.daily_usage.get(user_id, {}).get(date)
    
    def get_usage_summary(self, user_id: str, days: int = 7) -> Dict[str, Any]:
        """
        Get token usage summary for the last N days.
        
        Args:
            user_id: User identifier
            days: Number of days to include in summary
            
        Returns:
            Usage summary with daily breakdown and totals
        """
        try:
            summary = {
                "user_id": user_id,
                "period_days": days,
                "daily_usage": {},
                "total_usage": TokenUsage(),
                "average_daily": TokenUsage()
            }
            
            # Calculate date range
            from datetime import timedelta
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days-1)
            
            # Collect daily usage
            current_date = start_date
            while current_date <= end_date:
                date_str = current_date.strftime("%Y-%m-%d")
                daily_usage = self.get_daily_usage(user_id, date_str)
                
                if daily_usage:
                    summary["daily_usage"][date_str] = {
                        "prompt_tokens": daily_usage.prompt_tokens,
                        "completion_tokens": daily_usage.completion_tokens,
                        "total_tokens": daily_usage.total_tokens
                    }
                    
                    # Add to totals
                    summary["total_usage"].prompt_tokens += daily_usage.prompt_tokens
                    summary["total_usage"].completion_tokens += daily_usage.completion_tokens
                    summary["total_usage"].total_tokens += daily_usage.total_tokens
                else:
                    summary["daily_usage"][date_str] = {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                    }
                
                current_date += timedelta(days=1)
            
            # Calculate averages
            if days > 0:
                summary["average_daily"].prompt_tokens = summary["total_usage"].prompt_tokens // days
                summary["average_daily"].completion_tokens = summary["total_usage"].completion_tokens // days
                summary["average_daily"].total_tokens = summary["total_usage"].total_tokens // days
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to generate usage summary for {user_id}: {e}")
            return {"error": str(e)}
    
    def check_usage_limits(self, user_id: str, proposed_tokens: int) -> Dict[str, Any]:
        """
        Check if proposed token usage would exceed limits.
        
        Args:
            user_id: User identifier
            proposed_tokens: Number of tokens for proposed operation
            
        Returns:
            Dictionary with limit check results and recommendations
        """
        try:
            # Get current daily usage
            today_usage = self.get_daily_usage(user_id) or TokenUsage()
            
            # Define limits (can be made configurable)
            daily_limit = settings.get_feature_flag("daily_token_limit", 50000)
            context_limit = settings.get_feature_flag("max_context_tokens", 16000)
            
            result = {
                "within_daily_limit": (today_usage.total_tokens + proposed_tokens) <= daily_limit,
                "within_context_limit": proposed_tokens <= context_limit,
                "current_daily_usage": today_usage.total_tokens,
                "daily_limit": daily_limit,
                "context_limit": context_limit,
                "proposed_tokens": proposed_tokens,
                "remaining_daily": max(0, daily_limit - today_usage.total_tokens),
                "recommendations": []
            }
            
            # Add recommendations based on usage
            if not result["within_daily_limit"]:
                result["recommendations"].append("Daily token limit would be exceeded. Consider summarizing context or deferring non-critical operations.")
            
            if not result["within_context_limit"]:
                result["recommendations"].append("Context size exceeds limit. Apply summarization to reduce token count.")
            
            if today_usage.total_tokens > (daily_limit * 0.8):
                result["recommendations"].append("Approaching daily limit. Monitor usage closely.")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to check usage limits for {user_id}: {e}")
            return {"error": str(e)}


class ContextSummarizer:
    """Handles context summarization to keep under token limits."""
    
    def __init__(self, token_manager: TokenManager = None):
        self.logger = logging.getLogger(__name__)
        self.token_manager = token_manager or TokenManager()
        self.max_context_tokens = settings.get_feature_flag("max_context_tokens", 16000)
    
    def summarize_context(self, context: Context, max_tokens: int = None) -> Context:
        """
        Summarize context to fit within token limits.
        
        Args:
            context: Original context to summarize
            max_tokens: Maximum tokens allowed (defaults to configured limit)
            
        Returns:
            Summarized context within token limits
        """
        if max_tokens is None:
            max_tokens = self.max_context_tokens
        
        try:
            # If context is already within limits, return as-is
            if context.token_count <= max_tokens:
                return context
            
            self.logger.info(f"Summarizing context: {context.token_count} -> target: {max_tokens} tokens")
            
            # Create summarized context
            summarized_context = Context(
                user_context=context.user_context,
                priority_data=context.priority_data,  # Always keep priority data
                sources=context.sources.copy()
            )
            
            # Calculate remaining token budget after priority data
            priority_tokens = self._estimate_section_tokens(summarized_context.priority_data)
            remaining_tokens = max_tokens - priority_tokens - 500  # Buffer for structure
            
            # Allocate tokens to different sections based on importance
            token_allocation = self._allocate_tokens(remaining_tokens)
            
            # Summarize each section according to allocation
            summarized_context.goals = self._summarize_goals(
                context.goals, token_allocation["goals"]
            )
            
            summarized_context.business_data = self._summarize_business_data(
                context.business_data, token_allocation["business_data"]
            )
            
            summarized_context.chat_history = self._summarize_chat_history(
                context.chat_history, token_allocation["chat_history"]
            )
            
            summarized_context.guard_rails = self._summarize_guard_rails(
                context.guard_rails, token_allocation["guard_rails"]
            )
            
            # Recalculate token count
            summarized_context.token_count = self._estimate_context_tokens(summarized_context)
            
            # Log summarization
            self.token_manager.log_token_usage(
                context.user_context.user_id,
                TokenUsage(total_tokens=summarized_context.token_count),
                "context_summarization"
            )
            
            self.logger.info(f"Context summarized: {context.token_count} -> {summarized_context.token_count} tokens")
            return summarized_context
            
        except Exception as e:
            self.logger.error(f"Context summarization failed: {e}")
            # Return minimal context on error
            return Context(
                user_context=context.user_context,
                priority_data=context.priority_data,
                guard_rails=["Provide helpful and safe responses"]
            )
    
    def _allocate_tokens(self, available_tokens: int) -> Dict[str, int]:
        """Allocate available tokens to different context sections."""
        # Priority allocation (percentages)
        allocations = {
            "goals": 0.15,          # 15% for user goals
            "business_data": 0.25,  # 25% for business context
            "chat_history": 0.45,   # 45% for conversation history
            "guard_rails": 0.15     # 15% for guidelines
        }
        
        return {
            section: int(available_tokens * percentage)
            for section, percentage in allocations.items()
        }
    
    def _summarize_goals(self, goals: List[str], max_tokens: int) -> List[str]:
        """Summarize user goals to fit token limit."""
        if not goals:
            return []
        
        # Estimate tokens per goal
        total_content = " ".join(goals)
        current_tokens = len(total_content) // 4
        
        if current_tokens <= max_tokens:
            return goals
        
        # Keep most important goals (first ones are typically most recent/important)
        summarized_goals = []
        token_count = 0
        
        for goal in goals:
            goal_tokens = len(goal) // 4
            if token_count + goal_tokens <= max_tokens:
                summarized_goals.append(goal)
                token_count += goal_tokens
            else:
                break
        
        return summarized_goals
    
    def _summarize_business_data(self, business_data: Dict[str, Any], max_tokens: int) -> Dict[str, Any]:
        """Summarize business data to fit token limit."""
        if not business_data:
            return {}
        
        summarized_data = {}
        token_count = 0
        
        # Prioritize user-specific business info
        priority_keys = ["user_business_info", "Company Stage", "Industry"]
        
        # Add priority data first
        for key in priority_keys:
            if key in business_data and token_count < max_tokens:
                value = business_data[key]
                value_str = str(value)
                value_tokens = len(value_str) // 4
                
                if token_count + value_tokens <= max_tokens:
                    # Truncate if too long
                    if value_tokens > max_tokens // 3:  # Don't let one item take more than 1/3
                        max_chars = (max_tokens // 3) * 4
                        value_str = value_str[:max_chars] + "..."
                    
                    summarized_data[key] = value_str
                    token_count += len(value_str) // 4
        
        # Add remaining data if space allows
        for key, value in business_data.items():
            if key not in priority_keys and token_count < max_tokens:
                value_str = str(value)
                value_tokens = len(value_str) // 4
                
                if token_count + value_tokens <= max_tokens:
                    summarized_data[key] = value_str
                    token_count += value_tokens
        
        return summarized_data
    
    def _summarize_chat_history(self, chat_history: List[Message], max_tokens: int) -> List[Message]:
        """Summarize chat history to fit token limit."""
        if not chat_history:
            return []
        
        # Sort by timestamp (most recent first)
        sorted_history = sorted(chat_history, key=lambda m: m.timestamp, reverse=True)
        
        summarized_history = []
        token_count = 0
        
        for message in sorted_history:
            message_tokens = len(message.content) // 4
            
            if token_count + message_tokens <= max_tokens:
                summarized_history.append(message)
                token_count += message_tokens
            else:
                # Try to fit a truncated version
                remaining_tokens = max_tokens - token_count
                if remaining_tokens > 20:  # Minimum meaningful message size
                    max_chars = remaining_tokens * 4
                    truncated_content = message.content[:max_chars] + "..."
                    
                    truncated_message = Message(
                        id=message.id,
                        content=truncated_content,
                        role=message.role,
                        timestamp=message.timestamp,
                        token_count=remaining_tokens
                    )
                    summarized_history.append(truncated_message)
                break
        
        # Return in chronological order
        return sorted(summarized_history, key=lambda m: m.timestamp)
    
    def _summarize_guard_rails(self, guard_rails: List[str], max_tokens: int) -> List[str]:
        """Summarize guard rails to fit token limit."""
        if not guard_rails:
            return []
        
        # Keep essential guard rails
        essential_rails = [
            "Provide helpful, accurate, and relevant information",
            "Maintain professional and supportive tone",
            "Respect user privacy and confidentiality"
        ]
        
        summarized_rails = []
        token_count = 0
        
        # Add essential rails first
        for rail in essential_rails:
            if rail in guard_rails:
                rail_tokens = len(rail) // 4
                if token_count + rail_tokens <= max_tokens:
                    summarized_rails.append(rail)
                    token_count += rail_tokens
        
        # Add remaining rails if space allows
        for rail in guard_rails:
            if rail not in essential_rails and token_count < max_tokens:
                rail_tokens = len(rail) // 4
                if token_count + rail_tokens <= max_tokens:
                    summarized_rails.append(rail)
                    token_count += rail_tokens
        
        return summarized_rails
    
    def _estimate_section_tokens(self, data: Any) -> int:
        """Estimate token count for a data section."""
        if isinstance(data, str):
            return len(data) // 4
        elif isinstance(data, (list, dict)):
            content_str = str(data)
            return len(content_str) // 4
        else:
            return len(str(data)) // 4
    
    def _estimate_context_tokens(self, context: Context) -> int:
        """Estimate total token count for a context."""
        prompt_string = context.to_prompt_string()
        return int(len(prompt_string) // 4 * 1.2)  # Add buffer for formatting


# Convenience functions for easy access
def create_context_assembler() -> ContextAssembler:
    """Create a new ContextAssembler instance."""
    return ContextAssembler()


def create_token_manager() -> TokenManager:
    """Create a new TokenManager instance."""
    return TokenManager()


def create_context_summarizer(token_manager: TokenManager = None) -> ContextSummarizer:
    """Create a new ContextSummarizer instance."""
    return ContextSummarizer(token_manager)