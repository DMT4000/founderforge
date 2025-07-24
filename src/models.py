"""
Core data models for FounderForge AI Cofounder.
Defines dataclasses for UserContext, Memory, WorkflowResult, and Response with validation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from enum import Enum
import json
import uuid
import re


class MemoryType(Enum):
    """Enumeration for memory types."""
    SHORT_TERM = "SHORT_TERM"
    LONG_TERM = "LONG_TERM"


class ValidationError(Exception):
    """Custom exception for data validation errors."""
    pass


@dataclass
class TokenUsage:
    """Token usage tracking information."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    def __post_init__(self):
        """Validate token usage data."""
        if any(val < 0 for val in [self.prompt_tokens, self.completion_tokens, self.total_tokens]):
            raise ValidationError("Token counts cannot be negative")
        
        if self.total_tokens != self.prompt_tokens + self.completion_tokens:
            self.total_tokens = self.prompt_tokens + self.completion_tokens


@dataclass
class BusinessInfo:
    """Business information for user context."""
    company_name: str = ""
    industry: str = ""
    stage: str = ""  # e.g., "idea", "mvp", "growth", "scale"
    funding_status: str = ""
    team_size: int = 0
    description: str = ""
    
    def __post_init__(self):
        """Validate business information."""
        if self.team_size < 0:
            raise ValidationError("Team size cannot be negative")


@dataclass
class UserPreferences:
    """User preferences and settings."""
    communication_style: str = "professional"  # professional, casual, technical
    response_length: str = "medium"  # short, medium, long
    focus_areas: List[str] = field(default_factory=list)
    timezone: str = "UTC"
    language: str = "en"
    
    def __post_init__(self):
        """Validate user preferences."""
        valid_styles = ["professional", "casual", "technical"]
        if self.communication_style not in valid_styles:
            raise ValidationError(f"Communication style must be one of: {valid_styles}")
        
        valid_lengths = ["short", "medium", "long"]
        if self.response_length not in valid_lengths:
            raise ValidationError(f"Response length must be one of: {valid_lengths}")


@dataclass
class Message:
    """Individual message in conversation history."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    role: str = "user"  # user, assistant, system
    timestamp: datetime = field(default_factory=datetime.now)
    token_count: int = 0
    
    def __post_init__(self):
        """Validate message data."""
        if not self.content.strip():
            raise ValidationError("Message content cannot be empty")
        
        valid_roles = ["user", "assistant", "system"]
        if self.role not in valid_roles:
            raise ValidationError(f"Message role must be one of: {valid_roles}")
        
        if self.token_count < 0:
            raise ValidationError("Token count cannot be negative")


@dataclass
class UserContext:
    """Complete user context for AI interactions."""
    user_id: str
    goals: List[str] = field(default_factory=list)
    business_info: BusinessInfo = field(default_factory=BusinessInfo)
    chat_history: List[Message] = field(default_factory=list)
    preferences: UserPreferences = field(default_factory=UserPreferences)
    token_count: int = 0
    
    def __post_init__(self):
        """Validate user context data."""
        if not self.user_id or not self.user_id.strip():
            raise ValidationError("User ID cannot be empty")
        
        if self.token_count < 0:
            raise ValidationError("Token count cannot be negative")
    
    def add_message(self, content: str, role: str = "user") -> None:
        """Add a message to chat history."""
        message = Message(content=content, role=role)
        self.chat_history.append(message)
        self.token_count += message.token_count
    
    def get_recent_messages(self, limit: int = 10) -> List[Message]:
        """Get recent messages from chat history."""
        return sorted(self.chat_history, key=lambda m: m.timestamp, reverse=True)[:limit]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "user_id": self.user_id,
            "goals": self.goals,
            "business_info": {
                "company_name": self.business_info.company_name,
                "industry": self.business_info.industry,
                "stage": self.business_info.stage,
                "funding_status": self.business_info.funding_status,
                "team_size": self.business_info.team_size,
                "description": self.business_info.description
            },
            "preferences": {
                "communication_style": self.preferences.communication_style,
                "response_length": self.preferences.response_length,
                "focus_areas": self.preferences.focus_areas,
                "timezone": self.preferences.timezone,
                "language": self.preferences.language
            },
            "token_count": self.token_count,
            "chat_history": [
                {
                    "id": msg.id,
                    "content": msg.content,
                    "role": msg.role,
                    "timestamp": msg.timestamp.isoformat(),
                    "token_count": msg.token_count
                }
                for msg in self.chat_history
            ]
        }


@dataclass
class Memory:
    """Memory storage unit for user information."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    content: str = ""
    memory_type: MemoryType = MemoryType.SHORT_TERM
    confidence: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate memory data."""
        if not self.user_id or not self.user_id.strip():
            raise ValidationError("User ID cannot be empty")
        
        if not self.content or not self.content.strip():
            raise ValidationError("Memory content cannot be empty")
        
        if not 0.0 <= self.confidence <= 1.0:
            raise ValidationError("Confidence must be between 0.0 and 1.0")
        
        if self.expires_at and self.expires_at <= self.created_at:
            raise ValidationError("Expiration date must be after creation date")
    
    def is_expired(self) -> bool:
        """Check if memory has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None
        }


@dataclass
class AgentLog:
    """Log entry for agent execution."""
    agent_name: str
    action: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    execution_time: float
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = True
    error_message: str = ""
    
    def __post_init__(self):
        """Validate agent log data."""
        if not self.agent_name or not self.agent_name.strip():
            raise ValidationError("Agent name cannot be empty")
        
        if not self.action or not self.action.strip():
            raise ValidationError("Action cannot be empty")
        
        if self.execution_time < 0:
            raise ValidationError("Execution time cannot be negative")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "agent_name": self.agent_name,
            "action": self.action,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp.isoformat(),
            "success": self.success,
            "error_message": self.error_message
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentLog':
        """Create AgentLog from dictionary."""
        timestamp_str = data.get("timestamp")
        timestamp = datetime.fromisoformat(timestamp_str) if timestamp_str else datetime.now()
        
        return cls(
            agent_name=data.get("agent_name", "unknown"),
            action=data.get("action", "unknown"),
            input_data=data.get("input_data", {}),
            output_data=data.get("output_data", {}),
            execution_time=data.get("execution_time", 0.0),
            timestamp=timestamp,
            success=data.get("success", True),
            error_message=data.get("error_message", "")
        )


@dataclass
class WorkflowResult:
    """Result of multi-agent workflow execution."""
    success: bool
    result_data: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    agent_logs: List[AgentLog] = field(default_factory=list)
    confidence_score: float = 0.0
    workflow_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate workflow result data."""
        if self.execution_time < 0:
            raise ValidationError("Execution time cannot be negative")
        
        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValidationError("Confidence score must be between 0.0 and 1.0")
    
    def add_agent_log(self, log: AgentLog) -> None:
        """Add an agent log to the workflow result."""
        self.agent_logs.append(log)
    
    def get_total_execution_time(self) -> float:
        """Calculate total execution time from agent logs."""
        return sum(log.execution_time for log in self.agent_logs)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "workflow_id": self.workflow_id,
            "success": self.success,
            "result_data": self.result_data,
            "execution_time": self.execution_time,
            "confidence_score": self.confidence_score,
            "timestamp": self.timestamp.isoformat(),
            "agent_logs": [
                {
                    "agent_name": log.agent_name,
                    "action": log.action,
                    "input_data": log.input_data,
                    "output_data": log.output_data,
                    "execution_time": log.execution_time,
                    "timestamp": log.timestamp.isoformat(),
                    "success": log.success,
                    "error_message": log.error_message
                }
                for log in self.agent_logs
            ]
        }


@dataclass
class Response:
    """AI response with metadata and confidence information."""
    content: str
    confidence: float = 0.0
    sources: List[str] = field(default_factory=list)
    fallback_used: bool = False
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    response_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    processing_time: float = 0.0
    
    def __post_init__(self):
        """Validate response data."""
        if not self.content or not self.content.strip():
            raise ValidationError("Response content cannot be empty")
        
        if not 0.0 <= self.confidence <= 1.0:
            raise ValidationError("Confidence must be between 0.0 and 1.0")
        
        if self.processing_time < 0:
            raise ValidationError("Processing time cannot be negative")
    
    def is_high_confidence(self, threshold: float = 0.8) -> bool:
        """Check if response meets confidence threshold."""
        return self.confidence >= threshold
    
    def add_source(self, source: str) -> None:
        """Add a source reference to the response."""
        if source and source not in self.sources:
            self.sources.append(source)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "response_id": self.response_id,
            "content": self.content,
            "confidence": self.confidence,
            "sources": self.sources,
            "fallback_used": self.fallback_used,
            "timestamp": self.timestamp.isoformat(),
            "processing_time": self.processing_time,
            "token_usage": {
                "prompt_tokens": self.token_usage.prompt_tokens,
                "completion_tokens": self.token_usage.completion_tokens,
                "total_tokens": self.token_usage.total_tokens
            }
        }


# Utility functions for model validation and serialization

def validate_email(email: str) -> bool:
    """Validate email format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def validate_user_id(user_id: str) -> bool:
    """Validate user ID format."""
    if not user_id or not user_id.strip():
        return False
    # Allow alphanumeric, hyphens, and underscores
    pattern = r'^[a-zA-Z0-9_-]+$'
    return bool(re.match(pattern, user_id))


def serialize_datetime(dt: datetime) -> str:
    """Serialize datetime to ISO format string."""
    return dt.isoformat()


def deserialize_datetime(dt_str: str) -> datetime:
    """Deserialize ISO format string to datetime."""
    return datetime.fromisoformat(dt_str)


def create_user_context_from_dict(data: Dict[str, Any]) -> UserContext:
    """Create UserContext from dictionary data."""
    business_info = BusinessInfo(**data.get("business_info", {}))
    preferences = UserPreferences(**data.get("preferences", {}))
    
    chat_history = []
    for msg_data in data.get("chat_history", []):
        message = Message(
            id=msg_data.get("id", str(uuid.uuid4())),
            content=msg_data["content"],
            role=msg_data.get("role", "user"),
            timestamp=deserialize_datetime(msg_data["timestamp"]),
            token_count=msg_data.get("token_count", 0)
        )
        chat_history.append(message)
    
    return UserContext(
        user_id=data["user_id"],
        goals=data.get("goals", []),
        business_info=business_info,
        chat_history=chat_history,
        preferences=preferences,
        token_count=data.get("token_count", 0)
    )


def create_memory_from_dict(data: Dict[str, Any]) -> Memory:
    """Create Memory from dictionary data."""
    return Memory(
        id=data.get("id", str(uuid.uuid4())),
        user_id=data["user_id"],
        content=data["content"],
        memory_type=MemoryType(data["memory_type"]),
        confidence=data["confidence"],
        created_at=deserialize_datetime(data["created_at"]),
        expires_at=deserialize_datetime(data["expires_at"]) if data.get("expires_at") else None
    )