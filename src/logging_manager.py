"""
FounderForge Logging Manager

Provides centralized logging configuration and management for all system components.
Implements structured logging with file rotation, performance metrics, and audit trails.
"""

import json
import logging
import logging.handlers
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import psutil


class LogLevel(Enum):
    """Log level enumeration"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogCategory(Enum):
    """Log category enumeration for structured logging"""
    SYSTEM = "system"
    AGENT = "agent"
    TOKEN = "token"
    PERFORMANCE = "performance"
    SECURITY = "security"
    AUDIT = "audit"
    ERROR = "error"
    API = "api"
    DATABASE = "database"
    CONTEXT = "context"
    MEMORY = "memory"


@dataclass
class LogEntry:
    """Structured log entry"""
    timestamp: str
    level: str
    category: str
    component: str
    message: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    error_details: Optional[Dict[str, Any]] = None


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    timestamp: str
    component: str
    operation: str
    execution_time: float
    memory_usage: float
    cpu_usage: float
    success: bool
    user_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TokenUsageLog:
    """Token usage logging structure"""
    timestamp: str
    user_id: str
    operation: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_estimate: Optional[float] = None
    model: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AuditLogEntry:
    """Data access audit log entry"""
    timestamp: str
    user_id: str
    action: str
    resource: str
    success: bool
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class LoggingManager:
    """
    Centralized logging manager for FounderForge system.
    
    Provides structured logging with file rotation, performance tracking,
    and audit trails for privacy compliance simulation.
    """
    
    def __init__(self, 
                 log_dir: str = "data/logs",
                 log_level: LogLevel = LogLevel.INFO,
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5,
                 enable_console: bool = True):
        """
        Initialize logging manager.
        
        Args:
            log_dir: Directory for log files
            log_level: Default logging level
            max_file_size: Maximum size per log file before rotation
            backup_count: Number of backup files to keep
            enable_console: Whether to enable console logging
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_level = log_level
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        self.enable_console = enable_console
        
        # Create subdirectories for different log types
        self.system_log_dir = self.log_dir / "system"
        self.agent_log_dir = self.log_dir / "agents"
        self.token_log_dir = self.log_dir / "tokens"
        self.performance_log_dir = self.log_dir / "performance"
        self.security_log_dir = self.log_dir / "security"
        self.audit_log_dir = self.log_dir / "audit"
        
        for log_subdir in [self.system_log_dir, self.agent_log_dir, self.token_log_dir,
                          self.performance_log_dir, self.security_log_dir, self.audit_log_dir]:
            log_subdir.mkdir(parents=True, exist_ok=True)
        
        # Initialize loggers
        self._loggers: Dict[str, logging.Logger] = {}
        self._setup_logging()
        
        # Performance monitoring
        self._performance_lock = threading.Lock()
        self._start_time = time.time()
        
        # System logger
        self.logger = self.get_logger("logging_manager", LogCategory.SYSTEM)
        self.logger.info("Logging manager initialized", extra={
            "log_dir": str(self.log_dir),
            "log_level": log_level.value
        })
    
    def _setup_logging(self):
        """Set up logging configuration"""
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.log_level.value))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        if self.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
        
        # File handler for system logs
        system_log_file = self.system_log_dir / "system.log"
        file_handler = logging.handlers.RotatingFileHandler(
            system_log_file,
            maxBytes=self.max_file_size,
            backupCount=self.backup_count
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    def get_logger(self, name: str, category: LogCategory = LogCategory.SYSTEM) -> logging.Logger:
        """
        Get or create a logger for a specific component.
        
        Args:
            name: Logger name (usually component name)
            category: Log category for organization
            
        Returns:
            Configured logger instance
        """
        logger_key = f"{category.value}.{name}"
        
        if logger_key not in self._loggers:
            logger = logging.getLogger(logger_key)
            
            # Add category-specific file handler
            log_file = self._get_log_file_path(category, name)
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=self.max_file_size,
                backupCount=self.backup_count
            )
            
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            self._loggers[logger_key] = logger
        
        return self._loggers[logger_key]
    
    def _get_log_file_path(self, category: LogCategory, component: str) -> Path:
        """Get log file path for a category and component"""
        date_str = datetime.now().strftime("%Y%m%d")
        
        if category == LogCategory.AGENT:
            return self.agent_log_dir / f"{component}_{date_str}.log"
        elif category == LogCategory.TOKEN:
            return self.token_log_dir / f"tokens_{date_str}.jsonl"
        elif category == LogCategory.PERFORMANCE:
            return self.performance_log_dir / f"performance_{date_str}.jsonl"
        elif category == LogCategory.SECURITY:
            return self.security_log_dir / f"security_{date_str}.log"
        elif category == LogCategory.AUDIT:
            return self.audit_log_dir / f"audit_{date_str}.jsonl"
        else:
            return self.system_log_dir / f"{component}_{date_str}.log"
    
    def log_structured(self, 
                      level: LogLevel,
                      category: LogCategory,
                      component: str,
                      message: str,
                      user_id: Optional[str] = None,
                      session_id: Optional[str] = None,
                      execution_time: Optional[float] = None,
                      metadata: Optional[Dict[str, Any]] = None,
                      error_details: Optional[Dict[str, Any]] = None):
        """
        Log a structured entry.
        
        Args:
            level: Log level
            category: Log category
            component: Component name
            message: Log message
            user_id: User ID if applicable
            session_id: Session ID if applicable
            execution_time: Execution time in seconds
            metadata: Additional metadata
            error_details: Error details if applicable
        """
        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            level=level.value,
            category=category.value,
            component=component,
            message=message,
            user_id=user_id,
            session_id=session_id,
            execution_time=execution_time,
            metadata=metadata,
            error_details=error_details
        )
        
        # Write to structured log file
        structured_log_file = self.log_dir / f"structured_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(structured_log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(asdict(entry)) + '\n')
        
        # Also log to component logger
        logger = self.get_logger(component, category)
        log_method = getattr(logger, level.value.lower())
        
        extra_data = {
            'user_id': user_id,
            'session_id': session_id,
            'execution_time': execution_time,
            'metadata': metadata
        }
        
        log_method(message, extra=extra_data)
    
    def log_token_usage(self, 
                       user_id: str,
                       operation: str,
                       prompt_tokens: int,
                       completion_tokens: int,
                       total_tokens: int,
                       cost_estimate: Optional[float] = None,
                       model: Optional[str] = None,
                       metadata: Optional[Dict[str, Any]] = None):
        """
        Log token usage for monitoring and cost tracking.
        
        Args:
            user_id: User ID
            operation: Operation that consumed tokens
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            total_tokens: Total tokens used
            cost_estimate: Estimated cost
            model: Model used
            metadata: Additional metadata
        """
        entry = TokenUsageLog(
            timestamp=datetime.now().isoformat(),
            user_id=user_id,
            operation=operation,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_estimate=cost_estimate,
            model=model,
            metadata=metadata
        )
        
        # Write to token log file
        token_log_file = self.token_log_dir / f"tokens_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(token_log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(asdict(entry)) + '\n')
        
        # Also write to user-specific token log
        user_token_log = self.token_log_dir / f"user_{user_id}_tokens.jsonl"
        with open(user_token_log, 'a', encoding='utf-8') as f:
            f.write(json.dumps(asdict(entry)) + '\n')
    
    def log_performance_metrics(self,
                               component: str,
                               operation: str,
                               execution_time: float,
                               success: bool,
                               user_id: Optional[str] = None,
                               metadata: Optional[Dict[str, Any]] = None):
        """
        Log performance metrics for monitoring and optimization.
        
        Args:
            component: Component name
            operation: Operation performed
            execution_time: Execution time in seconds
            success: Whether operation was successful
            user_id: User ID if applicable
            metadata: Additional metadata
        """
        with self._performance_lock:
            # Get current system metrics
            process = psutil.Process()
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB
            cpu_usage = process.cpu_percent()
            
            entry = PerformanceMetrics(
                timestamp=datetime.now().isoformat(),
                component=component,
                operation=operation,
                execution_time=execution_time,
                memory_usage=memory_usage,
                cpu_usage=cpu_usage,
                success=success,
                user_id=user_id,
                metadata=metadata
            )
            
            # Write to performance log file
            perf_log_file = self.performance_log_dir / f"performance_{datetime.now().strftime('%Y%m%d')}.jsonl"
            with open(perf_log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(asdict(entry)) + '\n')
    
    def log_audit_event(self,
                       user_id: str,
                       action: str,
                       resource: str,
                       success: bool,
                       ip_address: Optional[str] = None,
                       user_agent: Optional[str] = None,
                       metadata: Optional[Dict[str, Any]] = None):
        """
        Log audit events for privacy compliance simulation.
        
        Args:
            user_id: User ID
            action: Action performed
            resource: Resource accessed
            success: Whether action was successful
            ip_address: IP address if available
            user_agent: User agent if available
            metadata: Additional metadata
        """
        entry = AuditLogEntry(
            timestamp=datetime.now().isoformat(),
            user_id=user_id,
            action=action,
            resource=resource,
            success=success,
            ip_address=ip_address,
            user_agent=user_agent,
            metadata=metadata
        )
        
        # Write to audit log file
        audit_log_file = self.audit_log_dir / f"audit_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(audit_log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(asdict(entry)) + '\n')
        
        # Also log to security logger for critical events
        if action in ['data_access', 'data_deletion', 'user_creation', 'authentication']:
            security_logger = self.get_logger("audit", LogCategory.SECURITY)
            security_logger.info(f"Audit event: {action} on {resource} by {user_id}", extra={
                'user_id': user_id,
                'action': action,
                'resource': resource,
                'success': success
            })
    
    def get_log_stats(self, days: int = 7) -> Dict[str, Any]:
        """
        Get logging statistics for the past N days.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dictionary with logging statistics
        """
        stats = {
            'total_entries': 0,
            'entries_by_level': {},
            'entries_by_category': {},
            'entries_by_component': {},
            'token_usage': {
                'total_tokens': 0,
                'total_cost': 0.0,
                'operations': {}
            },
            'performance_metrics': {
                'avg_execution_time': 0.0,
                'max_execution_time': 0.0,
                'avg_memory_usage': 0.0,
                'max_memory_usage': 0.0
            },
            'audit_events': {
                'total_events': 0,
                'events_by_action': {}
            }
        }
        
        # Analyze structured logs
        cutoff_date = datetime.now() - timedelta(days=days)
        
        for i in range(days):
            date = (datetime.now() - timedelta(days=i)).strftime('%Y%m%d')
            structured_log_file = self.log_dir / f"structured_{date}.jsonl"
            
            if structured_log_file.exists():
                with open(structured_log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            entry = json.loads(line.strip())
                            entry_time = datetime.fromisoformat(entry['timestamp'])
                            
                            if entry_time >= cutoff_date:
                                stats['total_entries'] += 1
                                
                                # Count by level
                                level = entry['level']
                                stats['entries_by_level'][level] = stats['entries_by_level'].get(level, 0) + 1
                                
                                # Count by category
                                category = entry['category']
                                stats['entries_by_category'][category] = stats['entries_by_category'].get(category, 0) + 1
                                
                                # Count by component
                                component = entry['component']
                                stats['entries_by_component'][component] = stats['entries_by_component'].get(component, 0) + 1
                        except (json.JSONDecodeError, KeyError):
                            continue
        
        return stats
    
    def cleanup_old_logs(self, days_to_keep: int = 30):
        """
        Clean up log files older than specified days.
        
        Args:
            days_to_keep: Number of days of logs to keep
        """
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        deleted_files = []
        
        for log_subdir in [self.system_log_dir, self.agent_log_dir, self.token_log_dir,
                          self.performance_log_dir, self.security_log_dir, self.audit_log_dir]:
            for log_file in log_subdir.glob("*"):
                if log_file.is_file():
                    file_date = datetime.fromtimestamp(log_file.stat().st_mtime)
                    if file_date < cutoff_date:
                        log_file.unlink()
                        deleted_files.append(str(log_file))
        
        self.logger.info(f"Cleaned up {len(deleted_files)} old log files", extra={
            'deleted_files': deleted_files,
            'days_to_keep': days_to_keep
        })
        
        return deleted_files


# Global logging manager instance
_logging_manager: Optional[LoggingManager] = None


def get_logging_manager() -> LoggingManager:
    """Get the global logging manager instance"""
    global _logging_manager
    if _logging_manager is None:
        _logging_manager = LoggingManager()
    return _logging_manager


def setup_logging(log_dir: str = "data/logs",
                 log_level: LogLevel = LogLevel.INFO,
                 enable_console: bool = True) -> LoggingManager:
    """
    Set up global logging configuration.
    
    Args:
        log_dir: Directory for log files
        log_level: Default logging level
        enable_console: Whether to enable console logging
        
    Returns:
        Configured logging manager
    """
    global _logging_manager
    _logging_manager = LoggingManager(
        log_dir=log_dir,
        log_level=log_level,
        enable_console=enable_console
    )
    return _logging_manager


# Convenience functions for common logging operations
def log_info(component: str, message: str, **kwargs):
    """Log info message"""
    get_logging_manager().log_structured(
        LogLevel.INFO, LogCategory.SYSTEM, component, message, **kwargs
    )


def log_error(component: str, message: str, **kwargs):
    """Log error message"""
    get_logging_manager().log_structured(
        LogLevel.ERROR, LogCategory.ERROR, component, message, **kwargs
    )


def log_performance(component: str, operation: str, execution_time: float, success: bool, **kwargs):
    """Log performance metrics"""
    get_logging_manager().log_performance_metrics(
        component, operation, execution_time, success, **kwargs
    )


def log_token_usage(user_id: str, operation: str, prompt_tokens: int, completion_tokens: int, total_tokens: int, **kwargs):
    """Log token usage"""
    get_logging_manager().log_token_usage(
        user_id, operation, prompt_tokens, completion_tokens, total_tokens, **kwargs
    )


def log_audit(user_id: str, action: str, resource: str, success: bool, **kwargs):
    """Log audit event"""
    get_logging_manager().log_audit_event(
        user_id, action, resource, success, **kwargs
    )