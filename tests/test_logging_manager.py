"""
Tests for FounderForge Logging Manager
"""

import json
import os
import tempfile
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.logging_manager import (
    LoggingManager, LogLevel, LogCategory, LogEntry, PerformanceMetrics,
    TokenUsageLog, AuditLogEntry, setup_logging, get_logging_manager,
    log_info, log_error, log_performance, log_token_usage, log_audit
)


class TestLoggingManager(unittest.TestCase):
    """Test cases for LoggingManager"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = Path(self.temp_dir) / "test_logs"
        self.logging_manager = LoggingManager(
            log_dir=str(self.log_dir),
            log_level=LogLevel.DEBUG,
            enable_console=False
        )
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test logging manager initialization"""
        self.assertTrue(self.log_dir.exists())
        self.assertTrue((self.log_dir / "system").exists())
        self.assertTrue((self.log_dir / "agents").exists())
        self.assertTrue((self.log_dir / "tokens").exists())
        self.assertTrue((self.log_dir / "performance").exists())
        self.assertTrue((self.log_dir / "security").exists())
        self.assertTrue((self.log_dir / "audit").exists())
    
    def test_get_logger(self):
        """Test logger creation and retrieval"""
        logger1 = self.logging_manager.get_logger("test_component", LogCategory.SYSTEM)
        logger2 = self.logging_manager.get_logger("test_component", LogCategory.SYSTEM)
        
        # Should return the same logger instance
        self.assertIs(logger1, logger2)
        
        # Different category should create different logger
        logger3 = self.logging_manager.get_logger("test_component", LogCategory.AGENT)
        self.assertIsNot(logger1, logger3)
    
    def test_structured_logging(self):
        """Test structured logging functionality"""
        test_metadata = {"key": "value", "number": 42}
        
        self.logging_manager.log_structured(
            level=LogLevel.INFO,
            category=LogCategory.SYSTEM,
            component="test_component",
            message="Test message",
            user_id="test_user",
            session_id="test_session",
            execution_time=1.5,
            metadata=test_metadata
        )
        
        # Check structured log file
        date_str = datetime.now().strftime('%Y%m%d')
        structured_log_file = self.log_dir / f"structured_{date_str}.jsonl"
        
        self.assertTrue(structured_log_file.exists())
        
        with open(structured_log_file, 'r', encoding='utf-8') as f:
            log_entry = json.loads(f.readline().strip())
        
        self.assertEqual(log_entry['level'], 'INFO')
        self.assertEqual(log_entry['category'], 'system')
        self.assertEqual(log_entry['component'], 'test_component')
        self.assertEqual(log_entry['message'], 'Test message')
        self.assertEqual(log_entry['user_id'], 'test_user')
        self.assertEqual(log_entry['session_id'], 'test_session')
        self.assertEqual(log_entry['execution_time'], 1.5)
        self.assertEqual(log_entry['metadata'], test_metadata)
    
    def test_token_usage_logging(self):
        """Test token usage logging"""
        self.logging_manager.log_token_usage(
            user_id="test_user",
            operation="test_operation",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cost_estimate=0.01,
            model="gemini-2.5-flash",
            metadata={"context": "test"}
        )
        
        # Check token log file
        date_str = datetime.now().strftime('%Y%m%d')
        token_log_file = self.log_dir / "tokens" / f"tokens_{date_str}.jsonl"
        
        self.assertTrue(token_log_file.exists())
        
        with open(token_log_file, 'r', encoding='utf-8') as f:
            log_entry = json.loads(f.readline().strip())
        
        self.assertEqual(log_entry['user_id'], 'test_user')
        self.assertEqual(log_entry['operation'], 'test_operation')
        self.assertEqual(log_entry['prompt_tokens'], 100)
        self.assertEqual(log_entry['completion_tokens'], 50)
        self.assertEqual(log_entry['total_tokens'], 150)
        self.assertEqual(log_entry['cost_estimate'], 0.01)
        self.assertEqual(log_entry['model'], 'gemini-2.5-flash')
        
        # Check user-specific token log
        user_token_log = self.log_dir / "tokens" / "user_test_user_tokens.jsonl"
        self.assertTrue(user_token_log.exists())
    
    @patch('psutil.Process')
    def test_performance_metrics_logging(self, mock_process):
        """Test performance metrics logging"""
        # Mock psutil.Process
        mock_process_instance = MagicMock()
        mock_process_instance.memory_info.return_value.rss = 100 * 1024 * 1024  # 100MB
        mock_process_instance.cpu_percent.return_value = 25.5
        mock_process.return_value = mock_process_instance
        
        self.logging_manager.log_performance_metrics(
            component="test_component",
            operation="test_operation",
            execution_time=2.5,
            success=True,
            user_id="test_user",
            metadata={"context": "test"}
        )
        
        # Check performance log file
        date_str = datetime.now().strftime('%Y%m%d')
        perf_log_file = self.log_dir / "performance" / f"performance_{date_str}.jsonl"
        
        self.assertTrue(perf_log_file.exists())
        
        with open(perf_log_file, 'r', encoding='utf-8') as f:
            log_entry = json.loads(f.readline().strip())
        
        self.assertEqual(log_entry['component'], 'test_component')
        self.assertEqual(log_entry['operation'], 'test_operation')
        self.assertEqual(log_entry['execution_time'], 2.5)
        self.assertEqual(log_entry['memory_usage'], 100.0)
        self.assertEqual(log_entry['cpu_usage'], 25.5)
        self.assertTrue(log_entry['success'])
        self.assertEqual(log_entry['user_id'], 'test_user')
    
    def test_audit_logging(self):
        """Test audit event logging"""
        self.logging_manager.log_audit_event(
            user_id="test_user",
            action="data_access",
            resource="user_profile",
            success=True,
            ip_address="127.0.0.1",
            user_agent="test_agent",
            metadata={"context": "test"}
        )
        
        # Check audit log file
        date_str = datetime.now().strftime('%Y%m%d')
        audit_log_file = self.log_dir / "audit" / f"audit_{date_str}.jsonl"
        
        self.assertTrue(audit_log_file.exists())
        
        with open(audit_log_file, 'r', encoding='utf-8') as f:
            log_entry = json.loads(f.readline().strip())
        
        self.assertEqual(log_entry['user_id'], 'test_user')
        self.assertEqual(log_entry['action'], 'data_access')
        self.assertEqual(log_entry['resource'], 'user_profile')
        self.assertTrue(log_entry['success'])
        self.assertEqual(log_entry['ip_address'], '127.0.0.1')
        self.assertEqual(log_entry['user_agent'], 'test_agent')
    
    def test_log_stats(self):
        """Test log statistics generation"""
        # Generate some test logs
        for i in range(5):
            self.logging_manager.log_structured(
                LogLevel.INFO, LogCategory.SYSTEM, "test_component", f"Message {i}"
            )
        
        for i in range(3):
            self.logging_manager.log_structured(
                LogLevel.ERROR, LogCategory.ERROR, "test_component", f"Error {i}"
            )
        
        stats = self.logging_manager.get_log_stats(days=1)
        
        self.assertEqual(stats['total_entries'], 8)
        self.assertEqual(stats['entries_by_level']['INFO'], 5)
        self.assertEqual(stats['entries_by_level']['ERROR'], 3)
        self.assertEqual(stats['entries_by_category']['system'], 5)
        self.assertEqual(stats['entries_by_category']['error'], 3)
    
    def test_cleanup_old_logs(self):
        """Test log cleanup functionality"""
        # Create some test log files with old timestamps
        old_log_file = self.log_dir / "system" / "old_log.log"
        old_log_file.touch()
        
        # Set file modification time to 35 days ago
        old_time = time.time() - (35 * 24 * 60 * 60)
        os.utime(old_log_file, (old_time, old_time))
        
        # Create a recent log file
        recent_log_file = self.log_dir / "system" / "recent_log.log"
        recent_log_file.touch()
        
        # Clean up logs older than 30 days
        deleted_files = self.logging_manager.cleanup_old_logs(days_to_keep=30)
        
        self.assertFalse(old_log_file.exists())
        self.assertTrue(recent_log_file.exists())
        self.assertEqual(len(deleted_files), 1)


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = Path(self.temp_dir) / "test_logs"
        
        # Set up global logging manager
        global _logging_manager
        from src.logging_manager import _logging_manager
        _logging_manager = LoggingManager(
            log_dir=str(self.log_dir),
            enable_console=False
        )
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
        # Reset global logging manager
        global _logging_manager
        from src.logging_manager import _logging_manager
        _logging_manager = None
    
    def test_convenience_functions(self):
        """Test convenience logging functions"""
        log_info("test_component", "Info message", user_id="test_user")
        log_error("test_component", "Error message", user_id="test_user")
        log_performance("test_component", "test_operation", 1.5, True, user_id="test_user")
        log_token_usage("test_user", "test_operation", 100, 50, 150)
        log_audit("test_user", "data_access", "user_profile", True)
        
        # Check that files were created
        date_str = datetime.now().strftime('%Y%m%d')
        
        structured_log_file = self.log_dir / f"structured_{date_str}.jsonl"
        self.assertTrue(structured_log_file.exists())
        
        token_log_file = self.log_dir / "tokens" / f"tokens_{date_str}.jsonl"
        self.assertTrue(token_log_file.exists())
        
        audit_log_file = self.log_dir / "audit" / f"audit_{date_str}.jsonl"
        self.assertTrue(audit_log_file.exists())


class TestDataStructures(unittest.TestCase):
    """Test data structure classes"""
    
    def test_log_entry(self):
        """Test LogEntry dataclass"""
        entry = LogEntry(
            timestamp="2025-07-23T10:00:00",
            level="INFO",
            category="system",
            component="test",
            message="Test message",
            user_id="test_user",
            execution_time=1.5
        )
        
        self.assertEqual(entry.timestamp, "2025-07-23T10:00:00")
        self.assertEqual(entry.level, "INFO")
        self.assertEqual(entry.category, "system")
        self.assertEqual(entry.component, "test")
        self.assertEqual(entry.message, "Test message")
        self.assertEqual(entry.user_id, "test_user")
        self.assertEqual(entry.execution_time, 1.5)
    
    def test_token_usage_log(self):
        """Test TokenUsageLog dataclass"""
        log = TokenUsageLog(
            timestamp="2025-07-23T10:00:00",
            user_id="test_user",
            operation="test_operation",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cost_estimate=0.01,
            model="gemini-2.5-flash"
        )
        
        self.assertEqual(log.user_id, "test_user")
        self.assertEqual(log.operation, "test_operation")
        self.assertEqual(log.prompt_tokens, 100)
        self.assertEqual(log.completion_tokens, 50)
        self.assertEqual(log.total_tokens, 150)
        self.assertEqual(log.cost_estimate, 0.01)
        self.assertEqual(log.model, "gemini-2.5-flash")
    
    def test_audit_log_entry(self):
        """Test AuditLogEntry dataclass"""
        entry = AuditLogEntry(
            timestamp="2025-07-23T10:00:00",
            user_id="test_user",
            action="data_access",
            resource="user_profile",
            success=True,
            ip_address="127.0.0.1"
        )
        
        self.assertEqual(entry.user_id, "test_user")
        self.assertEqual(entry.action, "data_access")
        self.assertEqual(entry.resource, "user_profile")
        self.assertTrue(entry.success)
        self.assertEqual(entry.ip_address, "127.0.0.1")


if __name__ == '__main__':
    unittest.main()