"""Tests for the feature flag management system."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.feature_flag_manager import FeatureFlagManager, FeatureFlagChange


class TestFeatureFlagManager(unittest.TestCase):
    """Test cases for FeatureFlagManager."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir)
        self.manager = FeatureFlagManager(self.config_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_initialization_creates_default_flags(self):
        """Test that initialization creates default feature flags."""
        flags = self.manager.get_all_flags()
        
        # Check that default flags are created
        self.assertIn("enable_memory_system", flags)
        self.assertIn("enable_multi_agent", flags)
        self.assertIn("confidence_threshold", flags)
        self.assertIn("max_context_tokens", flags)
        
        # Check default values
        self.assertTrue(flags["enable_memory_system"])
        self.assertEqual(flags["confidence_threshold"], 0.8)
        self.assertEqual(flags["max_context_tokens"], 16000)
    
    def test_get_flag_returns_correct_value(self):
        """Test getting individual flag values."""
        # Test existing flag
        value = self.manager.get_flag("enable_memory_system")
        self.assertTrue(value)
        
        # Test non-existing flag with default
        value = self.manager.get_flag("non_existing_flag", "default_value")
        self.assertEqual(value, "default_value")
        
        # Test non-existing flag without default
        value = self.manager.get_flag("non_existing_flag")
        self.assertIsNone(value)
    
    def test_is_enabled_returns_boolean(self):
        """Test is_enabled method for boolean flags."""
        # Test enabled flag
        self.assertTrue(self.manager.is_enabled("enable_memory_system"))
        
        # Test non-existing flag
        self.assertFalse(self.manager.is_enabled("non_existing_flag"))
    
    def test_set_flag_updates_value(self):
        """Test setting flag values."""
        # Set a new flag
        result = self.manager.set_flag("test_flag", True, "test_user", "Testing")
        self.assertTrue(result)
        
        # Verify the flag was set
        value = self.manager.get_flag("test_flag")
        self.assertTrue(value)
        
        # Set the same value again (should return False)
        result = self.manager.set_flag("test_flag", True, "test_user", "Testing")
        self.assertFalse(result)
    
    def test_toggle_flag_changes_boolean_value(self):
        """Test toggling boolean flags."""
        # Set initial value
        self.manager.set_flag("toggle_test", True, "test_user")
        
        # Toggle the flag
        new_value = self.manager.toggle_flag("toggle_test", "test_user", "Testing toggle")
        self.assertFalse(new_value)
        
        # Verify the flag was toggled
        value = self.manager.get_flag("toggle_test")
        self.assertFalse(value)
        
        # Toggle again
        new_value = self.manager.toggle_flag("toggle_test", "test_user")
        self.assertTrue(new_value)
    
    def test_toggle_non_boolean_flag_raises_error(self):
        """Test that toggling non-boolean flags raises an error."""
        # Set a non-boolean flag
        self.manager.set_flag("numeric_flag", 42, "test_user")
        
        # Try to toggle it
        with self.assertRaises(ValueError):
            self.manager.toggle_flag("numeric_flag", "test_user")
    
    def test_change_tracking_logs_modifications(self):
        """Test that flag changes are logged."""
        # Make some changes
        self.manager.set_flag("test_flag_1", True, "user1", "First test")
        self.manager.set_flag("test_flag_2", "value", "user2", "Second test")
        self.manager.toggle_flag("enable_memory_system", "user3", "Toggle test")
        
        # Get change history
        changes = self.manager.get_change_history()
        
        # Should have at least 3 changes
        self.assertGreaterEqual(len(changes), 3)
        
        # Check that changes are FeatureFlagChange objects
        for change in changes:
            self.assertIsInstance(change, FeatureFlagChange)
            self.assertIsNotNone(change.timestamp)
            self.assertIsNotNone(change.flag_name)
            self.assertIsNotNone(change.changed_by)
    
    def test_change_history_filtering(self):
        """Test filtering change history by flag name."""
        # Make changes to different flags
        self.manager.set_flag("flag_a", True, "user1")
        self.manager.set_flag("flag_b", False, "user2")
        self.manager.set_flag("flag_a", False, "user3")
        
        # Get history for specific flag
        changes = self.manager.get_change_history("flag_a")
        
        # Should only have changes for flag_a
        for change in changes:
            self.assertEqual(change.flag_name, "flag_a")
    
    def test_export_import_flags(self):
        """Test exporting and importing flags."""
        # Set some custom flags
        self.manager.set_flag("custom_flag_1", True, "test")
        self.manager.set_flag("custom_flag_2", 42, "test")
        
        # Export flags
        export_path = self.config_dir / "exported_flags.json"
        self.manager.export_flags(export_path)
        
        # Verify export file exists and contains data
        self.assertTrue(export_path.exists())
        with open(export_path, 'r') as f:
            exported_data = json.load(f)
        
        self.assertIn("custom_flag_1", exported_data)
        self.assertIn("custom_flag_2", exported_data)
        
        # Create a new manager and import
        new_manager = FeatureFlagManager(Path(tempfile.mkdtemp()))
        changes_made = new_manager.import_flags(export_path, "import_test")
        
        # Verify import worked
        self.assertGreater(changes_made, 0)
        self.assertTrue(new_manager.get_flag("custom_flag_1"))
        self.assertEqual(new_manager.get_flag("custom_flag_2"), 42)
    
    def test_validation_detects_issues(self):
        """Test flag validation."""
        # Set invalid values
        self.manager.set_flag("confidence_threshold", 1.5, "test")  # Should be 0-1
        self.manager.set_flag("max_context_tokens", -100, "test")  # Should be positive
        
        # Run validation
        issues = self.manager.validate_flags()
        
        # Should detect issues
        self.assertGreater(len(issues), 0)
        
        # Check specific issues
        issue_text = " ".join(issues)
        self.assertIn("confidence_threshold", issue_text)
        self.assertIn("max_context_tokens", issue_text)
    
    def test_cache_refresh_on_file_change(self):
        """Test that cache is refreshed when file is modified externally."""
        # Get initial flags
        initial_flags = self.manager.get_all_flags()
        
        # Modify file directly
        flags_file = self.config_dir / "feature_flags.json"
        with open(flags_file, 'r') as f:
            flags_data = json.load(f)
        
        flags_data["external_change"] = True
        
        with open(flags_file, 'w') as f:
            json.dump(flags_data, f)
        
        # Get flags again (should refresh cache)
        updated_flags = self.manager.get_all_flags()
        
        # Should include the external change
        self.assertIn("external_change", updated_flags)
        self.assertTrue(updated_flags["external_change"])
    
    def test_thread_safety(self):
        """Test thread safety of flag operations."""
        import threading
        import time
        
        results = []
        errors = []
        
        def worker(worker_id):
            try:
                for i in range(10):
                    flag_name = f"thread_test_{worker_id}_{i}"
                    self.manager.set_flag(flag_name, i, f"worker_{worker_id}")
                    value = self.manager.get_flag(flag_name)
                    results.append((flag_name, value))
                    time.sleep(0.001)  # Small delay to increase chance of race conditions
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        self.assertEqual(len(errors), 0, f"Thread safety errors: {errors}")
        self.assertEqual(len(results), 50)  # 5 workers * 10 operations each


class TestFeatureFlagManagerIntegration(unittest.TestCase):
    """Integration tests for feature flag manager."""
    
    def test_integration_with_settings(self):
        """Test integration with settings module."""
        from config.settings import settings
        
        # Test that we can access feature flags through settings
        flag_value = settings.get_feature_flag("enable_memory_system")
        self.assertIsNotNone(flag_value)
        
        # Test updating through settings
        original_value = settings.get_feature_flag("enable_memory_system")
        settings.update_feature_flag("enable_memory_system", not original_value)
        
        # Verify change
        new_value = settings.get_feature_flag("enable_memory_system")
        self.assertEqual(new_value, not original_value)
        
        # Restore original value
        settings.update_feature_flag("enable_memory_system", original_value)


if __name__ == '__main__':
    unittest.main()