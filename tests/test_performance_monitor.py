"""
Tests for FounderForge Performance Monitor
"""

import json
import tempfile
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.performance_monitor import (
    PerformanceMonitor, MetricType, AlertLevel, PerformanceThreshold,
    Alert, HealthCheckResult, PerformanceMetric,
    get_performance_monitor, setup_performance_monitoring,
    record_response_time, record_accuracy, record_error_rate, record_database_performance
)
from src.logging_manager import LoggingManager


class TestPerformanceMetric(unittest.TestCase):
    """Test cases for PerformanceMetric"""
    
    def setUp(self):
        """Set up test environment"""
        self.metric = PerformanceMetric(MetricType.RESPONSE_TIME, "test_component")
    
    def test_add_value(self):
        """Test adding metric values"""
        self.metric.add_value(1.5)
        self.metric.add_value(2.0)
        self.metric.add_value(1.8)
        
        self.assertEqual(len(self.metric.values), 3)
        self.assertEqual(self.metric.values, [1.5, 2.0, 1.8])
    
    def test_get_recent_values(self):
        """Test getting recent values within time window"""
        now = datetime.now()
        
        # Add values with specific timestamps
        self.metric.add_value(1.0, now - timedelta(seconds=90))  # Too old
        self.metric.add_value(2.0, now - timedelta(seconds=30))  # Recent
        self.metric.add_value(3.0, now - timedelta(seconds=10))  # Recent
        
        recent_values = self.metric.get_recent_values(window_seconds=60)
        self.assertEqual(recent_values, [2.0, 3.0])
    
    def test_get_statistics(self):
        """Test getting statistical summary"""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for value in values:
            self.metric.add_value(value)
        
        stats = self.metric.get_statistics()
        
        self.assertEqual(stats['count'], 5)
        self.assertEqual(stats['mean'], 3.0)
        self.assertEqual(stats['median'], 3.0)
        self.assertEqual(stats['min'], 1.0)
        self.assertEqual(stats['max'], 5.0)
        self.assertGreater(stats['std_dev'], 0)
    
    def test_empty_statistics(self):
        """Test statistics with no data"""
        stats = self.metric.get_statistics()
        
        self.assertEqual(stats['count'], 0)
        self.assertEqual(stats['mean'], 0.0)
        self.assertEqual(stats['median'], 0.0)
        self.assertEqual(stats['min'], 0.0)
        self.assertEqual(stats['max'], 0.0)
        self.assertEqual(stats['std_dev'], 0.0)


class TestPerformanceMonitor(unittest.TestCase):
    """Test cases for PerformanceMonitor"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = Path(self.temp_dir) / "test_logs"
        
        # Set up logging manager
        self.logging_manager = LoggingManager(
            log_dir=str(self.log_dir),
            enable_console=False
        )
        
        # Create performance monitor
        self.alerts_received = []
        self.monitor = PerformanceMonitor(
            alert_handlers=[self.alert_handler],
            monitoring_interval=1
        )
    
    def tearDown(self):
        """Clean up test environment"""
        self.monitor.stop_monitoring()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def alert_handler(self, alert: Alert):
        """Test alert handler"""
        self.alerts_received.append(alert)
    
    def test_initialization(self):
        """Test performance monitor initialization"""
        self.assertIsNotNone(self.monitor.logger)
        self.assertEqual(len(self.monitor.alert_handlers), 1)
        self.assertFalse(self.monitor.monitoring_active)
    
    def test_set_threshold(self):
        """Test setting performance thresholds"""
        self.monitor.set_threshold(
            MetricType.RESPONSE_TIME,
            warning=1.0,
            critical=2.0,
            window=30,
            min_samples=3
        )
        
        threshold = self.monitor.thresholds[MetricType.RESPONSE_TIME]
        self.assertEqual(threshold.warning_threshold, 1.0)
        self.assertEqual(threshold.critical_threshold, 2.0)
        self.assertEqual(threshold.measurement_window, 30)
        self.assertEqual(threshold.min_samples, 3)
    
    def test_record_metric(self):
        """Test recording metrics"""
        self.monitor.record_metric(MetricType.RESPONSE_TIME, "test_component", 1.5)
        
        metric_key = "response_time_test_component"
        self.assertIn(metric_key, self.monitor.metrics)
        
        metric = self.monitor.metrics[metric_key]
        self.assertEqual(len(metric.values), 1)
        self.assertEqual(metric.values[0], 1.5)
    
    def test_threshold_alerts(self):
        """Test threshold-based alerting"""
        # Set low thresholds for testing
        self.monitor.set_threshold(
            MetricType.RESPONSE_TIME,
            warning=1.0,
            critical=2.0,
            window=60,
            min_samples=2
        )
        
        # Record values that should trigger alerts
        self.monitor.record_metric(MetricType.RESPONSE_TIME, "test_component", 1.5)  # Above warning
        self.monitor.record_metric(MetricType.RESPONSE_TIME, "test_component", 1.6)  # Above warning
        
        # Should have received a warning alert
        self.assertEqual(len(self.alerts_received), 1)
        self.assertEqual(self.alerts_received[0].level, AlertLevel.WARNING)
        
        # Clear alerts and test critical threshold
        self.alerts_received.clear()
        
        self.monitor.record_metric(MetricType.RESPONSE_TIME, "test_component", 2.5)  # Above critical
        self.monitor.record_metric(MetricType.RESPONSE_TIME, "test_component", 2.6)  # Above critical
        
        # Should have received a critical alert
        self.assertEqual(len(self.alerts_received), 1)
        self.assertEqual(self.alerts_received[0].level, AlertLevel.CRITICAL)
    
    def test_accuracy_threshold_logic(self):
        """Test accuracy threshold logic (lower is worse)"""
        # Set accuracy thresholds
        self.monitor.set_threshold(
            MetricType.ACCURACY,
            warning=0.85,
            critical=0.80,
            window=60,
            min_samples=2
        )
        
        # Record low accuracy values
        self.monitor.record_metric(MetricType.ACCURACY, "test_component", 0.82)  # Below warning
        self.monitor.record_metric(MetricType.ACCURACY, "test_component", 0.83)  # Below warning
        
        # Should have received a warning alert
        self.assertEqual(len(self.alerts_received), 1)
        self.assertEqual(self.alerts_received[0].level, AlertLevel.WARNING)
    
    def test_health_checks(self):
        """Test health check functionality"""
        def healthy_check():
            return HealthCheckResult(
                component="test_component",
                status="healthy",
                message="All good",
                timestamp=datetime.now().isoformat()
            )
        
        def unhealthy_check():
            return HealthCheckResult(
                component="failing_component",
                status="critical",
                message="Something is wrong",
                timestamp=datetime.now().isoformat()
            )
        
        self.monitor.add_health_check("healthy_test", healthy_check)
        self.monitor.add_health_check("unhealthy_test", unhealthy_check)
        
        results = self.monitor.run_health_checks()
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results["healthy_test"].status, "healthy")
        self.assertEqual(results["unhealthy_test"].status, "critical")
        self.assertIsNotNone(results["healthy_test"].response_time)
    
    def test_health_check_exception_handling(self):
        """Test health check exception handling"""
        def failing_check():
            raise Exception("Health check failed")
        
        self.monitor.add_health_check("failing_check", failing_check)
        results = self.monitor.run_health_checks()
        
        self.assertEqual(results["failing_check"].status, "critical")
        self.assertIn("Health check failed", results["failing_check"].message)
    
    @patch('psutil.Process')
    def test_get_system_metrics(self, mock_process):
        """Test system metrics collection"""
        # Mock psutil.Process
        mock_process_instance = MagicMock()
        mock_process_instance.cpu_percent.return_value = 25.5
        mock_process_instance.memory_info.return_value.rss = 100 * 1024 * 1024  # 100MB
        mock_process_instance.memory_percent.return_value = 10.0
        mock_process_instance.open_files.return_value = []
        mock_process_instance.num_threads.return_value = 5
        mock_process_instance.create_time.return_value = time.time() - 3600  # 1 hour ago
        mock_process.return_value = mock_process_instance
        
        metrics = self.monitor.get_system_metrics()
        
        self.assertEqual(metrics['cpu_usage'], 25.5)
        self.assertEqual(metrics['memory_usage'], 100.0)
        self.assertEqual(metrics['memory_percent'], 10.0)
        self.assertEqual(metrics['open_files'], 0)
        self.assertEqual(metrics['threads'], 5)
        self.assertGreater(metrics['uptime'], 3500)  # Should be around 3600
    
    def test_get_metric_statistics(self):
        """Test getting metric statistics"""
        # Record some metrics
        for i in range(5):
            self.monitor.record_metric(MetricType.RESPONSE_TIME, "test_component", float(i + 1))
        
        stats = self.monitor.get_metric_statistics(
            MetricType.RESPONSE_TIME, 
            "test_component", 
            window_seconds=300
        )
        
        self.assertIsNotNone(stats)
        self.assertEqual(stats['count'], 5)
        self.assertEqual(stats['mean'], 3.0)
        self.assertEqual(stats['min'], 1.0)
        self.assertEqual(stats['max'], 5.0)
    
    def test_get_performance_summary(self):
        """Test getting comprehensive performance summary"""
        # Add some test data
        self.monitor.record_metric(MetricType.RESPONSE_TIME, "test_component", 1.5)
        
        def test_health_check():
            return HealthCheckResult(
                component="test",
                status="healthy",
                message="OK",
                timestamp=datetime.now().isoformat()
            )
        
        self.monitor.add_health_check("test_check", test_health_check)
        
        summary = self.monitor.get_performance_summary()
        
        self.assertIn('timestamp', summary)
        self.assertIn('system_metrics', summary)
        self.assertIn('health_checks', summary)
        self.assertIn('recent_alerts', summary)
        self.assertIn('metric_statistics', summary)
        
        self.assertEqual(summary['health_checks']['test_check'].status, 'healthy')
    
    @patch('psutil.Process')
    def test_monitoring_loop(self, mock_process):
        """Test background monitoring loop"""
        # Mock psutil.Process
        mock_process_instance = MagicMock()
        mock_process_instance.cpu_percent.return_value = 25.5
        mock_process_instance.memory_info.return_value.rss = 100 * 1024 * 1024
        mock_process_instance.memory_percent.return_value = 10.0
        mock_process_instance.open_files.return_value = []
        mock_process_instance.num_threads.return_value = 5
        mock_process_instance.create_time.return_value = time.time() - 3600
        mock_process.return_value = mock_process_instance
        
        # Start monitoring
        self.monitor.start_monitoring()
        self.assertTrue(self.monitor.monitoring_active)
        
        # Wait a bit for monitoring to run
        time.sleep(2)
        
        # Stop monitoring
        self.monitor.stop_monitoring()
        self.assertFalse(self.monitor.monitoring_active)
        
        # Check that system metrics were recorded
        self.assertIn("cpu_usage_system", self.monitor.metrics)
        self.assertIn("memory_usage_system", self.monitor.metrics)
    
    def test_export_metrics(self):
        """Test metrics export functionality"""
        # Record some test metrics
        self.monitor.record_metric(MetricType.RESPONSE_TIME, "test_component", 1.5)
        self.monitor.record_metric(MetricType.ACCURACY, "test_component", 0.95)
        
        # Generate an alert
        self.monitor.set_threshold(MetricType.RESPONSE_TIME, 1.0, 2.0, 60, 1)
        self.monitor.record_metric(MetricType.RESPONSE_TIME, "test_component", 1.8)
        
        # Export metrics
        export_file = Path(self.temp_dir) / "metrics_export.json"
        self.monitor.export_metrics(str(export_file), hours=1)
        
        self.assertTrue(export_file.exists())
        
        with open(export_file, 'r', encoding='utf-8') as f:
            export_data = json.load(f)
        
        self.assertIn('export_timestamp', export_data)
        self.assertIn('metrics', export_data)
        self.assertIn('alerts', export_data)
        self.assertEqual(export_data['hours_exported'], 1)


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = Path(self.temp_dir) / "test_logs"
        
        # Set up logging manager
        self.logging_manager = LoggingManager(
            log_dir=str(self.log_dir),
            enable_console=False
        )
        
        # Set up global performance monitor
        global _performance_monitor
        from src.performance_monitor import _performance_monitor
        _performance_monitor = PerformanceMonitor()
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
        # Reset global performance monitor
        global _performance_monitor
        from src.performance_monitor import _performance_monitor
        if _performance_monitor:
            _performance_monitor.stop_monitoring()
        _performance_monitor = None
    
    def test_convenience_functions(self):
        """Test convenience recording functions"""
        record_response_time("test_component", 1.5)
        record_accuracy("test_component", 0.95)
        record_error_rate("test_component", 0.02)
        record_database_performance("select_operation", 0.005)
        
        monitor = get_performance_monitor()
        
        # Check that metrics were recorded
        self.assertIn("response_time_test_component", monitor.metrics)
        self.assertIn("accuracy_test_component", monitor.metrics)
        self.assertIn("error_rate_test_component", monitor.metrics)
        self.assertIn("database_performance_select_operation", monitor.metrics)


class TestDataStructures(unittest.TestCase):
    """Test data structure classes"""
    
    def test_performance_threshold(self):
        """Test PerformanceThreshold dataclass"""
        threshold = PerformanceThreshold(
            metric_type=MetricType.RESPONSE_TIME,
            warning_threshold=1.0,
            critical_threshold=2.0,
            measurement_window=60,
            min_samples=5
        )
        
        self.assertEqual(threshold.metric_type, MetricType.RESPONSE_TIME)
        self.assertEqual(threshold.warning_threshold, 1.0)
        self.assertEqual(threshold.critical_threshold, 2.0)
        self.assertEqual(threshold.measurement_window, 60)
        self.assertEqual(threshold.min_samples, 5)
    
    def test_alert(self):
        """Test Alert dataclass"""
        alert = Alert(
            timestamp="2025-07-23T10:00:00",
            level=AlertLevel.WARNING,
            metric_type=MetricType.RESPONSE_TIME,
            component="test_component",
            message="Threshold exceeded",
            current_value=1.5,
            threshold=1.0
        )
        
        self.assertEqual(alert.level, AlertLevel.WARNING)
        self.assertEqual(alert.metric_type, MetricType.RESPONSE_TIME)
        self.assertEqual(alert.component, "test_component")
        self.assertEqual(alert.current_value, 1.5)
        self.assertEqual(alert.threshold, 1.0)
    
    def test_health_check_result(self):
        """Test HealthCheckResult dataclass"""
        result = HealthCheckResult(
            component="test_component",
            status="healthy",
            message="All systems operational",
            timestamp="2025-07-23T10:00:00",
            response_time=0.5
        )
        
        self.assertEqual(result.component, "test_component")
        self.assertEqual(result.status, "healthy")
        self.assertEqual(result.message, "All systems operational")
        self.assertEqual(result.response_time, 0.5)


if __name__ == '__main__':
    unittest.main()