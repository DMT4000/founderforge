"""
FounderForge Performance Monitor

Provides real-time performance monitoring, alerting, and health checks.
Monitors memory usage, response times, accuracy metrics, and system health.
"""

import json
import time
import threading
import psutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, NamedTuple
from dataclasses import dataclass, asdict
from enum import Enum
import statistics
import logging

try:
    from .logging_manager import get_logging_manager, LogLevel, LogCategory
except ImportError:
    from logging_manager import get_logging_manager, LogLevel, LogCategory


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics to monitor"""
    RESPONSE_TIME = "response_time"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    ACCURACY = "accuracy"
    ERROR_RATE = "error_rate"
    TOKEN_USAGE = "token_usage"
    DATABASE_PERFORMANCE = "database_performance"


@dataclass
class PerformanceThreshold:
    """Performance threshold configuration"""
    metric_type: MetricType
    warning_threshold: float
    critical_threshold: float
    measurement_window: int = 60  # seconds
    min_samples: int = 5


@dataclass
class Alert:
    """Performance alert"""
    timestamp: str
    level: AlertLevel
    metric_type: MetricType
    component: str
    message: str
    current_value: float
    threshold: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class HealthCheckResult:
    """Health check result"""
    component: str
    status: str  # "healthy", "warning", "critical"
    message: str
    timestamp: str
    response_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class PerformanceMetric:
    """Container for performance metric data"""
    
    def __init__(self, metric_type: MetricType, component: str):
        self.metric_type = metric_type
        self.component = component
        self.values: List[float] = []
        self.timestamps: List[datetime] = []
        self._lock = threading.Lock()
    
    def add_value(self, value: float, timestamp: Optional[datetime] = None):
        """Add a new metric value"""
        if timestamp is None:
            timestamp = datetime.now()
        
        with self._lock:
            self.values.append(value)
            self.timestamps.append(timestamp)
            
            # Keep only recent values (last hour)
            cutoff = datetime.now() - timedelta(hours=1)
            while self.timestamps and self.timestamps[0] < cutoff:
                self.values.pop(0)
                self.timestamps.pop(0)
    
    def get_recent_values(self, window_seconds: int = 60) -> List[float]:
        """Get values from the recent time window"""
        cutoff = datetime.now() - timedelta(seconds=window_seconds)
        
        with self._lock:
            recent_values = []
            for i, timestamp in enumerate(self.timestamps):
                if timestamp >= cutoff:
                    recent_values.append(self.values[i])
            return recent_values
    
    def get_statistics(self, window_seconds: int = 60) -> Dict[str, float]:
        """Get statistical summary of recent values"""
        values = self.get_recent_values(window_seconds)
        
        if not values:
            return {
                'count': 0,
                'mean': 0.0,
                'median': 0.0,
                'min': 0.0,
                'max': 0.0,
                'std_dev': 0.0
            }
        
        return {
            'count': len(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'min': min(values),
            'max': max(values),
            'std_dev': statistics.stdev(values) if len(values) > 1 else 0.0
        }


class PerformanceMonitor:
    """
    Real-time performance monitoring system.
    
    Monitors system metrics, tracks performance thresholds,
    generates alerts, and provides health checks.
    """
    
    def __init__(self, 
                 alert_handlers: Optional[List[Callable[[Alert], None]]] = None,
                 monitoring_interval: int = 30):
        """
        Initialize performance monitor.
        
        Args:
            alert_handlers: List of functions to handle alerts
            monitoring_interval: Interval between monitoring checks (seconds)
        """
        self.logging_manager = get_logging_manager()
        self.logger = self.logging_manager.get_logger("performance_monitor", LogCategory.PERFORMANCE)
        
        self.alert_handlers = alert_handlers or []
        self.monitoring_interval = monitoring_interval
        
        # Metrics storage
        self.metrics: Dict[str, PerformanceMetric] = {}
        self.metrics_lock = threading.Lock()
        
        # Performance thresholds
        self.thresholds: Dict[MetricType, PerformanceThreshold] = {
            MetricType.RESPONSE_TIME: PerformanceThreshold(
                MetricType.RESPONSE_TIME, 
                warning_threshold=2.0,  # 2 seconds
                critical_threshold=5.0  # 5 seconds
            ),
            MetricType.MEMORY_USAGE: PerformanceThreshold(
                MetricType.MEMORY_USAGE,
                warning_threshold=500.0,  # 500 MB
                critical_threshold=1000.0  # 1 GB
            ),
            MetricType.CPU_USAGE: PerformanceThreshold(
                MetricType.CPU_USAGE,
                warning_threshold=70.0,  # 70%
                critical_threshold=90.0  # 90%
            ),
            MetricType.ACCURACY: PerformanceThreshold(
                MetricType.ACCURACY,
                warning_threshold=0.85,  # 85%
                critical_threshold=0.80  # 80%
            ),
            MetricType.ERROR_RATE: PerformanceThreshold(
                MetricType.ERROR_RATE,
                warning_threshold=0.05,  # 5%
                critical_threshold=0.10  # 10%
            ),
            MetricType.DATABASE_PERFORMANCE: PerformanceThreshold(
                MetricType.DATABASE_PERFORMANCE,
                warning_threshold=0.010,  # 10ms
                critical_threshold=0.050  # 50ms
            )
        }
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        # Health check functions
        self.health_checks: Dict[str, Callable[[], HealthCheckResult]] = {}
        
        # Alert history
        self.alert_history: List[Alert] = []
        self.alert_history_lock = threading.Lock()
        
        self.logger.info("Performance monitor initialized")
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add an alert handler function"""
        self.alert_handlers.append(handler)
    
    def set_threshold(self, metric_type: MetricType, warning: float, critical: float, 
                     window: int = 60, min_samples: int = 5):
        """
        Set performance threshold for a metric type.
        
        Args:
            metric_type: Type of metric
            warning: Warning threshold value
            critical: Critical threshold value
            window: Measurement window in seconds
            min_samples: Minimum samples required for threshold check
        """
        self.thresholds[metric_type] = PerformanceThreshold(
            metric_type=metric_type,
            warning_threshold=warning,
            critical_threshold=critical,
            measurement_window=window,
            min_samples=min_samples
        )
        
        self.logger.info(f"Updated threshold for {metric_type.value}", extra={
            'metric_type': metric_type.value,
            'warning_threshold': warning,
            'critical_threshold': critical
        })
    
    def record_metric(self, metric_type: MetricType, component: str, value: float):
        """
        Record a performance metric value.
        
        Args:
            metric_type: Type of metric
            component: Component name
            value: Metric value
        """
        metric_key = f"{metric_type.value}_{component}"
        
        with self.metrics_lock:
            if metric_key not in self.metrics:
                self.metrics[metric_key] = PerformanceMetric(metric_type, component)
            
            self.metrics[metric_key].add_value(value)
        
        # Check thresholds
        self._check_thresholds(metric_type, component, value)
        
        # Log to performance metrics
        self.logging_manager.log_performance_metrics(
            component=component,
            operation=metric_type.value,
            execution_time=value if metric_type == MetricType.RESPONSE_TIME else 0.0,
            success=True,
            metadata={'metric_type': metric_type.value, 'value': value}
        )
    
    def _check_thresholds(self, metric_type: MetricType, component: str, current_value: float):
        """Check if current value exceeds thresholds"""
        if metric_type not in self.thresholds:
            return
        
        threshold = self.thresholds[metric_type]
        metric_key = f"{metric_type.value}_{component}"
        
        if metric_key in self.metrics:
            recent_values = self.metrics[metric_key].get_recent_values(threshold.measurement_window)
            
            if len(recent_values) >= threshold.min_samples:
                avg_value = statistics.mean(recent_values)
                
                # Check critical threshold
                if self._exceeds_threshold(metric_type, avg_value, threshold.critical_threshold):
                    self._generate_alert(
                        AlertLevel.CRITICAL,
                        metric_type,
                        component,
                        f"Critical threshold exceeded: {avg_value:.3f} > {threshold.critical_threshold}",
                        avg_value,
                        threshold.critical_threshold
                    )
                # Check warning threshold
                elif self._exceeds_threshold(metric_type, avg_value, threshold.warning_threshold):
                    self._generate_alert(
                        AlertLevel.WARNING,
                        metric_type,
                        component,
                        f"Warning threshold exceeded: {avg_value:.3f} > {threshold.warning_threshold}",
                        avg_value,
                        threshold.warning_threshold
                    )
    
    def _exceeds_threshold(self, metric_type: MetricType, value: float, threshold: float) -> bool:
        """Check if value exceeds threshold based on metric type"""
        if metric_type == MetricType.ACCURACY:
            # For accuracy, lower values are worse
            return value < threshold
        else:
            # For other metrics, higher values are worse
            return value > threshold
    
    def _generate_alert(self, level: AlertLevel, metric_type: MetricType, component: str,
                       message: str, current_value: float, threshold: float):
        """Generate and handle an alert"""
        alert = Alert(
            timestamp=datetime.now().isoformat(),
            level=level,
            metric_type=metric_type,
            component=component,
            message=message,
            current_value=current_value,
            threshold=threshold
        )
        
        # Store in history
        with self.alert_history_lock:
            self.alert_history.append(alert)
            # Keep only recent alerts (last 24 hours)
            cutoff = datetime.now() - timedelta(hours=24)
            self.alert_history = [
                a for a in self.alert_history 
                if datetime.fromisoformat(a.timestamp) >= cutoff
            ]
        
        # Log alert
        log_level = LogLevel.WARNING if level == AlertLevel.WARNING else LogLevel.CRITICAL
        self.logging_manager.log_structured(
            log_level,
            LogCategory.PERFORMANCE,
            component,
            message,
            metadata={
                'alert_level': level.value,
                'metric_type': metric_type.value,
                'current_value': current_value,
                'threshold': threshold
            }
        )
        
        # Call alert handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"Alert handler failed: {e}")
    
    def add_health_check(self, name: str, check_function: Callable[[], HealthCheckResult]):
        """
        Add a health check function.
        
        Args:
            name: Health check name
            check_function: Function that returns HealthCheckResult
        """
        self.health_checks[name] = check_function
        self.logger.info(f"Added health check: {name}")
    
    def run_health_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all health checks and return results"""
        results = {}
        
        for name, check_function in self.health_checks.items():
            try:
                start_time = time.time()
                result = check_function()
                result.response_time = time.time() - start_time
                results[name] = result
                
                self.logger.debug(f"Health check {name}: {result.status}")
                
            except Exception as e:
                results[name] = HealthCheckResult(
                    component=name,
                    status="critical",
                    message=f"Health check failed: {str(e)}",
                    timestamp=datetime.now().isoformat()
                )
                self.logger.error(f"Health check {name} failed: {e}")
        
        return results
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        process = psutil.Process()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'cpu_usage': process.cpu_percent(),
            'memory_usage': process.memory_info().rss / 1024 / 1024,  # MB
            'memory_percent': process.memory_percent(),
            'open_files': len(process.open_files()),
            'threads': process.num_threads(),
            'uptime': time.time() - process.create_time()
        }
    
    def get_metric_statistics(self, metric_type: MetricType, component: str, 
                            window_seconds: int = 300) -> Optional[Dict[str, float]]:
        """
        Get statistics for a specific metric.
        
        Args:
            metric_type: Type of metric
            component: Component name
            window_seconds: Time window for statistics
            
        Returns:
            Dictionary with statistics or None if no data
        """
        metric_key = f"{metric_type.value}_{component}"
        
        with self.metrics_lock:
            if metric_key in self.metrics:
                return self.metrics[metric_key].get_statistics(window_seconds)
        
        return None
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'system_metrics': self.get_system_metrics(),
            'health_checks': self.run_health_checks(),
            'recent_alerts': [],
            'metric_statistics': {}
        }
        
        # Get recent alerts (last hour)
        cutoff = datetime.now() - timedelta(hours=1)
        with self.alert_history_lock:
            summary['recent_alerts'] = [
                asdict(alert) for alert in self.alert_history
                if datetime.fromisoformat(alert.timestamp) >= cutoff
            ]
        
        # Get statistics for all metrics
        with self.metrics_lock:
            for metric_key, metric in self.metrics.items():
                stats = metric.get_statistics(300)  # 5 minute window
                if stats['count'] > 0:
                    summary['metric_statistics'][metric_key] = stats
        
        return summary
    
    def start_monitoring(self):
        """Start background monitoring thread"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        self.logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                # Record system metrics
                system_metrics = self.get_system_metrics()
                self.record_metric(MetricType.CPU_USAGE, "system", system_metrics['cpu_usage'])
                self.record_metric(MetricType.MEMORY_USAGE, "system", system_metrics['memory_usage'])
                
                # Run health checks
                health_results = self.run_health_checks()
                
                # Check for unhealthy components
                for name, result in health_results.items():
                    if result.status == "critical":
                        self._generate_alert(
                            AlertLevel.CRITICAL,
                            MetricType.RESPONSE_TIME,  # Generic metric type
                            name,
                            f"Health check failed: {result.message}",
                            0.0,
                            0.0
                        )
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(self.monitoring_interval)
    
    def export_metrics(self, output_file: str, hours: int = 24):
        """
        Export metrics to file for analysis.
        
        Args:
            output_file: Output file path
            hours: Number of hours of data to export
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'hours_exported': hours,
            'metrics': {}
        }
        
        with self.metrics_lock:
            for metric_key, metric in self.metrics.items():
                # Filter data by time window
                filtered_data = []
                for i, timestamp in enumerate(metric.timestamps):
                    if timestamp >= cutoff:
                        filtered_data.append({
                            'timestamp': timestamp.isoformat(),
                            'value': metric.values[i]
                        })
                
                if filtered_data:
                    export_data['metrics'][metric_key] = {
                        'metric_type': metric.metric_type.value,
                        'component': metric.component,
                        'data': filtered_data,
                        'statistics': metric.get_statistics(hours * 3600)
                    }
        
        # Export alerts
        with self.alert_history_lock:
            alerts_data = []
            for alert in self.alert_history:
                if datetime.fromisoformat(alert.timestamp) >= cutoff:
                    alert_dict = asdict(alert)
                    # Convert enum to string for JSON serialization
                    alert_dict['level'] = alert.level.value
                    alert_dict['metric_type'] = alert.metric_type.value
                    alerts_data.append(alert_dict)
            export_data['alerts'] = alerts_data
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Metrics exported to {output_file}")


# Global performance monitor instance
_performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor


def setup_performance_monitoring(alert_handlers: Optional[List[Callable[[Alert], None]]] = None,
                                monitoring_interval: int = 30) -> PerformanceMonitor:
    """
    Set up global performance monitoring.
    
    Args:
        alert_handlers: List of functions to handle alerts
        monitoring_interval: Interval between monitoring checks
        
    Returns:
        Configured performance monitor
    """
    global _performance_monitor
    _performance_monitor = PerformanceMonitor(
        alert_handlers=alert_handlers,
        monitoring_interval=monitoring_interval
    )
    return _performance_monitor


# Convenience functions
def record_response_time(component: str, response_time: float):
    """Record response time metric"""
    get_performance_monitor().record_metric(MetricType.RESPONSE_TIME, component, response_time)


def record_accuracy(component: str, accuracy: float):
    """Record accuracy metric"""
    get_performance_monitor().record_metric(MetricType.ACCURACY, component, accuracy)


def record_error_rate(component: str, error_rate: float):
    """Record error rate metric"""
    get_performance_monitor().record_metric(MetricType.ERROR_RATE, component, error_rate)


def record_database_performance(operation: str, execution_time: float):
    """Record database performance metric"""
    get_performance_monitor().record_metric(MetricType.DATABASE_PERFORMANCE, operation, execution_time)