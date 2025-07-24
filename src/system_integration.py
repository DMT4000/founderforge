"""
FounderForge System Integration

Integrates all system components with centralized logging and monitoring.
Ensures proper initialization and configuration of logging and performance monitoring.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import time

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from .logging_manager import (
        setup_logging, get_logging_manager, LogLevel, LogCategory,
        log_info, log_error, log_performance, log_token_usage, log_audit
    )
    from .performance_monitor import (
        setup_performance_monitoring, get_performance_monitor, MetricType,
        AlertLevel, HealthCheckResult, record_response_time, record_accuracy
    )
except ImportError:
    from logging_manager import (
        setup_logging, get_logging_manager, LogLevel, LogCategory,
        log_info, log_error, log_performance, log_token_usage, log_audit
    )
    from performance_monitor import (
        setup_performance_monitoring, get_performance_monitor, MetricType,
        AlertLevel, HealthCheckResult, record_response_time, record_accuracy
    )


class SystemIntegrator:
    """
    Integrates all system components with centralized logging and monitoring.
    
    Provides initialization, health checks, and system-wide configuration
    for logging and performance monitoring.
    """
    
    def __init__(self, 
                 log_dir: str = "data/logs",
                 log_level: LogLevel = LogLevel.INFO,
                 enable_console: bool = True,
                 monitoring_interval: int = 30):
        """
        Initialize system integration.
        
        Args:
            log_dir: Directory for log files
            log_level: Default logging level
            enable_console: Whether to enable console logging
            monitoring_interval: Performance monitoring interval in seconds
        """
        self.log_dir = log_dir
        self.log_level = log_level
        self.enable_console = enable_console
        self.monitoring_interval = monitoring_interval
        
        # Initialize logging and monitoring
        self.logging_manager = None
        self.performance_monitor = None
        self.initialized = False
    
    def initialize_system(self) -> bool:
        """
        Initialize the complete logging and monitoring system.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Set up logging
            self.logging_manager = setup_logging(
                log_dir=self.log_dir,
                log_level=self.log_level,
                enable_console=self.enable_console
            )
            
            # Set up performance monitoring with alert handlers
            alert_handlers = [
                self._console_alert_handler,
                self._log_alert_handler
            ]
            
            self.performance_monitor = setup_performance_monitoring(
                alert_handlers=alert_handlers,
                monitoring_interval=self.monitoring_interval
            )
            
            # Configure performance thresholds based on requirements
            self._configure_performance_thresholds()
            
            # Add health checks for key components
            self._setup_health_checks()
            
            # Start background monitoring
            self.performance_monitor.start_monitoring()
            
            self.initialized = True
            
            log_info("system_integrator", "System logging and monitoring initialized successfully")
            
            return True
            
        except Exception as e:
            print(f"Failed to initialize system: {e}")
            return False
    
    def _configure_performance_thresholds(self):
        """Configure performance thresholds based on requirements"""
        # Memory retrieval should be sub-10ms (Requirement 2.2)
        self.performance_monitor.set_threshold(
            MetricType.DATABASE_PERFORMANCE,
            warning=0.008,  # 8ms warning
            critical=0.010,  # 10ms critical
            window=60,
            min_samples=3
        )
        
        # Response time thresholds
        self.performance_monitor.set_threshold(
            MetricType.RESPONSE_TIME,
            warning=2.0,  # 2 seconds warning
            critical=5.0,  # 5 seconds critical
            window=60,
            min_samples=3
        )
        
        # Accuracy should be 90% or higher (Requirement 1.7)
        self.performance_monitor.set_threshold(
            MetricType.ACCURACY,
            warning=0.85,  # 85% warning
            critical=0.80,  # 80% critical (below confidence threshold)
            window=300,  # 5 minute window
            min_samples=5
        )
        
        # Memory usage thresholds
        self.performance_monitor.set_threshold(
            MetricType.MEMORY_USAGE,
            warning=512.0,  # 512 MB warning
            critical=1024.0,  # 1 GB critical
            window=60,
            min_samples=3
        )
        
        # CPU usage thresholds
        self.performance_monitor.set_threshold(
            MetricType.CPU_USAGE,
            warning=70.0,  # 70% warning
            critical=90.0,  # 90% critical
            window=60,
            min_samples=3
        )
        
        # Error rate thresholds
        self.performance_monitor.set_threshold(
            MetricType.ERROR_RATE,
            warning=0.05,  # 5% warning
            critical=0.10,  # 10% critical
            window=300,
            min_samples=10
        )
    
    def _setup_health_checks(self):
        """Set up health checks for key system components"""
        
        def database_health_check() -> HealthCheckResult:
            """Check database connectivity and performance"""
            try:
                try:
                    from .database import get_db_manager
                except ImportError:
                    from database import get_db_manager
                
                start_time = time.time()
                db_manager = get_db_manager()
                
                # Test database connection with a simple query
                with db_manager.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT 1")
                    cursor.fetchone()
                
                response_time = time.time() - start_time
                
                if response_time > 0.1:  # 100ms
                    return HealthCheckResult(
                        component="database",
                        status="warning",
                        message=f"Database response slow: {response_time:.3f}s",
                        timestamp=time.time(),
                        response_time=response_time
                    )
                
                return HealthCheckResult(
                    component="database",
                    status="healthy",
                    message="Database connection OK",
                    timestamp=time.time(),
                    response_time=response_time
                )
                
            except Exception as e:
                return HealthCheckResult(
                    component="database",
                    status="critical",
                    message=f"Database connection failed: {str(e)}",
                    timestamp=time.time()
                )
        
        def vector_store_health_check() -> HealthCheckResult:
            """Check vector store functionality"""
            try:
                try:
                    from .vector_store import VectorStore
                except ImportError:
                    from vector_store import VectorStore
                
                start_time = time.time()
                vector_store = VectorStore()
                
                # Test vector store with a simple operation
                test_docs = ["test document"]
                test_embeddings = vector_store.embed_documents(test_docs)
                
                response_time = time.time() - start_time
                
                if len(test_embeddings) != 1:
                    return HealthCheckResult(
                        component="vector_store",
                        status="critical",
                        message="Vector store embedding failed",
                        timestamp=time.time(),
                        response_time=response_time
                    )
                
                return HealthCheckResult(
                    component="vector_store",
                    status="healthy",
                    message="Vector store operational",
                    timestamp=time.time(),
                    response_time=response_time
                )
                
            except Exception as e:
                return HealthCheckResult(
                    component="vector_store",
                    status="critical",
                    message=f"Vector store check failed: {str(e)}",
                    timestamp=time.time()
                )
        
        def gemini_api_health_check() -> HealthCheckResult:
            """Check Gemini API connectivity"""
            try:
                try:
                    from .gemini_client import GeminiClient
                except ImportError:
                    from gemini_client import GeminiClient
                
                start_time = time.time()
                client = GeminiClient()
                
                # Test with a simple prompt
                response = client.generate_content(
                    prompt="Say 'OK' if you can respond.",
                    max_tokens=10
                )
                
                response_time = time.time() - start_time
                
                if not response or not response.content:
                    return HealthCheckResult(
                        component="gemini_api",
                        status="critical",
                        message="Gemini API returned empty response",
                        timestamp=time.time(),
                        response_time=response_time
                    )
                
                return HealthCheckResult(
                    component="gemini_api",
                    status="healthy",
                    message="Gemini API responding",
                    timestamp=time.time(),
                    response_time=response_time
                )
                
            except Exception as e:
                return HealthCheckResult(
                    component="gemini_api",
                    status="warning",
                    message=f"Gemini API check failed: {str(e)}",
                    timestamp=time.time()
                )
        
        def log_directory_health_check() -> HealthCheckResult:
            """Check log directory accessibility and disk space"""
            try:
                log_path = Path(self.log_dir)
                
                # Check if log directory exists and is writable
                if not log_path.exists():
                    return HealthCheckResult(
                        component="log_directory",
                        status="critical",
                        message="Log directory does not exist",
                        timestamp=time.time()
                    )
                
                # Test write access
                test_file = log_path / "health_check_test.tmp"
                try:
                    test_file.write_text("test")
                    test_file.unlink()
                except Exception:
                    return HealthCheckResult(
                        component="log_directory",
                        status="critical",
                        message="Log directory not writable",
                        timestamp=time.time()
                    )
                
                # Check disk space (warn if less than 100MB)
                import shutil
                free_space = shutil.disk_usage(log_path).free / (1024 * 1024)  # MB
                
                if free_space < 100:
                    return HealthCheckResult(
                        component="log_directory",
                        status="warning",
                        message=f"Low disk space: {free_space:.1f}MB remaining",
                        timestamp=time.time(),
                        metadata={"free_space_mb": free_space}
                    )
                
                return HealthCheckResult(
                    component="log_directory",
                    status="healthy",
                    message=f"Log directory OK, {free_space:.1f}MB free",
                    timestamp=time.time(),
                    metadata={"free_space_mb": free_space}
                )
                
            except Exception as e:
                return HealthCheckResult(
                    component="log_directory",
                    status="critical",
                    message=f"Log directory check failed: {str(e)}",
                    timestamp=time.time()
                )
        
        # Register health checks
        self.performance_monitor.add_health_check("database", database_health_check)
        self.performance_monitor.add_health_check("vector_store", vector_store_health_check)
        self.performance_monitor.add_health_check("gemini_api", gemini_api_health_check)
        self.performance_monitor.add_health_check("log_directory", log_directory_health_check)
    
    def _console_alert_handler(self, alert):
        """Handle alerts by printing to console"""
        level_colors = {
            AlertLevel.INFO: "\033[94m",      # Blue
            AlertLevel.WARNING: "\033[93m",   # Yellow
            AlertLevel.CRITICAL: "\033[91m"   # Red
        }
        reset_color = "\033[0m"
        
        color = level_colors.get(alert.level, "")
        print(f"{color}[{alert.level.value.upper()}] {alert.component}: {alert.message}{reset_color}")
    
    def _log_alert_handler(self, alert):
        """Handle alerts by logging them"""
        if alert.level == AlertLevel.CRITICAL:
            log_error("performance_alert", alert.message, metadata={
                'component': alert.component,
                'metric_type': alert.metric_type.value,
                'current_value': alert.current_value,
                'threshold': alert.threshold
            })
        else:
            log_info("performance_alert", alert.message, metadata={
                'component': alert.component,
                'metric_type': alert.metric_type.value,
                'current_value': alert.current_value,
                'threshold': alert.threshold
            })
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status including logging and monitoring.
        
        Returns:
            Dictionary with system status information
        """
        if not self.initialized:
            return {"status": "not_initialized", "message": "System not initialized"}
        
        try:
            # Get performance summary
            perf_summary = self.performance_monitor.get_performance_summary()
            
            # Get logging statistics
            log_stats = self.logging_manager.get_log_stats(days=1)
            
            # Determine overall system health
            health_checks = perf_summary.get('health_checks', {})
            critical_components = [
                name for name, result in health_checks.items()
                if result.status == 'critical'
            ]
            
            warning_components = [
                name for name, result in health_checks.items()
                if result.status == 'warning'
            ]
            
            if critical_components:
                overall_status = "critical"
                status_message = f"Critical issues in: {', '.join(critical_components)}"
            elif warning_components:
                overall_status = "warning"
                status_message = f"Warnings in: {', '.join(warning_components)}"
            else:
                overall_status = "healthy"
                status_message = "All systems operational"
            
            return {
                "status": overall_status,
                "message": status_message,
                "timestamp": time.time(),
                "performance_summary": perf_summary,
                "logging_stats": log_stats,
                "critical_components": critical_components,
                "warning_components": warning_components
            }
            
        except Exception as e:
            log_error("system_integrator", f"Failed to get system status: {e}")
            return {
                "status": "error",
                "message": f"Status check failed: {str(e)}",
                "timestamp": time.time()
            }
    
    def shutdown_system(self):
        """Gracefully shutdown logging and monitoring"""
        try:
            if self.performance_monitor:
                self.performance_monitor.stop_monitoring()
            
            if self.logging_manager:
                # Clean up old logs
                self.logging_manager.cleanup_old_logs(days_to_keep=30)
            
            log_info("system_integrator", "System logging and monitoring shutdown complete")
            
        except Exception as e:
            print(f"Error during system shutdown: {e}")
    
    def export_system_metrics(self, output_file: str, hours: int = 24):
        """
        Export comprehensive system metrics for analysis.
        
        Args:
            output_file: Output file path
            hours: Number of hours of data to export
        """
        if not self.initialized:
            raise RuntimeError("System not initialized")
        
        try:
            # Export performance metrics
            perf_export_file = output_file.replace('.json', '_performance.json')
            self.performance_monitor.export_metrics(perf_export_file, hours)
            
            # Get logging statistics
            log_stats = self.logging_manager.get_log_stats(days=hours // 24 or 1)
            
            # Get system status
            system_status = self.get_system_status()
            
            # Combine all data
            combined_export = {
                "export_timestamp": time.time(),
                "hours_exported": hours,
                "system_status": system_status,
                "logging_statistics": log_stats,
                "performance_metrics_file": perf_export_file
            }
            
            import json
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(combined_export, f, indent=2, default=str)
            
            log_info("system_integrator", f"System metrics exported to {output_file}")
            
        except Exception as e:
            log_error("system_integrator", f"Failed to export system metrics: {e}")
            raise


# Global system integrator instance
_system_integrator: Optional[SystemIntegrator] = None


def get_system_integrator() -> SystemIntegrator:
    """Get the global system integrator instance"""
    global _system_integrator
    if _system_integrator is None:
        _system_integrator = SystemIntegrator()
    return _system_integrator


def initialize_system(log_dir: str = "data/logs",
                     log_level: LogLevel = LogLevel.INFO,
                     enable_console: bool = True,
                     monitoring_interval: int = 30) -> bool:
    """
    Initialize the complete FounderForge logging and monitoring system.
    
    Args:
        log_dir: Directory for log files
        log_level: Default logging level
        enable_console: Whether to enable console logging
        monitoring_interval: Performance monitoring interval in seconds
        
    Returns:
        True if initialization successful, False otherwise
    """
    global _system_integrator
    _system_integrator = SystemIntegrator(
        log_dir=log_dir,
        log_level=log_level,
        enable_console=enable_console,
        monitoring_interval=monitoring_interval
    )
    return _system_integrator.initialize_system()


def get_system_status() -> Dict[str, Any]:
    """Get comprehensive system status"""
    return get_system_integrator().get_system_status()


def shutdown_system():
    """Gracefully shutdown the system"""
    get_system_integrator().shutdown_system()


def export_system_metrics(output_file: str, hours: int = 24):
    """Export comprehensive system metrics"""
    get_system_integrator().export_system_metrics(output_file, hours)


if __name__ == "__main__":
    # Example usage
    print("Initializing FounderForge logging and monitoring system...")
    
    if initialize_system():
        print("System initialized successfully!")
        
        # Get system status
        status = get_system_status()
        print(f"System status: {status['status']} - {status['message']}")
        
        # Wait a bit to collect some metrics
        import time
        time.sleep(5)
        
        # Export metrics
        export_system_metrics("system_metrics_export.json", hours=1)
        print("Metrics exported to system_metrics_export.json")
        
        # Shutdown
        shutdown_system()
        print("System shutdown complete.")
    else:
        print("Failed to initialize system!")