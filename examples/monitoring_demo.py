#!/usr/bin/env python3
"""
FounderForge Performance Monitoring Demo

Demonstrates the performance monitoring and alerting capabilities.
Shows how to:
- Set up monitoring thresholds
- Record metrics that trigger alerts
- Monitor system health
- Export performance data
"""

import sys
import time
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from system_integration import initialize_system, get_system_status, shutdown_system
from performance_monitor import (
    get_performance_monitor, MetricType, record_response_time, 
    record_accuracy, record_error_rate, record_database_performance
)
from logging_manager import log_info, log_error, log_performance


def demo_threshold_alerts():
    """Demonstrate threshold-based alerting"""
    print("🔔 Demonstrating threshold alerts...")
    
    monitor = get_performance_monitor()
    
    # Set aggressive thresholds for demo
    monitor.set_threshold(MetricType.RESPONSE_TIME, warning=1.0, critical=2.0, window=30, min_samples=2)
    monitor.set_threshold(MetricType.ACCURACY, warning=0.90, critical=0.85, window=30, min_samples=2)
    monitor.set_threshold(MetricType.ERROR_RATE, warning=0.05, critical=0.10, window=30, min_samples=2)
    
    print("  Setting up aggressive thresholds for demo...")
    print("  - Response time: warning=1.0s, critical=2.0s")
    print("  - Accuracy: warning=90%, critical=85%")
    print("  - Error rate: warning=5%, critical=10%")
    print()
    
    # Record metrics that should trigger warnings
    print("  Recording metrics that should trigger warnings...")
    record_response_time("demo_api", 1.2)  # Above warning threshold
    record_response_time("demo_api", 1.3)  # Above warning threshold
    
    record_accuracy("demo_model", 0.88)  # Below warning threshold
    record_accuracy("demo_model", 0.87)  # Below warning threshold
    
    record_error_rate("demo_service", 0.07)  # Above warning threshold
    record_error_rate("demo_service", 0.08)  # Above warning threshold
    
    time.sleep(1)  # Allow alerts to be processed
    
    # Record metrics that should trigger critical alerts
    print("  Recording metrics that should trigger critical alerts...")
    record_response_time("demo_api", 2.5)  # Above critical threshold
    record_response_time("demo_api", 2.8)  # Above critical threshold
    
    record_accuracy("demo_model", 0.82)  # Below critical threshold
    record_accuracy("demo_model", 0.80)  # Below critical threshold
    
    record_error_rate("demo_service", 0.12)  # Above critical threshold
    record_error_rate("demo_service", 0.15)  # Above critical threshold
    
    time.sleep(1)  # Allow alerts to be processed
    print("  ✓ Alert demonstration completed")


def demo_system_health_monitoring():
    """Demonstrate system health monitoring"""
    print("🏥 Demonstrating system health monitoring...")
    
    status = get_system_status()
    
    print(f"  Overall system status: {status['status']}")
    print(f"  Status message: {status['message']}")
    print()
    
    if 'performance_summary' in status and 'health_checks' in status['performance_summary']:
        health_checks = status['performance_summary']['health_checks']
        print(f"  Health checks ({len(health_checks)} components):")
        
        for component, result in health_checks.items():
            status_icon = {
                'healthy': '✅',
                'warning': '⚠️',
                'critical': '❌'
            }.get(result.status, '❓')
            
            response_time = f" ({result.response_time:.3f}s)" if result.response_time else ""
            print(f"    {status_icon} {component}: {result.status}{response_time}")
            print(f"       {result.message}")
    
    print("  ✓ Health monitoring demonstration completed")


def demo_performance_metrics():
    """Demonstrate performance metrics collection"""
    print("📊 Demonstrating performance metrics collection...")
    
    monitor = get_performance_monitor()
    
    # Simulate various operations with different performance characteristics
    operations = [
        ("fast_query", 0.005, True),
        ("medium_query", 0.025, True),
        ("slow_query", 0.150, True),
        ("failed_query", 0.300, False),
    ]
    
    print("  Simulating database operations...")
    for operation, time_taken, success in operations:
        record_database_performance(operation, time_taken)
        log_performance("database", operation, time_taken, success)
        print(f"    - {operation}: {time_taken:.3f}s ({'✓' if success else '✗'})")
    
    # Simulate API responses
    api_responses = [0.5, 0.8, 1.2, 0.3, 2.1, 0.7, 1.5, 0.9]
    print("  Simulating API responses...")
    for i, response_time in enumerate(api_responses):
        record_response_time("api_endpoint", response_time)
        print(f"    - Request {i+1}: {response_time:.1f}s")
    
    # Simulate model accuracy over time
    accuracy_scores = [0.95, 0.92, 0.88, 0.91, 0.85, 0.89, 0.93, 0.87]
    print("  Simulating model accuracy scores...")
    for i, accuracy in enumerate(accuracy_scores):
        record_accuracy("ai_model", accuracy)
        print(f"    - Batch {i+1}: {accuracy:.1%}")
    
    print("  ✓ Performance metrics demonstration completed")


def demo_metrics_analysis():
    """Demonstrate metrics analysis"""
    print("📈 Demonstrating metrics analysis...")
    
    monitor = get_performance_monitor()
    
    # Get statistics for different metrics
    metrics_to_analyze = [
        (MetricType.RESPONSE_TIME, "api_endpoint"),
        (MetricType.ACCURACY, "ai_model"),
        (MetricType.DATABASE_PERFORMANCE, "fast_query"),
    ]
    
    for metric_type, component in metrics_to_analyze:
        stats = monitor.get_metric_statistics(metric_type, component, window_seconds=300)
        
        if stats and stats['count'] > 0:
            print(f"  📊 {metric_type.value} for {component}:")
            print(f"     Count: {stats['count']}")
            print(f"     Mean: {stats['mean']:.3f}")
            print(f"     Median: {stats['median']:.3f}")
            print(f"     Min: {stats['min']:.3f}")
            print(f"     Max: {stats['max']:.3f}")
            print(f"     Std Dev: {stats['std_dev']:.3f}")
            print()
    
    # Get comprehensive performance summary
    summary = monitor.get_performance_summary()
    
    print("  📋 System Performance Summary:")
    if 'system_metrics' in summary:
        sys_metrics = summary['system_metrics']
        print(f"     CPU Usage: {sys_metrics.get('cpu_usage', 0):.1f}%")
        print(f"     Memory Usage: {sys_metrics.get('memory_usage', 0):.1f} MB")
        print(f"     Open Files: {sys_metrics.get('open_files', 0)}")
        print(f"     Threads: {sys_metrics.get('threads', 0)}")
    
    if 'recent_alerts' in summary:
        alert_count = len(summary['recent_alerts'])
        print(f"     Recent Alerts: {alert_count}")
    
    print("  ✓ Metrics analysis demonstration completed")


def main():
    """Main demo function"""
    print("FounderForge Performance Monitoring Demo")
    print("=" * 50)
    
    # Use temporary directory for demo
    temp_log_dir = tempfile.mkdtemp(prefix="founderforge_monitoring_demo_")
    print(f"Using temporary log directory: {temp_log_dir}")
    print()
    
    try:
        # Initialize system
        print("🚀 Initializing monitoring system...")
        if not initialize_system(
            log_dir=temp_log_dir,
            enable_console=False,  # Reduce noise for demo
            monitoring_interval=5
        ):
            print("❌ Failed to initialize system")
            return
        
        print("✅ System initialized successfully")
        print()
        
        # Run demonstrations
        demo_performance_metrics()
        print()
        
        demo_threshold_alerts()
        print()
        
        # Wait for monitoring to collect data
        print("⏳ Waiting for monitoring data collection...")
        time.sleep(6)
        
        demo_system_health_monitoring()
        print()
        
        demo_metrics_analysis()
        print()
        
        # Export metrics for analysis
        print("💾 Exporting metrics for analysis...")
        export_file = Path(temp_log_dir) / "demo_metrics_export.json"
        
        from system_integration import export_system_metrics
        export_system_metrics(str(export_file), hours=1)
        
        if export_file.exists():
            size = export_file.stat().st_size
            print(f"   ✅ Metrics exported to {export_file.name} ({size} bytes)")
        
        print()
        print("=" * 50)
        print("✅ Performance monitoring demo completed successfully!")
        print()
        print("Key features demonstrated:")
        print("  🔔 Threshold-based alerting")
        print("  🏥 System health monitoring")
        print("  📊 Performance metrics collection")
        print("  📈 Real-time metrics analysis")
        print("  💾 Metrics export for analysis")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        print("\n🧹 Shutting down system...")
        shutdown_system()
        
        # Clean up temp directory
        import shutil
        try:
            shutil.rmtree(temp_log_dir, ignore_errors=True)
            print("✅ Cleanup completed")
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")


if __name__ == "__main__":
    main()