#!/usr/bin/env python3
"""
A/B Testing Experiment Management Script for FounderForge AI Cofounder

This script provides a command-line interface for managing A/B testing experiments:
- Create new experiments with variant scripts
- Run experiments with different configurations
- Monitor experiment progress and results
- Generate quick analysis reports

Usage:
    python scripts/manage_ab_experiments.py create --name "Test Name" --description "Description"
    python scripts/manage_ab_experiments.py run --experiment-id exp_123 --iterations 5
    python scripts/manage_ab_experiments.py analyze --experiment-id exp_123
    python scripts/manage_ab_experiments.py list [--status active]
"""

import argparse
import sys
import os
import datetime
import json
from pathlib import Path
from typing import List, Dict, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from experiment_manager import ExperimentManager
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/logs/ab_experiments.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class ABTestManager:
    """Command-line interface for A/B testing experiments"""
    
    def __init__(self):
        self.experiment_manager = ExperimentManager()
    
    def create_experiment(self, name: str, description: str, 
                         variant_a_script: str = None, variant_b_script: str = None,
                         success_metrics: List[str] = None):
        """Create a new A/B testing experiment"""
        
        # Default success metrics if not provided
        if success_metrics is None:
            success_metrics = ["accuracy", "response_time", "user_satisfaction"]
        
        # Default script templates if not provided
        if variant_a_script is None:
            variant_a_script = self._get_default_variant_script("A")
        
        if variant_b_script is None:
            variant_b_script = self._get_default_variant_script("B")
        
        try:
            experiment_id = self.experiment_manager.create_experiment(
                name=name,
                description=description,
                variant_a_script=variant_a_script,
                variant_b_script=variant_b_script,
                success_metrics=success_metrics
            )
            
            print(f"✓ Created experiment: {experiment_id}")
            print(f"  Name: {name}")
            print(f"  Description: {description}")
            print(f"  Success Metrics: {', '.join(success_metrics)}")
            print(f"\nNext steps:")
            print(f"1. Edit variant scripts in: data/experiments/scripts/variant_a/{experiment_id}/")
            print(f"2. Edit variant scripts in: data/experiments/scripts/variant_b/{experiment_id}/")
            print(f"3. Run experiment: python scripts/manage_ab_experiments.py run --experiment-id {experiment_id}")
            
            return experiment_id
            
        except Exception as e:
            logger.error(f"Failed to create experiment: {e}")
            return None
    
    def _get_default_variant_script(self, variant: str) -> str:
        """Generate default script template for a variant"""
        return f'''#!/usr/bin/env python3
"""
Variant {variant} Script for A/B Testing Experiment

This script should implement the variant {variant} logic and output metrics as JSON.
The last line of stdout should be a JSON object with metrics.

Example output:
{{"accuracy": 0.85, "response_time": 1.2, "user_satisfaction": 4.2}}
"""

import json
import time
import random
import sys

def run_variant_{variant.lower()}():
    """Implement variant {variant} logic here"""
    
    # Simulate some processing
    start_time = time.time()
    
    # TODO: Replace with actual variant {variant} implementation
    # This is just a template - implement your actual logic here
    
    # Simulate different performance characteristics
    if "{variant}" == "A":
        # Variant A: Baseline implementation
        accuracy = random.uniform(0.80, 0.90)
        response_time = random.uniform(1.0, 2.0)
        user_satisfaction = random.uniform(3.5, 4.5)
    else:
        # Variant B: Experimental implementation
        accuracy = random.uniform(0.85, 0.95)
        response_time = random.uniform(0.8, 1.5)
        user_satisfaction = random.uniform(4.0, 5.0)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Return metrics as JSON (this should be the last line of output)
    metrics = {{
        "accuracy": accuracy,
        "response_time": response_time,
        "user_satisfaction": user_satisfaction,
        "execution_time": execution_time
    }}
    
    return metrics

if __name__ == "__main__":
    try:
        result = run_variant_{variant.lower()}()
        
        # Output metrics as JSON (must be last line for parsing)
        print(json.dumps(result))
        
    except Exception as e:
        print(f"Error in variant {variant}: {{e}}", file=sys.stderr)
        sys.exit(1)
'''
    
    def run_experiment(self, experiment_id: str, variant: str = "both", 
                      iterations: int = 5):
        """Run an experiment with specified parameters"""
        
        try:
            print(f"Running experiment {experiment_id}...")
            print(f"Variant: {variant}, Iterations: {iterations}")
            
            results = self.experiment_manager.run_experiment(
                experiment_id=experiment_id,
                variant=variant,
                iterations=iterations
            )
            
            # Display results summary
            successful_runs = sum(1 for r in results if r.success)
            total_runs = len(results)
            
            print(f"\n✓ Experiment completed!")
            print(f"  Total runs: {total_runs}")
            print(f"  Successful runs: {successful_runs}")
            print(f"  Success rate: {successful_runs/total_runs*100:.1f}%")
            
            # Show variant breakdown
            variant_a_runs = [r for r in results if r.variant == "A"]
            variant_b_runs = [r for r in results if r.variant == "B"]
            
            if variant_a_runs:
                avg_time_a = sum(r.execution_time for r in variant_a_runs) / len(variant_a_runs)
                print(f"  Variant A: {len(variant_a_runs)} runs, avg time: {avg_time_a:.2f}s")
            
            if variant_b_runs:
                avg_time_b = sum(r.execution_time for r in variant_b_runs) / len(variant_b_runs)
                print(f"  Variant B: {len(variant_b_runs)} runs, avg time: {avg_time_b:.2f}s")
            
            print(f"\nNext steps:")
            print(f"1. Analyze results: python scripts/manage_ab_experiments.py analyze --experiment-id {experiment_id}")
            print(f"2. View detailed results in: data/experiments/results/")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to run experiment: {e}")
            return None
    
    def analyze_experiment(self, experiment_id: str):
        """Analyze experiment results and display summary"""
        
        try:
            analysis = self.experiment_manager.analyze_experiment(experiment_id)
            
            print(f"Analysis for experiment: {experiment_id}")
            print(f"Analysis date: {analysis.analysis_date}")
            print(f"Confidence level: {analysis.confidence_level:.2f}")
            print(f"\nRecommendation: {analysis.recommendation}")
            
            # Display statistical summary
            stats = analysis.statistical_summary
            
            print(f"\n--- Variant A Results ---")
            variant_a = stats.get("variant_a", {})
            print(f"Total runs: {variant_a.get('total_runs', 0)}")
            print(f"Successful runs: {variant_a.get('successful_runs', 0)}")
            print(f"Average execution time: {variant_a.get('avg_execution_time', 0):.2f}s")
            
            print(f"\n--- Variant B Results ---")
            variant_b = stats.get("variant_b", {})
            print(f"Total runs: {variant_b.get('total_runs', 0)}")
            print(f"Successful runs: {variant_b.get('successful_runs', 0)}")
            print(f"Average execution time: {variant_b.get('avg_execution_time', 0):.2f}s")
            
            print(f"\n--- Metric Comparisons ---")
            comparisons = stats.get("comparison", {})
            for metric, comparison in comparisons.items():
                improvement = comparison.get("improvement_percent", 0)
                better_variant = comparison.get("better_variant", "Unknown")
                significant = comparison.get("significant", False)
                
                status = "✓ Significant" if significant else "○ Not significant"
                print(f"{metric}: {improvement:+.1f}% (Variant {better_variant} better) {status}")
            
            print(f"\nDetailed analysis saved to: data/experiments/analysis/")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze experiment: {e}")
            return None
    
    def list_experiments(self, status: str = None):
        """List experiments with optional status filter"""
        
        try:
            experiments = self.experiment_manager.list_experiments(status=status)
            
            if not experiments:
                print("No experiments found.")
                return
            
            print(f"Found {len(experiments)} experiments:")
            print()
            
            for exp in experiments:
                status_icon = {
                    "active": "🟢",
                    "completed": "✅", 
                    "failed": "❌",
                    "cancelled": "⏹️",
                    "archived": "📦"
                }.get(exp["status"], "❓")
                
                print(f"{status_icon} {exp['name']} ({exp['id']})")
                print(f"   Status: {exp['status']}")
                print(f"   Started: {exp['start_date']}")
                if exp['end_date']:
                    print(f"   Ended: {exp['end_date']}")
                print(f"   Description: {exp['description']}")
                print()
            
        except Exception as e:
            logger.error(f"Failed to list experiments: {e}")
    
    def get_stats(self):
        """Display overall experiment statistics"""
        
        try:
            stats = self.experiment_manager.get_experiment_stats()
            
            print("Experiment Statistics:")
            print(f"Total experiments: {stats.get('total_experiments', 0)}")
            print(f"Total results: {stats.get('total_results', 0)}")
            print(f"Success rate: {stats.get('success_rate', 0):.1f}%")
            
            print(f"\nBy status:")
            status_counts = stats.get('status_counts', {})
            for status, count in status_counts.items():
                print(f"  {status}: {count}")
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")

def main():
    """Main function for command-line interface"""
    parser = argparse.ArgumentParser(description="Manage A/B testing experiments")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Create experiment command
    create_parser = subparsers.add_parser("create", help="Create new experiment")
    create_parser.add_argument("--name", required=True, help="Experiment name")
    create_parser.add_argument("--description", required=True, help="Experiment description")
    create_parser.add_argument("--metrics", nargs="+", default=["accuracy", "response_time", "user_satisfaction"],
                              help="Success metrics to track")
    
    # Run experiment command
    run_parser = subparsers.add_parser("run", help="Run experiment")
    run_parser.add_argument("--experiment-id", required=True, help="Experiment ID to run")
    run_parser.add_argument("--variant", choices=["A", "B", "both"], default="both",
                           help="Which variant to run")
    run_parser.add_argument("--iterations", type=int, default=5,
                           help="Number of iterations per variant")
    
    # Analyze experiment command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze experiment results")
    analyze_parser.add_argument("--experiment-id", required=True, help="Experiment ID to analyze")
    
    # List experiments command
    list_parser = subparsers.add_parser("list", help="List experiments")
    list_parser.add_argument("--status", choices=["active", "completed", "failed", "cancelled", "archived"],
                            help="Filter by status")
    
    # Stats command
    subparsers.add_parser("stats", help="Show experiment statistics")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    manager = ABTestManager()
    
    try:
        if args.command == "create":
            manager.create_experiment(
                name=args.name,
                description=args.description,
                success_metrics=args.metrics
            )
        
        elif args.command == "run":
            manager.run_experiment(
                experiment_id=args.experiment_id,
                variant=args.variant,
                iterations=args.iterations
            )
        
        elif args.command == "analyze":
            manager.analyze_experiment(args.experiment_id)
        
        elif args.command == "list":
            manager.list_experiments(status=args.status)
        
        elif args.command == "stats":
            manager.get_stats()
        
    except Exception as e:
        logger.error(f"Command failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()