#!/usr/bin/env python3
"""
Script to run the FounderForge AI evaluation harness.

This script provides a command-line interface for running comprehensive
AI quality assessments, including accuracy measurement, confidence threshold
validation, and fallback mechanism testing.
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation_harness import EvaluationHarness
from src.agents import AgentOrchestrator
from src.context_manager import ContextAssembler
from src.confidence_manager import ConfidenceManager
from src.gemini_client import GeminiClient, MockMode
from src.database import DatabaseManager
from config.settings import settings


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('data/evaluation_logs.log')
        ]
    )


async def initialize_components(mock_mode: bool = False):
    """Initialize all required components for evaluation."""
    # Initialize database
    db_manager = DatabaseManager("data/founderforge.db")
    
    # Initialize Gemini client
    api_key = settings.get_setting('gemini_api_key')
    if not api_key and not mock_mode:
        raise ValueError("Gemini API key not found. Set GEMINI_API_KEY environment variable or use --mock-mode")
    
    gemini_client = GeminiClient(
        api_key=api_key or "mock_key",
        mock_mode=MockMode.SUCCESS if mock_mode else MockMode.DISABLED
    )
    
    # Initialize context manager
    context_manager = ContextAssembler(
        db_manager=db_manager,
        vector_store_path="data/vector_index"
    )
    
    # Initialize confidence manager
    confidence_manager = ConfidenceManager()
    
    # Initialize agent orchestrator
    agent_orchestrator = AgentOrchestrator(
        gemini_client=gemini_client,
        context_manager=context_manager,
        confidence_manager=confidence_manager,
        db_path="data/founderforge.db"
    )
    
    return gemini_client, context_manager, confidence_manager, agent_orchestrator


async def run_full_evaluation(args):
    """Run full evaluation suite."""
    print("🚀 Starting FounderForge AI Evaluation")
    print("=" * 50)
    
    # Initialize components
    try:
        gemini_client, context_manager, confidence_manager, agent_orchestrator = await initialize_components(args.mock_mode)
        print("✅ Components initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize components: {e}")
        return 1
    
    # Initialize evaluation harness
    harness = EvaluationHarness(
        gemini_client=gemini_client,
        context_manager=context_manager,
        confidence_manager=confidence_manager,
        agent_orchestrator=agent_orchestrator,
        test_data_path=args.test_data_path
    )
    
    print(f"📊 Loaded {len(harness.test_scenarios)} test scenarios")
    
    # Run evaluation
    try:
        print("\n🔍 Running evaluation...")
        summary = await harness.run_evaluation(
            scenario_ids=args.scenarios,
            mock_mode=args.mock_mode
        )
        
        # Display results
        print("\n📈 Evaluation Results:")
        print(f"  Total Scenarios: {summary.total_scenarios}")
        print(f"  Passed: {summary.passed} ({summary.passed/summary.total_scenarios*100:.1f}%)")
        print(f"  Failed: {summary.failed} ({summary.failed/summary.total_scenarios*100:.1f}%)")
        print(f"  Partial: {summary.partial} ({summary.partial/summary.total_scenarios*100:.1f}%)")
        print(f"  Errors: {summary.errors} ({summary.errors/summary.total_scenarios*100:.1f}%)")
        print(f"  Overall Accuracy: {summary.overall_accuracy:.3f}")
        print(f"  Average Confidence: {summary.average_confidence:.3f}")
        print(f"  Average Execution Time: {summary.average_execution_time:.2f}s")
        print(f"  Fallback Usage Rate: {summary.fallback_usage_rate:.3f}")
        
        # Check if targets are met
        target_met = summary.overall_accuracy >= harness.target_accuracy
        print(f"\n🎯 Target Achievement:")
        print(f"  Accuracy Target (90%): {'✅ ACHIEVED' if target_met else '❌ MISSED'}")
        
        # Generate and save report if requested
        if args.generate_report:
            report = harness.generate_report(summary)
            report_file = Path(args.test_data_path) / f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            with open(report_file, 'w') as f:
                f.write(report)
            print(f"📄 Detailed report saved to: {report_file}")
        
        return 0 if target_met else 1
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return 1


async def run_confidence_validation(args):
    """Run confidence threshold validation."""
    print("🔍 Starting Confidence Threshold Validation")
    print("=" * 50)
    
    # Initialize components
    try:
        gemini_client, context_manager, confidence_manager, agent_orchestrator = await initialize_components(args.mock_mode)
        print("✅ Components initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize components: {e}")
        return 1
    
    # Initialize evaluation harness
    harness = EvaluationHarness(
        gemini_client=gemini_client,
        context_manager=context_manager,
        confidence_manager=confidence_manager,
        agent_orchestrator=agent_orchestrator,
        test_data_path=args.test_data_path
    )
    
    try:
        print("\n🎯 Testing confidence thresholds...")
        validation_result = await harness.run_confidence_threshold_validation()
        
        # Display results
        print("\n📊 Confidence Threshold Validation Results:")
        print(f"  Current Threshold: {validation_result['current_threshold']}")
        print(f"  Optimal Threshold: {validation_result['optimal_threshold']}")
        print(f"  Recommendation: {validation_result['recommendation']}")
        
        print("\n📈 Threshold Performance:")
        for threshold, metrics in validation_result['threshold_results'].items():
            print(f"  {threshold:.1f}: Accuracy={metrics['accuracy']:.3f}, "
                  f"Passed={metrics['passed']}, Fallback Rate={metrics['fallback_rate']:.3f}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Confidence validation failed: {e}")
        return 1


async def run_fallback_testing(args):
    """Run fallback mechanism testing."""
    print("🛡️ Starting Fallback Mechanism Testing")
    print("=" * 50)
    
    # Initialize components
    try:
        gemini_client, context_manager, confidence_manager, agent_orchestrator = await initialize_components(args.mock_mode)
        print("✅ Components initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize components: {e}")
        return 1
    
    # Initialize evaluation harness
    harness = EvaluationHarness(
        gemini_client=gemini_client,
        context_manager=context_manager,
        confidence_manager=confidence_manager,
        agent_orchestrator=agent_orchestrator,
        test_data_path=args.test_data_path
    )
    
    try:
        print("\n🔧 Testing fallback mechanisms...")
        fallback_results = await harness.test_fallback_mechanisms()
        
        # Display results
        print("\n📊 Fallback Mechanism Test Results:")
        print(f"  Total Tests: {fallback_results['total_tests']}")
        print(f"  Passed Tests: {fallback_results['passed_tests']}")
        print(f"  Success Rate: {fallback_results['success_rate']:.3f}")
        
        print("\n📋 Detailed Results:")
        for result in fallback_results['detailed_results']:
            status = "✅" if result['test_passed'] else "❌"
            print(f"  {status} {result['reason']}: Expected={result['expected_fallback']}, "
                  f"Actual={result['actual_fallback']}")
        
        success = fallback_results['success_rate'] >= 0.8  # 80% success rate target
        print(f"\n🎯 Fallback Testing: {'✅ PASSED' if success else '❌ FAILED'}")
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"❌ Fallback testing failed: {e}")
        return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="FounderForge AI Evaluation Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full evaluation
  python scripts/run_evaluation.py full

  # Run specific scenarios
  python scripts/run_evaluation.py full --scenarios funding_001 planning_001

  # Run with mock mode (no API calls)
  python scripts/run_evaluation.py full --mock-mode

  # Validate confidence thresholds
  python scripts/run_evaluation.py confidence

  # Test fallback mechanisms
  python scripts/run_evaluation.py fallback

  # Generate detailed report
  python scripts/run_evaluation.py full --generate-report
        """
    )
    
    parser.add_argument(
        'command',
        choices=['full', 'confidence', 'fallback'],
        help='Evaluation command to run'
    )
    
    parser.add_argument(
        '--scenarios',
        nargs='*',
        help='Specific scenario IDs to test (default: all)'
    )
    
    parser.add_argument(
        '--mock-mode',
        action='store_true',
        help='Use mock responses instead of real API calls'
    )
    
    parser.add_argument(
        '--test-data-path',
        default='data/evaluation',
        help='Path to test data directory (default: data/evaluation)'
    )
    
    parser.add_argument(
        '--generate-report',
        action='store_true',
        help='Generate detailed evaluation report'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.verbose)
    
    # Ensure test data directory exists
    Path(args.test_data_path).mkdir(parents=True, exist_ok=True)
    
    # Run appropriate command
    if args.command == 'full':
        return asyncio.run(run_full_evaluation(args))
    elif args.command == 'confidence':
        return asyncio.run(run_confidence_validation(args))
    elif args.command == 'fallback':
        return asyncio.run(run_fallback_testing(args))


if __name__ == "__main__":
    sys.exit(main())