#!/usr/bin/env python3
"""
Test script for daily planning agent workflow.
Tests the complete workflow including Planner, Tool-Caller, and Coach agents.
"""

import asyncio
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Import required modules
try:
    from src.daily_planner import DailyPlanningWorkflow, DailyPlanningInput, create_sample_planning_input
    from src.agents import AgentOrchestrator
    from src.gemini_client import GeminiClient, MockMode
    from src.context_manager import ContextAssembler
    from src.confidence_manager import ConfidenceManager
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.info("Make sure you're running from the project root directory")
    exit(1)


class DailyPlanningTester:
    """Test harness for daily planning workflow."""
    
    def __init__(self):
        """Initialize test components."""
        self.setup_test_environment()
        self.initialize_components()
        
    def setup_test_environment(self):
        """Setup test data directories and files."""
        # Create test data directories
        test_data_dir = Path("data/business_data")
        test_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create sample business data files
        self.create_sample_data_files(test_data_dir)
        
    def create_sample_data_files(self, data_dir: Path):
        """Create sample data files for testing."""
        
        # Sample user business data
        user_business_data = {
            "company_name": "TechStart Inc",
            "stage": "early_stage",
            "industry": "fintech",
            "team_size": 8,
            "funding_stage": "seed",
            "monthly_revenue": 15000,
            "key_metrics": {
                "user_growth": "15% MoM",
                "retention_rate": "85%",
                "burn_rate": "$25k/month"
            }
        }
        
        with open(data_dir / "test_user_001_business.json", 'w') as f:
            json.dump(user_business_data, f, indent=2)
        
        # Sample user goals
        user_goals = {
            "short_term": [
                "Launch MVP by end of quarter",
                "Onboard 100 beta users",
                "Finalize Series A pitch deck"
            ],
            "long_term": [
                "Raise Series A funding",
                "Expand to 3 new markets",
                "Build team to 25 people"
            ]
        }
        
        with open(data_dir / "test_user_001_goals.json", 'w') as f:
            json.dump(user_goals, f, indent=2)
        
        # Sample user activities
        user_activities = {
            "recent_activities": [
                {
                    "date": "2024-01-15",
                    "activity": "Completed user research interviews",
                    "outcome": "Identified 3 key feature requests"
                },
                {
                    "date": "2024-01-14",
                    "activity": "Team standup meeting",
                    "outcome": "Aligned on sprint priorities"
                },
                {
                    "date": "2024-01-13",
                    "activity": "Investor call with ABC Ventures",
                    "outcome": "Positive feedback, follow-up scheduled"
                }
            ]
        }
        
        with open(data_dir / "test_user_001_activities.json", 'w') as f:
            json.dump(user_activities, f, indent=2)
        
        # Sample industry insights
        industry_insights = {
            "fintech_trends": [
                "AI-powered financial planning tools gaining traction",
                "Regulatory compliance becoming more important",
                "Open banking APIs creating new opportunities"
            ],
            "best_practices": [
                "Focus on user experience and simplicity",
                "Build strong security from day one",
                "Establish partnerships with financial institutions early"
            ]
        }
        
        with open(data_dir / "industry_insights.json", 'w') as f:
            json.dump(industry_insights, f, indent=2)
        
        logger.info("Sample data files created successfully")
    
    def initialize_components(self):
        """Initialize all required components for testing."""
        # Initialize with mock mode for testing
        self.gemini_client = GeminiClient(mock_mode=MockMode.SUCCESS)
        self.context_manager = ContextAssembler()
        self.confidence_manager = ConfidenceManager()
        
        # Initialize agent orchestrator
        self.orchestrator = AgentOrchestrator(
            self.gemini_client,
            self.context_manager,
            self.confidence_manager
        )
        
        # Initialize daily planning workflow
        self.workflow = DailyPlanningWorkflow(self.orchestrator)
        
        logger.info("Test components initialized successfully")
    
    async def test_basic_daily_planning(self) -> Dict[str, Any]:
        """Test basic daily planning workflow."""
        logger.info("=== Testing Basic Daily Planning ===")
        
        # Create sample planning input
        planning_input = await create_sample_planning_input()
        
        # Add user context
        user_context = {
            "business_info": {
                "stage": "early_stage",
                "industry": "fintech",
                "company_name": "TechStart Inc",
                "team_size": 8
            },
            "goals": [
                "Launch MVP by end of quarter",
                "Raise Series A funding"
            ],
            "preferences": {
                "communication_style": "professional",
                "response_length": "detailed"
            }
        }
        
        # Execute workflow
        start_time = time.time()
        daily_plan, workflow_result = await self.workflow.generate_daily_plan(
            planning_input, user_context
        )
        execution_time = time.time() - start_time
        
        # Validate results
        test_results = {
            "test_name": "basic_daily_planning",
            "success": workflow_result.success,
            "execution_time": execution_time,
            "within_time_limit": execution_time < 60.0,  # Under 1 minute
            "plan_generated": daily_plan is not None,
            "action_items_count": len(daily_plan.action_items),
            "has_time_blocks": len(daily_plan.time_blocks) > 0,
            "has_coaching_message": len(daily_plan.motivational_message) > 0,
            "confidence_score": daily_plan.confidence_score,
            "agent_chain_executed": self._validate_agent_chain(workflow_result)
        }
        
        logger.info(f"Basic planning test completed in {execution_time:.2f}s")
        logger.info(f"Generated {len(daily_plan.action_items)} action items")
        logger.info(f"Confidence score: {daily_plan.confidence_score:.2f}")
        
        return test_results
    
    async def test_parallel_planning(self) -> Dict[str, Any]:
        """Test parallel task processing capability."""
        logger.info("=== Testing Parallel Planning ===")
        
        # Create multiple planning inputs
        planning_inputs = []
        for i in range(3):
            planning_input = DailyPlanningInput(
                user_id=f"test_user_{i:03d}",
                current_priorities=[
                    f"Priority task {i+1}",
                    f"Secondary task {i+1}",
                    f"Follow-up task {i+1}"
                ],
                available_time="6 hours",
                energy_level="high",
                focus_areas=["product", "business"]
            )
            planning_inputs.append(planning_input)
        
        # Execute parallel planning
        start_time = time.time()
        results = await self.workflow.execute_parallel_planning(planning_inputs, max_workers=3)
        execution_time = time.time() - start_time
        
        # Validate results
        successful_plans = sum(1 for plan, result in results if result.success)
        
        test_results = {
            "test_name": "parallel_planning",
            "total_plans": len(planning_inputs),
            "successful_plans": successful_plans,
            "execution_time": execution_time,
            "within_time_limit": execution_time < 60.0,  # Under 1 minute for all
            "average_time_per_plan": execution_time / len(planning_inputs),
            "parallel_efficiency": execution_time < (len(planning_inputs) * 20.0)  # Should be faster than sequential
        }
        
        logger.info(f"Parallel planning test completed in {execution_time:.2f}s")
        logger.info(f"Successfully generated {successful_plans}/{len(planning_inputs)} plans")
        
        return test_results
    
    async def test_agent_chain_integration(self) -> Dict[str, Any]:
        """Test that Planner, Tool-Caller, and Coach agents are properly chained."""
        logger.info("=== Testing Agent Chain Integration ===")
        
        # Create planning input that should trigger all agents
        planning_input = DailyPlanningInput(
            user_id="test_user_chain",
            current_priorities=[
                "Analyze customer feedback data",  # Should trigger Tool-Caller
                "Prepare strategic presentation",   # Should trigger Planner
                "Team motivation session"          # Should trigger Coach
            ],
            available_time="8 hours",
            energy_level="high",
            upcoming_deadlines=["Board meeting next week"],
            focus_areas=["product", "team", "strategy"]
        )
        
        user_context = {
            "business_info": {
                "stage": "growth",
                "industry": "saas",
                "team_size": 15
            }
        }
        
        # Execute workflow
        daily_plan, workflow_result = await self.workflow.generate_daily_plan(
            planning_input, user_context
        )
        
        # Analyze agent execution
        agent_outputs = workflow_result.result_data.get("agent_outputs", {})
        
        test_results = {
            "test_name": "agent_chain_integration",
            "planner_executed": "planner_output" in agent_outputs,
            "tool_caller_executed": "tool_caller_output" in agent_outputs,
            "coach_executed": "coach_output" in agent_outputs,
            "orchestrator_executed": "orchestrator_output" in agent_outputs,
            "all_agents_chained": len(agent_outputs) >= 3,
            "plan_quality": self._assess_plan_quality(daily_plan),
            "coaching_personalized": self._assess_coaching_personalization(agent_outputs.get("coach_output", {}))
        }
        
        logger.info(f"Agent chain test completed")
        logger.info(f"Agents executed: {list(agent_outputs.keys())}")
        
        return test_results
    
    async def test_local_data_integration(self) -> Dict[str, Any]:
        """Test integration with local data sources."""
        logger.info("=== Testing Local Data Integration ===")
        
        planning_input = DailyPlanningInput(
            user_id="test_user_001",  # Has sample data files
            current_priorities=["Review business metrics", "Plan product roadmap"],
            available_time="6 hours",
            energy_level="medium"
        )
        
        # Execute workflow
        daily_plan, workflow_result = await self.workflow.generate_daily_plan(planning_input)
        
        # Check if local data was integrated
        task_data = workflow_result.result_data.get("task_data", {})
        
        test_results = {
            "test_name": "local_data_integration",
            "business_data_loaded": "business_data" in task_data,
            "industry_insights_loaded": "industry_insights" in task_data,
            "recent_activities_loaded": "recent_activities" in task_data,
            "planning_templates_loaded": "planning_templates" in task_data,
            "data_influenced_plan": self._check_data_influence(daily_plan, task_data),
            "plan_saved_locally": self._check_plan_saved(daily_plan)
        }
        
        logger.info("Local data integration test completed")
        
        return test_results
    
    async def test_performance_requirements(self) -> Dict[str, Any]:
        """Test performance requirements (under 1-minute completion)."""
        logger.info("=== Testing Performance Requirements ===")
        
        # Test multiple scenarios with timing
        scenarios = [
            ("simple", DailyPlanningInput(user_id="perf_test_1", current_priorities=["Task 1", "Task 2"])),
            ("medium", DailyPlanningInput(user_id="perf_test_2", current_priorities=["Task 1", "Task 2", "Task 3", "Task 4"], energy_level="high")),
            ("complex", DailyPlanningInput(user_id="perf_test_3", current_priorities=["Complex analysis", "Strategic planning", "Team coordination", "Customer research", "Product development"], available_time="8 hours", energy_level="high"))
        ]
        
        performance_results = []
        
        for scenario_name, planning_input in scenarios:
            start_time = time.time()
            daily_plan, workflow_result = await self.workflow.generate_daily_plan(planning_input)
            execution_time = time.time() - start_time
            
            performance_results.append({
                "scenario": scenario_name,
                "execution_time": execution_time,
                "within_limit": execution_time < 60.0,
                "success": workflow_result.success,
                "action_items_count": len(daily_plan.action_items)
            })
        
        # Calculate overall performance metrics
        avg_time = sum(r["execution_time"] for r in performance_results) / len(performance_results)
        all_within_limit = all(r["within_limit"] for r in performance_results)
        
        test_results = {
            "test_name": "performance_requirements",
            "scenarios_tested": len(scenarios),
            "average_execution_time": avg_time,
            "all_within_time_limit": all_within_limit,
            "fastest_time": min(r["execution_time"] for r in performance_results),
            "slowest_time": max(r["execution_time"] for r in performance_results),
            "scenario_results": performance_results
        }
        
        logger.info(f"Performance test completed - Average time: {avg_time:.2f}s")
        
        return test_results
    
    def _validate_agent_chain(self, workflow_result) -> bool:
        """Validate that the agent chain was properly executed."""
        agent_outputs = workflow_result.result_data.get("agent_outputs", {})
        
        # Check for key agents in daily planning workflow
        required_agents = ["planner_output"]  # Minimum requirement
        optional_agents = ["tool_caller_output", "coach_output", "orchestrator_output"]
        
        has_required = all(agent in agent_outputs for agent in required_agents)
        has_optional = any(agent in agent_outputs for agent in optional_agents)
        
        return has_required and has_optional
    
    def _assess_plan_quality(self, daily_plan) -> Dict[str, bool]:
        """Assess the quality of the generated daily plan."""
        return {
            "has_action_items": len(daily_plan.action_items) > 0,
            "has_priorities": any(item.priority in ["high", "medium", "low"] for item in daily_plan.action_items),
            "has_time_estimates": any(item.estimated_time for item in daily_plan.action_items),
            "has_categories": any(item.category for item in daily_plan.action_items),
            "has_time_blocks": len(daily_plan.time_blocks) > 0,
            "has_success_metrics": len(daily_plan.success_metrics) > 0
        }
    
    def _assess_coaching_personalization(self, coach_output) -> bool:
        """Assess if coaching output is personalized."""
        message = coach_output.get("message", "")
        
        # Check for personalization indicators
        personalization_indicators = [
            len(message) > 50,  # Substantial message
            "you" in message.lower(),  # Direct address
            any(word in message.lower() for word in ["goal", "business", "team", "success"])  # Relevant content
        ]
        
        return sum(personalization_indicators) >= 2
    
    def _check_data_influence(self, daily_plan, task_data) -> bool:
        """Check if local data influenced the plan generation."""
        business_data = task_data.get("business_data", {})
        
        # Simple check - if business data exists and plan has relevant categories
        if business_data and daily_plan.action_items:
            categories = [item.category for item in daily_plan.action_items]
            return len(set(categories)) > 1  # Multiple categories suggest data influence
        
        return False
    
    def _check_plan_saved(self, daily_plan) -> bool:
        """Check if the plan was saved locally."""
        plans_dir = Path("data/business_data/daily_plans")
        if plans_dir.exists():
            plan_files = list(plans_dir.glob(f"{daily_plan.plan_id}.json"))
            return len(plan_files) > 0
        return False
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and compile results."""
        logger.info("Starting comprehensive daily planning workflow tests...")
        
        test_results = {}
        
        try:
            # Run individual tests
            test_results["basic_planning"] = await self.test_basic_daily_planning()
            test_results["parallel_planning"] = await self.test_parallel_planning()
            test_results["agent_chain"] = await self.test_agent_chain_integration()
            test_results["local_data"] = await self.test_local_data_integration()
            test_results["performance"] = await self.test_performance_requirements()
            
            # Calculate overall success
            successful_tests = sum(1 for test in test_results.values() 
                                 if test.get("success", True) and 
                                    test.get("within_time_limit", True))
            
            test_results["summary"] = {
                "total_tests": len(test_results) - 1,  # Exclude summary itself
                "successful_tests": successful_tests,
                "overall_success": successful_tests == len(test_results) - 1,
                "workflow_performance": self.workflow.get_performance_metrics()
            }
            
        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            test_results["error"] = str(e)
        
        return test_results


async def main():
    """Main test execution function."""
    print("=" * 60)
    print("DAILY PLANNING AGENT WORKFLOW TEST SUITE")
    print("=" * 60)
    
    # Initialize tester
    tester = DailyPlanningTester()
    
    # Run all tests
    results = await tester.run_all_tests()
    
    # Print results
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    if "summary" in results:
        summary = results["summary"]
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Successful Tests: {summary['successful_tests']}")
        print(f"Overall Success: {summary['overall_success']}")
        print(f"Workflow Performance: {summary['workflow_performance']}")
    
    print("\nDetailed Results:")
    for test_name, result in results.items():
        if test_name != "summary":
            print(f"\n{test_name.upper()}:")
            if isinstance(result, dict):
                for key, value in result.items():
                    print(f"  {key}: {value}")
            else:
                print(f"  {result}")
    
    # Save results to file
    results_file = Path("test_results_daily_planning.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to: {results_file}")
    
    # Return success status
    return results.get("summary", {}).get("overall_success", False)


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)