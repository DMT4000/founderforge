"""
Demo script for the multi-agent system using LangGraph.
Shows how to use the agent orchestrator for different workflow types.
"""

import asyncio
import json
import logging
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.agents import AgentOrchestrator
from src.gemini_client import GeminiClient, MockMode
from src.context_manager import ContextAssembler
from src.confidence_manager import ConfidenceManager


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def demo_funding_form_workflow():
    """Demonstrate funding form processing workflow."""
    print("\n" + "="*60)
    print("DEMO: Funding Form Processing Workflow")
    print("="*60)
    
    # Initialize components (using mock mode for demo)
    gemini_client = GeminiClient(api_key="demo_key", mock_mode=MockMode.SUCCESS)
    context_manager = ContextAssembler()
    confidence_manager = ConfidenceManager()
    
    # Initialize orchestrator
    orchestrator = AgentOrchestrator(
        gemini_client=gemini_client,
        context_manager=context_manager,
        confidence_manager=confidence_manager
    )
    
    # Sample funding form data
    task_data = {
        "company_name": "TechStart Inc.",
        "funding_amount": 500000,
        "business_plan": "We are developing an innovative AI-powered platform for small businesses. Our solution helps automate customer service and improve operational efficiency. We have a strong team with 5 years of experience and are seeking Series A funding to scale our operations.",
        "industry": "Technology",
        "stage": "Series A",
        "team_size": 8
    }
    
    user_context = {
        "business_info": {
            "company_name": "TechStart Inc.",
            "industry": "Technology",
            "stage": "Series A",
            "team_size": 8
        },
        "goals": ["Secure Series A funding", "Scale operations", "Expand team"]
    }
    
    print(f"Processing funding form for: {task_data['company_name']}")
    print(f"Funding amount: ${task_data['funding_amount']:,}")
    
    # Execute workflow
    try:
        result = await orchestrator.execute_workflow(
            workflow_type="funding_form",
            user_id="demo_user_1",
            task_data=task_data,
            user_context=user_context
        )
        
        print(f"\nWorkflow completed successfully: {result.success}")
        print(f"Execution time: {result.execution_time:.2f} seconds")
        print(f"Confidence score: {result.confidence_score:.2f}")
        print(f"Agents executed: {len(result.agent_logs)}")
        
        # Show agent execution summary
        print("\nAgent Execution Summary:")
        for log in result.agent_logs:
            status = "✓" if log.success else "✗"
            print(f"  {status} {log.agent_name}: {log.action} ({log.execution_time:.2f}s)")
        
        # Show key results
        if "validator_output" in result.result_data:
            validator_result = result.result_data["validator_output"]
            print(f"\nValidation Result: {'PASSED' if validator_result.get('is_valid') else 'FAILED'}")
            if validator_result.get('errors'):
                print("Validation Errors:")
                for error in validator_result['errors']:
                    print(f"  - {error}")
        
        if "planner_output" in result.result_data:
            planner_result = result.result_data["planner_output"]
            print(f"\nPlan Generated: {planner_result.get('executive_summary', 'N/A')}")
            action_items = planner_result.get('action_items', [])
            if action_items:
                print(f"Action Items ({len(action_items)}):")
                for i, item in enumerate(action_items[:3], 1):  # Show first 3
                    print(f"  {i}. {item.get('title', 'N/A')} (Priority: {item.get('priority', 'N/A')})")
        
    except Exception as e:
        print(f"Workflow failed: {e}")


async def demo_daily_planning_workflow():
    """Demonstrate daily planning workflow."""
    print("\n" + "="*60)
    print("DEMO: Daily Planning Workflow")
    print("="*60)
    
    # Initialize components (using mock mode for demo)
    gemini_client = GeminiClient(api_key="demo_key", mock_mode=MockMode.SUCCESS)
    context_manager = ContextAssembler()
    confidence_manager = ConfidenceManager()
    
    # Initialize orchestrator
    orchestrator = AgentOrchestrator(
        gemini_client=gemini_client,
        context_manager=context_manager,
        confidence_manager=confidence_manager
    )
    
    # Sample daily planning data
    task_data = {
        "date": "2025-01-21",
        "priorities": ["Product development", "Team meetings", "Investor calls"],
        "constraints": ["Limited budget", "Remote team coordination"],
        "recent_activities": ["Completed MVP testing", "Hired new developer", "Met with potential investors"]
    }
    
    user_context = {
        "business_info": {
            "company_name": "StartupCo",
            "industry": "SaaS",
            "stage": "MVP",
            "team_size": 5
        },
        "goals": ["Launch product", "Raise seed funding", "Build user base"],
        "preferences": {
            "communication_style": "professional",
            "response_length": "medium"
        }
    }
    
    print(f"Generating daily plan for: {task_data['date']}")
    print(f"Top priorities: {', '.join(task_data['priorities'])}")
    
    # Execute workflow
    try:
        result = await orchestrator.execute_workflow(
            workflow_type="daily_planning",
            user_id="demo_user_2",
            task_data=task_data,
            user_context=user_context
        )
        
        print(f"\nWorkflow completed successfully: {result.success}")
        print(f"Execution time: {result.execution_time:.2f} seconds")
        print(f"Confidence score: {result.confidence_score:.2f}")
        print(f"Agents executed: {len(result.agent_logs)}")
        
        # Show agent execution summary
        print("\nAgent Execution Summary:")
        for log in result.agent_logs:
            status = "✓" if log.success else "✗"
            print(f"  {status} {log.agent_name}: {log.action} ({log.execution_time:.2f}s)")
        
        # Show planning results
        if "planner_output" in result.result_data:
            planner_result = result.result_data["planner_output"]
            print(f"\nDaily Plan: {planner_result.get('executive_summary', 'N/A')}")
            
            action_items = planner_result.get('action_items', [])
            if action_items:
                print(f"\nToday's Action Items ({len(action_items)}):")
                for i, item in enumerate(action_items, 1):
                    print(f"  {i}. {item.get('title', 'N/A')}")
                    print(f"     Priority: {item.get('priority', 'N/A')}, Timeline: {item.get('timeline', 'N/A')}")
        
        # Show coaching output
        if "coach_output" in result.result_data:
            coach_result = result.result_data["coach_output"]
            print(f"\nCoaching Message:")
            print(f"  {coach_result.get('message', 'N/A')}")
        
    except Exception as e:
        print(f"Workflow failed: {e}")


async def demo_agent_performance_metrics():
    """Demonstrate agent performance metrics."""
    print("\n" + "="*60)
    print("DEMO: Agent Performance Metrics")
    print("="*60)
    
    # Initialize components
    gemini_client = GeminiClient(api_key="demo_key", mock_mode=MockMode.SUCCESS)
    context_manager = ContextAssembler()
    confidence_manager = ConfidenceManager()
    
    orchestrator = AgentOrchestrator(
        gemini_client=gemini_client,
        context_manager=context_manager,
        confidence_manager=confidence_manager
    )
    
    # Run a simple workflow to generate some metrics
    await orchestrator.execute_workflow(
        workflow_type="general",
        user_id="metrics_demo",
        task_data={"test": "data"},
        user_context={"goals": ["test"]}
    )
    
    # Get performance metrics
    metrics = orchestrator.get_agent_performance_metrics()
    
    print("Agent Performance Metrics:")
    print("-" * 40)
    for agent_name, agent_metrics in metrics.items():
        print(f"\n{agent_name.upper()}:")
        print(f"  Executions: {agent_metrics['execution_count']}")
        print(f"  Success Rate: {agent_metrics['success_rate']:.1%}")
        print(f"  Avg Time: {agent_metrics['average_execution_time']:.3f}s")
        print(f"  Total Time: {agent_metrics['total_execution_time']:.3f}s")
    
    # Show available workflows
    workflows = orchestrator.get_available_workflows()
    print(f"\nAvailable Workflows: {', '.join(workflows)}")
    
    # Health check
    health_status = await orchestrator.health_check()
    print(f"\nHealth Status:")
    for component, status in health_status.items():
        status_icon = "✓" if status else "✗"
        print(f"  {status_icon} {component}")


async def main():
    """Run all demos."""
    print("FounderForge Multi-Agent System Demo")
    print("====================================")
    
    try:
        # Run demos
        await demo_funding_form_workflow()
        await demo_daily_planning_workflow()
        await demo_agent_performance_metrics()
        
        print("\n" + "="*60)
        print("Demo completed successfully!")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\nDemo failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())