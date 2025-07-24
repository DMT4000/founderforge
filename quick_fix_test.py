#!/usr/bin/env python3
"""
Quick test to verify daily planning workflow is working properly.
"""

import asyncio
import logging
from src.daily_planner import DailyPlanningWorkflow, DailyPlanningInput
from src.agents import AgentOrchestrator
from src.gemini_client import GeminiClient, MockMode
from src.context_manager import ContextAssembler
from src.confidence_manager import ConfidenceManager

# Setup logging
logging.basicConfig(level=logging.INFO)

async def test_daily_planning():
    """Test daily planning workflow execution."""
    
    # Initialize components
    gemini_client = GeminiClient(api_key="test_key", mock_mode=MockMode.SUCCESS)
    context_manager = ContextAssembler()
    confidence_manager = ConfidenceManager()
    orchestrator = AgentOrchestrator(gemini_client, context_manager, confidence_manager)
    
    # Create workflow
    workflow = DailyPlanningWorkflow(orchestrator)
    
    # Create planning input
    planning_input = DailyPlanningInput(
        user_id="test_user_quick",
        current_priorities=[
            "Review business metrics",
            "Prepare for investor meeting",
            "Team planning session"
        ],
        available_time="6 hours",
        energy_level="high",
        focus_areas=["business", "team"]
    )
    
    user_context = {
        "business_info": {
            "stage": "early_stage", 
            "industry": "tech",
            "team_size": 8
        },
        "goals": ["Launch product", "Raise Series A"]
    }
    
    print("Testing Daily Planning Workflow...")
    print(f"User: {planning_input.user_id}")
    print(f"Priorities: {planning_input.current_priorities}")
    
    try:
        # Execute daily planning
        daily_plan, workflow_result = await workflow.generate_daily_plan(
            planning_input, user_context
        )
        
        print(f"\n✅ Workflow completed successfully!")
        print(f"   Success: {workflow_result.success}")
        print(f"   Execution time: {workflow_result.execution_time:.3f}s")
        print(f"   Confidence: {workflow_result.confidence_score:.2f}")
        
        # Check agent execution
        print(f"\n📋 Daily Plan Generated:")
        print(f"   Action items: {len(daily_plan.action_items)}")
        print(f"   Time blocks: {len(daily_plan.time_blocks)}")
        print(f"   Has coaching message: {len(daily_plan.motivational_message) > 0}")
        
        # Show action items
        if daily_plan.action_items:
            print(f"\n📝 Action Items:")
            for i, item in enumerate(daily_plan.action_items[:3], 1):
                print(f"   {i}. {item.title} ({item.priority} priority)")
        
        # Check agent execution
        agent_outputs = workflow_result.result_data.get("agent_outputs", {})
        print(f"\n🤖 Agent Execution:")
        print(f"   Agents executed: {list(agent_outputs.keys())}")
        print(f"   Agent logs: {len(workflow_result.agent_logs)}")
        
        for log in workflow_result.agent_logs:
            print(f"   - {log.agent_name}: {log.action} ({log.execution_time:.3f}s)")
        
        # Performance check
        if workflow_result.execution_time < 60.0:
            print(f"\n⚡ Performance: PASSED (under 1 minute)")
        else:
            print(f"\n⚠️  Performance: FAILED (over 1 minute)")
        
        # Agent chain check
        expected_agents = ["planner", "tool_caller", "coach"]
        executed_agent_names = [log.agent_name for log in workflow_result.agent_logs]
        
        chain_complete = all(agent in executed_agent_names for agent in expected_agents)
        print(f"\n🔗 Agent Chain: {'COMPLETE' if chain_complete else 'INCOMPLETE'}")
        
        if not chain_complete:
            missing = set(expected_agents) - set(executed_agent_names)
            print(f"   Missing agents: {list(missing)}")
        
        return workflow_result.success and chain_complete
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_daily_planning())
    print(f"\n{'✅ TEST PASSED' if success else '❌ TEST FAILED'}")
    exit(0 if success else 1) 