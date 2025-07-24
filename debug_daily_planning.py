#!/usr/bin/env python3
"""
Debug script to test daily planning workflow step by step.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import required modules
try:
    from src.daily_planner import DailyPlanningWorkflow, DailyPlanningInput
    from src.agents import AgentOrchestrator
    from src.gemini_client import GeminiClient, MockMode
    from src.context_manager import ContextAssembler
    from src.confidence_manager import ConfidenceManager
except ImportError as e:
    logger.error(f"Import error: {e}")
    exit(1)


async def debug_orchestrator_directly():
    """Debug the orchestrator agent directly."""
    print("=" * 60)
    print("DEBUG: Testing Orchestrator Agent Directly")
    print("=" * 60)
    
    # Initialize components
    gemini_client = GeminiClient(api_key="debug_key", mock_mode=MockMode.SUCCESS)
    context_manager = ContextAssembler()
    confidence_manager = ConfidenceManager()
    
    orchestrator = AgentOrchestrator(
        gemini_client=gemini_client,
        context_manager=context_manager,
        confidence_manager=confidence_manager
    )
    
    # Test orchestrator agent directly
    orchestrator_agent = orchestrator.agents["orchestrator"]
    
    # Create test state
    test_state = {
        "user_id": "debug_user",
        "user_context": {"goals": ["test"]},
        "current_task": "daily_planning",
        "task_data": {"test": "data"},
        "agent_outputs": {},
        "workflow_result": {},
        "confidence_scores": {},
        "execution_logs": [],
        "error_messages": [],
        "next_agent": None,
        "completed": False
    }
    
    print("1. Testing orchestrator agent execution...")
    print(f"   - Input state keys: {list(test_state.keys())}")
    print(f"   - Current task: {test_state['current_task']}")
    
    # Execute orchestrator
    try:
        result = await orchestrator_agent.execute(test_state)
        
        print("2. Orchestrator execution result:")
        print(f"   - Result keys: {list(result.keys())}")
        
        for key, value in result.items():
            print(f"   - {key}: {type(value).__name__}")
            if isinstance(value, dict):
                print(f"     Dict keys: {list(value.keys())}")
            elif isinstance(value, list):
                print(f"     List length: {len(value)}")
            else:
                print(f"     Value: {value}")
        
        # Check next_agent specifically
        next_agent = result.get("next_agent")
        print(f"   - Next agent from result: {next_agent}")
        
        # Test routing logic
        test_state.update(result)
        test_state["next_agent"] = next_agent
        
        print("3. Testing routing logic...")
        route_result = orchestrator._route_from_orchestrator(test_state)
        print(f"   - Route result: {route_result}")
        
    except Exception as e:
        print(f"ERROR: Orchestrator execution failed: {e}")
        import traceback
        traceback.print_exc()


async def debug_workflow():
    """Debug the daily planning workflow step by step."""
    print("\n" + "=" * 60)
    print("DEBUG: Full Workflow Test")
    print("=" * 60)
    
    # Initialize components with mock mode
    print("1. Initializing components...")
    gemini_client = GeminiClient(api_key="debug_key", mock_mode=MockMode.SUCCESS)
    context_manager = ContextAssembler()
    confidence_manager = ConfidenceManager()
    
    # Initialize orchestrator
    print("2. Initializing orchestrator...")
    orchestrator = AgentOrchestrator(
        gemini_client=gemini_client,
        context_manager=context_manager,
        confidence_manager=confidence_manager
    )
    
    # Test orchestrator execution directly
    print("3. Testing orchestrator workflow execution...")
    
    planning_input = DailyPlanningInput(
        user_id="debug_user",
        current_priorities=["Test task 1", "Test task 2"],
        available_time="4 hours",
        energy_level="high"
    )
    
    user_context = {
        "business_info": {
            "stage": "early_stage",
            "industry": "tech",
            "team_size": 5
        },
        "goals": ["Launch product", "Raise funding"]
    }
    
    task_data = {
        "planning_input": planning_input.to_dict(),
        "user_context": user_context
    }
    
    print("4. Executing workflow...")
    print(f"   - Workflow type: daily_planning")
    print(f"   - User ID: {planning_input.user_id}")
    print(f"   - Task data keys: {list(task_data.keys())}")
    
    try:
        workflow_result = await orchestrator.execute_workflow(
            workflow_type="daily_planning",
            user_id=planning_input.user_id,
            task_data=task_data,
            user_context=user_context
        )
        
        print("5. Workflow execution completed!")
        print(f"   - Success: {workflow_result.success}")
        print(f"   - Execution time: {workflow_result.execution_time:.3f}s")
        print(f"   - Confidence score: {workflow_result.confidence_score}")
        print(f"   - Agent logs count: {len(workflow_result.agent_logs)}")
        
        # Show agent outputs
        agent_outputs = workflow_result.result_data.get("agent_outputs", {})
        print(f"   - Agent outputs: {list(agent_outputs.keys())}")
        
        for agent_name, output in agent_outputs.items():
            print(f"     * {agent_name}: {type(output).__name__}")
            if isinstance(output, dict):
                print(f"       Keys: {list(output.keys())}")
        
        # Show agent execution logs
        print("6. Agent execution details:")
        for i, log in enumerate(workflow_result.agent_logs, 1):
            print(f"   {i}. {log.agent_name}: {log.action}")
            print(f"      Success: {log.success}, Time: {log.execution_time:.3f}s")
            if log.error_message:
                print(f"      Error: {log.error_message}")
        
        # Check if this is the expected chain for daily planning
        expected_agents = ["orchestrator", "planner", "tool_caller", "coach"]
        executed_agents = [log.agent_name for log in workflow_result.agent_logs]
        
        print("7. Agent chain analysis:")
        print(f"   - Expected: {expected_agents}")
        print(f"   - Executed: {executed_agents}")
        
        missing_agents = set(expected_agents) - set(executed_agents)
        if missing_agents:
            print(f"   - Missing: {list(missing_agents)}")
        else:
            print("   - ✓ All expected agents executed")
            
    except Exception as e:
        print(f"ERROR: Workflow execution failed: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Main debug function."""
    await debug_orchestrator_directly()
    await debug_workflow()
    print("\nDebug completed!")


if __name__ == "__main__":
    asyncio.run(main()) 