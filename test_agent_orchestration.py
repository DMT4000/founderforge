#!/usr/bin/env python3
"""
Test script for agent orchestration system.
Verifies that the multi-agent system works correctly with LangGraph.
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agents import AgentOrchestrator
from gemini_client import GeminiClient, MockMode
from context_manager import ContextAssembler
from confidence_manager import ConfidenceManager
from models import UserContext, BusinessInfo, UserPreferences

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_agent_orchestration():
    """Test the multi-agent orchestration system."""
    
    print("🚀 Testing Agent Orchestration System")
    print("=" * 50)
    
    try:
        # Initialize components
        print("1. Initializing components...")
        
        # Use mock mode for testing
        gemini_client = GeminiClient(api_key="test", mock_mode=MockMode.SUCCESS)
        context_manager = ContextAssembler()
        confidence_manager = ConfidenceManager()
        
        # Initialize orchestrator
        orchestrator = AgentOrchestrator(
            gemini_client=gemini_client,
            context_manager=context_manager,
            confidence_manager=confidence_manager
        )
        
        print("✅ Components initialized successfully")
        
        # Test health check
        print("\n2. Running health check...")
        health_status = await orchestrator.health_check()
        
        print("Health Status:")
        for component, status in health_status.items():
            status_icon = "✅" if status else "❌"
            print(f"  {status_icon} {component}: {'Healthy' if status else 'Unhealthy'}")
        
        # Test workflow execution
        print("\n3. Testing workflow execution...")
        
        # Create test user context
        user_context = {
            "user_id": "test_user_001",
            "business_info": {
                "company_name": "TestCorp",
                "industry": "Technology",
                "stage": "mvp",
                "team_size": 5
            },
            "goals": [
                "Secure Series A funding",
                "Launch MVP product",
                "Build development team"
            ],
            "preferences": {
                "communication_style": "professional",
                "response_length": "medium"
            }
        }
        
        # Test funding form workflow
        print("\n  Testing funding form workflow...")
        funding_task_data = {
            "company_name": "TestCorp",
            "funding_amount": 1000000,
            "business_plan": "We are building an innovative AI-powered platform that helps entrepreneurs validate their business ideas and connect with potential investors. Our MVP has shown strong user engagement with 1000+ active users and 15% month-over-month growth.",
            "team_experience": "Combined 20+ years in tech and business development",
            "market_size": "TAM of $50B in business intelligence market"
        }
        
        funding_result = await orchestrator.execute_workflow(
            workflow_type="funding_form",
            user_id="test_user_001",
            task_data=funding_task_data,
            user_context=user_context
        )
        
        print(f"  ✅ Funding workflow completed:")
        print(f"    - Success: {funding_result.success}")
        print(f"    - Execution time: {funding_result.execution_time:.2f}s")
        print(f"    - Confidence: {funding_result.confidence_score:.2f}")
        print(f"    - Agents executed: {len(funding_result.agent_logs)}")
        
        # Test daily planning workflow
        print("\n  Testing daily planning workflow...")
        planning_task_data = {
            "current_priorities": [
                "Product development",
                "Team hiring",
                "Investor meetings"
            ],
            "available_time": "8 hours",
            "energy_level": "high",
            "upcoming_deadlines": [
                "MVP demo next week",
                "Investor pitch in 2 weeks"
            ]
        }
        
        planning_result = await orchestrator.execute_workflow(
            workflow_type="daily_planning",
            user_id="test_user_001",
            task_data=planning_task_data,
            user_context=user_context
        )
        
        print(f"  ✅ Planning workflow completed:")
        print(f"    - Success: {planning_result.success}")
        print(f"    - Execution time: {planning_result.execution_time:.2f}s")
        print(f"    - Confidence: {planning_result.confidence_score:.2f}")
        print(f"    - Agents executed: {len(planning_result.agent_logs)}")
        
        # Test agent performance metrics
        print("\n4. Agent performance metrics...")
        metrics = orchestrator.get_agent_performance_metrics()
        
        for agent_name, agent_metrics in metrics.items():
            print(f"  {agent_name}:")
            print(f"    - Executions: {agent_metrics['execution_count']}")
            print(f"    - Success rate: {agent_metrics['success_rate']:.2%}")
            print(f"    - Avg time: {agent_metrics['average_execution_time']:.2f}s")
        
        # Test available workflows
        print("\n5. Available workflows:")
        workflows = orchestrator.get_available_workflows()
        for workflow in workflows:
            print(f"  - {workflow}")
        
        print("\n🎉 All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        logger.exception("Test execution failed")
        return False


async def test_individual_agents():
    """Test individual agent functionality."""
    
    print("\n🔧 Testing Individual Agents")
    print("=" * 30)
    
    try:
        # Initialize components
        gemini_client = GeminiClient(api_key="test", mock_mode=MockMode.SUCCESS)
        context_manager = ContextAssembler()
        confidence_manager = ConfidenceManager()
        
        orchestrator = AgentOrchestrator(
            gemini_client=gemini_client,
            context_manager=context_manager,
            confidence_manager=confidence_manager
        )
        
        # Test state structure
        test_state = {
            "user_id": "test_user",
            "user_context": {"goals": ["test goal"]},
            "current_task": "test_task",
            "task_data": {"test": "data"},
            "agent_outputs": {},
            "workflow_result": {},
            "confidence_scores": {},
            "execution_logs": [],
            "error_messages": [],
            "next_agent": None,
            "completed": False
        }
        
        # Test orchestrator agent
        print("Testing Orchestrator Agent...")
        orchestrator_result = await orchestrator.agents["orchestrator"].execute(test_state)
        print(f"✅ Orchestrator: {orchestrator_result.get('orchestrator_output', {}).get('workflow_type', 'N/A')}")
        
        # Test validator agent
        print("Testing Validator Agent...")
        validator_result = await orchestrator.agents["validator"].execute(test_state)
        print(f"✅ Validator: {validator_result.get('validator_output', {}).get('is_valid', False)}")
        
        # Test planner agent
        print("Testing Planner Agent...")
        planner_result = await orchestrator.agents["planner"].execute(test_state)
        print(f"✅ Planner: {len(planner_result.get('planner_output', {}).get('action_items', []))} action items")
        
        # Test tool caller agent
        print("Testing Tool Caller Agent...")
        tool_caller_result = await orchestrator.agents["tool_caller"].execute(test_state)
        print(f"✅ Tool Caller: {len(tool_caller_result.get('tool_caller_output', {}).get('tool_results', []))} tools executed")
        
        # Test coach agent
        print("Testing Coach Agent...")
        coach_result = await orchestrator.agents["coach"].execute(test_state)
        print(f"✅ Coach: {len(coach_result.get('coach_output', {}).get('message', ''))} chars in response")
        
        print("✅ All individual agents tested successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Individual agent test failed: {str(e)}")
        logger.exception("Individual agent test failed")
        return False


def check_log_files():
    """Check if log files are being created correctly."""
    
    print("\n📝 Checking Log Files")
    print("=" * 20)
    
    log_dirs = [
        "data/agent_logs",
        "data/workflow_logs"
    ]
    
    for log_dir in log_dirs:
        if os.path.exists(log_dir):
            files = os.listdir(log_dir)
            print(f"✅ {log_dir}: {len(files)} files")
            for file in files[:3]:  # Show first 3 files
                print(f"  - {file}")
        else:
            print(f"⚠️  {log_dir}: Directory not found")


async def main():
    """Main test function."""
    
    print("🧪 FounderForge Agent Orchestration Test Suite")
    print("=" * 60)
    
    # Ensure data directories exist
    os.makedirs("data/agent_logs", exist_ok=True)
    os.makedirs("data/workflow_logs", exist_ok=True)
    
    # Run tests
    test_results = []
    
    # Test individual agents
    individual_result = await test_individual_agents()
    test_results.append(("Individual Agents", individual_result))
    
    # Test orchestration
    orchestration_result = await test_agent_orchestration()
    test_results.append(("Agent Orchestration", orchestration_result))
    
    # Check log files
    check_log_files()
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 15)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} {test_name}")
    
    all_passed = all(result for _, result in test_results)
    
    if all_passed:
        print("\n🎉 All tests passed! Agent orchestration system is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the logs for details.")
    
    return all_passed


if __name__ == "__main__":
    asyncio.run(main())