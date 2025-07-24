#!/usr/bin/env python3
"""
Simple integration test to verify all components work together.
"""

import asyncio
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("Starting simple integration test...")

try:
    # Test imports
    print("1. Testing imports...")
    import sys
    import os
    
    # Change to src directory for imports
    os.chdir('src')
    sys.path.insert(0, '.')
    
    from database import initialize_database, get_db_manager
    from memory_repository import get_memory_repository
    from context_manager import ContextAssembler
    from agents import AgentOrchestrator
    from gemini_client import GeminiClient, MockMode
    from confidence_manager import ConfidenceManager
    from models import Memory, MemoryType
    
    # Change back to root directory
    os.chdir('..')
    
    print("✅ All imports successful")
    
    # Test database initialization
    print("2. Testing database initialization...")
    if initialize_database():
        print("✅ Database initialized")
    else:
        print("❌ Database initialization failed")
        sys.exit(1)
    
    # Test component initialization
    print("3. Testing component initialization...")
    db_manager = get_db_manager()
    memory_repository = get_memory_repository()
    context_manager = ContextAssembler()
    gemini_client = GeminiClient(mock_mode=MockMode.SUCCESS)
    confidence_manager = ConfidenceManager()
    
    agent_orchestrator = AgentOrchestrator(
        gemini_client=gemini_client,
        context_manager=context_manager,
        confidence_manager=confidence_manager
    )
    print("✅ All components initialized")
    
    # Test memory operations
    print("4. Testing memory operations...")
    test_user_id = "integration_test_user"
    
    # Create memory
    test_memory = Memory(
        user_id=test_user_id,
        content="Integration test memory",
        memory_type=MemoryType.SHORT_TERM,
        confidence=0.9
    )
    
    memory_id = memory_repository.create_memory(test_memory, confirm=False)
    if memory_id:
        print("✅ Memory creation successful")
    else:
        print("❌ Memory creation failed")
    
    # Retrieve memory
    start_time = time.time()
    memories = memory_repository.get_memories_by_user(test_user_id)
    retrieval_time = (time.time() - start_time) * 1000
    
    if len(memories) > 0 and retrieval_time < 10.0:
        print(f"✅ Memory retrieval successful ({retrieval_time:.2f}ms)")
    else:
        print(f"❌ Memory retrieval issue (time: {retrieval_time:.2f}ms, count: {len(memories)})")
    
    # Test context assembly
    print("5. Testing context assembly...")
    context = context_manager.assemble_context(test_user_id, "Test query")
    if context and context.token_count < 16000:
        print(f"✅ Context assembly successful ({context.token_count} tokens)")
    else:
        print(f"❌ Context assembly issue (tokens: {context.token_count if context else 'None'})")
    
    # Test agent workflow
    print("6. Testing agent workflow...")
    
    async def test_workflow():
        try:
            start_time = time.time()
            result = await agent_orchestrator.execute_workflow(
                workflow_type="general",
                user_id=test_user_id,
                task_data={"user_query": "Test query", "context": context.to_dict() if context else {}}
            )
            workflow_time = time.time() - start_time
            
            if result.success and workflow_time < 30.0:
                print(f"✅ Agent workflow successful ({workflow_time:.2f}s)")
                return True
            else:
                print(f"❌ Agent workflow issue (success: {result.success}, time: {workflow_time:.2f}s)")
                return False
        except Exception as e:
            print(f"❌ Agent workflow failed: {e}")
            return False
    
    workflow_success = asyncio.run(test_workflow())
    
    # Cleanup
    print("7. Cleaning up...")
    memory_repository.delete_user_memories(test_user_id, confirm=False)
    db_manager.execute_update("DELETE FROM users WHERE id = ?", (test_user_id,))
    print("✅ Cleanup completed")
    
    # Final result
    print("\n" + "="*50)
    if workflow_success:
        print("🎉 INTEGRATION TEST PASSED!")
        print("All core components are working together successfully.")
        sys.exit(0)
    else:
        print("❌ INTEGRATION TEST FAILED!")
        print("Some components are not working properly.")
        sys.exit(1)

except Exception as e:
    print(f"❌ Integration test failed with error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)