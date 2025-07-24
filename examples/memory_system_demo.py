#!/usr/bin/env python3
"""
Demo script showing the integrated memory system with SQLite and FAISS.
Demonstrates CRUD operations, semantic search, and performance tracking.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from datetime import datetime, timedelta
from src.database import get_db_manager, initialize_database
from src.memory_repository import get_memory_repository
from src.vector_store import get_vector_store, get_context_retriever
from src.models import Memory, MemoryType


def main():
    """Demonstrate the integrated memory system."""
    print("🧠 FounderForge Memory System Demo")
    print("=" * 50)
    
    # Initialize database
    print("\n1. Initializing database...")
    if initialize_database():
        print("✅ Database initialized successfully")
    else:
        print("❌ Database initialization failed")
        return
    
    # Get repository and vector store instances
    memory_repo = get_memory_repository()
    vector_store = get_vector_store()
    context_retriever = get_context_retriever()
    
    # Create test user
    db_manager = get_db_manager()
    db_manager.execute_update(
        "INSERT OR IGNORE INTO users (id, name, email) VALUES (?, ?, ?)",
        ("demo_user", "Demo User", "demo@founderforge.ai")
    )
    
    print("\n2. Creating sample memories...")
    
    # Sample memories for a startup founder
    sample_memories = [
        Memory(
            user_id="demo_user",
            content="I want to raise $2M Series A funding for my AI-powered customer service SaaS platform",
            memory_type=MemoryType.LONG_TERM,
            confidence=0.95
        ),
        Memory(
            user_id="demo_user", 
            content="Current MRR is $50K with 200 enterprise customers, growing 15% month-over-month",
            memory_type=MemoryType.LONG_TERM,
            confidence=0.9
        ),
        Memory(
            user_id="demo_user",
            content="Need help preparing investor pitch deck and financial projections",
            memory_type=MemoryType.SHORT_TERM,
            confidence=0.85
        ),
        Memory(
            user_id="demo_user",
            content="Target market: Mid-market companies (100-1000 employees) in North America",
            memory_type=MemoryType.LONG_TERM,
            confidence=0.8
        ),
        Memory(
            user_id="demo_user",
            content="Key competitors: Zendesk, Intercom, Freshworks - our AI differentiation is 40% faster resolution",
            memory_type=MemoryType.LONG_TERM,
            confidence=0.9
        ),
        Memory(
            user_id="demo_user",
            content="Team: 12 people (6 engineers, 3 sales, 2 marketing, 1 operations)",
            memory_type=MemoryType.LONG_TERM,
            confidence=0.95
        )
    ]
    
    # Store memories in both SQLite and FAISS
    stored_count = 0
    for memory in sample_memories:
        # Store in SQLite
        if memory_repo.create_memory(memory, confirm=False):
            # Store in vector store for semantic search
            if vector_store.add_memory(memory):
                stored_count += 1
                print(f"✅ Stored: {memory.content[:60]}...")
            else:
                print(f"❌ Failed to store in vector store: {memory.content[:60]}...")
        else:
            print(f"❌ Failed to store in database: {memory.content[:60]}...")
    
    print(f"\n📊 Successfully stored {stored_count} memories")
    
    # Demonstrate memory retrieval performance
    print("\n3. Testing memory retrieval performance...")
    
    import time
    start_time = time.perf_counter()
    memories = memory_repo.get_memories_by_user("demo_user")
    retrieval_time = time.perf_counter() - start_time
    
    print(f"✅ Retrieved {len(memories)} memories in {retrieval_time*1000:.2f}ms")
    
    if retrieval_time < 0.010:  # Sub-10ms requirement
        print("🎯 Performance target met: < 10ms")
    else:
        print("⚠️  Performance target missed: > 10ms")
    
    # Demonstrate semantic search
    print("\n4. Testing semantic search...")
    
    search_queries = [
        "How much funding should I raise?",
        "What are my revenue metrics?",
        "Who are my competitors?",
        "Tell me about my team structure"
    ]
    
    for query in search_queries:
        print(f"\n🔍 Query: '{query}'")
        
        # Search using vector store
        results = vector_store.search_similar(query, k=2, user_id="demo_user")
        
        for i, result in enumerate(results, 1):
            similarity = result["similarity_score"]
            content = result["content"]
            print(f"  {i}. [{similarity:.3f}] {content[:80]}...")
    
    # Demonstrate context retrieval
    print("\n5. Testing context retrieval...")
    
    context = context_retriever.retrieve_context(
        query="I need help with fundraising strategy",
        user_id="demo_user",
        max_results=3
    )
    
    print(f"📋 Retrieved {len(context)} relevant context items:")
    for i, item in enumerate(context, 1):
        print(f"  {i}. [{item['similarity_score']:.3f}] {item['content'][:80]}...")
    
    # Get user context summary
    print("\n6. Generating user context summary...")
    
    summary = context_retriever.get_user_context_summary("demo_user")
    print(f"👤 User: {summary['user_id']}")
    print(f"📊 Total memories: {summary['total_memories']}")
    print(f"🎯 Average confidence: {summary['avg_confidence']}")
    print(f"📈 Context quality: {summary['context_quality']}")
    print(f"🏷️  Memory types: {summary['memory_types']}")
    
    # Display performance statistics
    print("\n7. Performance statistics...")
    
    # Memory repository stats
    repo_stats = memory_repo.get_memory_stats("demo_user")
    print(f"💾 Database performance: {repo_stats['avg_retrieval_time_ms']:.2f}ms average")
    
    # Vector store stats
    vector_stats = vector_store.get_stats()
    print(f"🔍 Vector search performance: {vector_stats['avg_search_time_ms']:.2f}ms average")
    print(f"🧮 Embedding performance: {vector_stats['avg_embedding_time_ms']:.2f}ms average")
    print(f"📚 Total documents indexed: {vector_stats['active_documents']}")
    
    # Demonstrate memory deletion with confirmation
    print("\n8. Testing memory deletion...")
    
    if memories:
        memory_to_delete = memories[0]
        print(f"🗑️  Deleting memory: {memory_to_delete.content[:60]}...")
        
        # Delete from both stores
        if memory_repo.delete_memory(memory_to_delete.id, "demo_user", confirm=False):
            vector_store.remove_memory(memory_to_delete.id)
            print("✅ Memory deleted successfully")
        else:
            print("❌ Failed to delete memory")
    
    # Final verification
    print("\n9. Final verification...")
    
    remaining_memories = memory_repo.get_memories_by_user("demo_user")
    remaining_vector = vector_store.search_similar("startup", k=10, user_id="demo_user")
    
    print(f"📊 Remaining memories in database: {len(remaining_memories)}")
    print(f"🔍 Remaining memories in vector store: {len(remaining_vector)}")
    
    print("\n🎉 Memory system demo completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()