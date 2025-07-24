"""
Tests for MemoryRepository with performance validation and CRUD operations.
Validates sub-10ms retrieval performance and confirmation functionality.
"""

import pytest
import tempfile
import os
import time
from datetime import datetime, timedelta
from typing import List

from src.database import DatabaseManager
from src.memory_repository import MemoryRepository
from src.models import Memory, MemoryType


class TestMemoryRepository:
    """Test suite for MemoryRepository functionality."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_file.close()
        
        db_manager = DatabaseManager(temp_file.name)
        db_manager.initialize_schema()
        
        # Create test user to satisfy foreign key constraints
        db_manager.execute_update(
            "INSERT INTO users (id, name, email) VALUES (?, ?, ?)",
            ("test_user_123", "Test User", "test@example.com")
        )
        
        yield db_manager
        
        # Cleanup - close connection first
        db_manager.close()
        try:
            os.unlink(temp_file.name)
        except PermissionError:
            # On Windows, sometimes the file is still locked
            pass
    
    @pytest.fixture
    def memory_repo(self, temp_db):
        """Create MemoryRepository with temporary database."""
        return MemoryRepository(temp_db)
    
    @pytest.fixture
    def sample_memory(self):
        """Create sample memory for testing."""
        return Memory(
            user_id="test_user_123",
            content="User wants to raise Series A funding for their SaaS startup",
            memory_type=MemoryType.LONG_TERM,
            confidence=0.9,
            expires_at=datetime.now() + timedelta(days=30)
        )
    
    @pytest.fixture
    def sample_memories(self):
        """Create multiple sample memories for testing."""
        memories = []
        base_time = datetime.now()
        
        for i in range(10):
            memory = Memory(
                user_id="test_user_123",
                content=f"Memory content {i}: Business goal number {i}",
                memory_type=MemoryType.SHORT_TERM if i % 2 == 0 else MemoryType.LONG_TERM,
                confidence=0.8 + (i * 0.02),  # Varying confidence
                created_at=base_time + timedelta(minutes=i)
            )
            memories.append(memory)
        
        return memories
    
    def test_create_memory_success(self, memory_repo, sample_memory):
        """Test successful memory creation."""
        # Test without confirmation prompt
        result = memory_repo.create_memory(sample_memory, confirm=False)
        
        assert result is True
        
        # Verify memory was stored
        retrieved = memory_repo.get_memory_by_id(sample_memory.id)
        assert retrieved is not None
        assert retrieved.user_id == sample_memory.user_id
        assert retrieved.content == sample_memory.content
        assert retrieved.memory_type == sample_memory.memory_type
        assert retrieved.confidence == sample_memory.confidence
    
    def test_create_memory_with_confirmation(self, memory_repo, sample_memory):
        """Test memory creation with confirmation prompt."""
        # Test with confirmation (should succeed in test environment)
        result = memory_repo.create_memory(sample_memory, confirm=True)
        assert result is True
    
    def test_get_memory_by_id_performance(self, memory_repo, sample_memory):
        """Test memory retrieval performance meets sub-10ms target."""
        # Store memory first
        memory_repo.create_memory(sample_memory, confirm=False)
        
        # Measure retrieval performance
        start_time = time.perf_counter()
        retrieved = memory_repo.get_memory_by_id(sample_memory.id)
        execution_time = time.perf_counter() - start_time
        
        assert retrieved is not None
        assert execution_time < 0.010  # Sub-10ms requirement
        
        # Check performance tracking
        avg_performance = memory_repo.get_avg_performance()
        assert avg_performance > 0
        assert avg_performance < 0.010
    
    def test_get_memories_by_user_performance(self, memory_repo, sample_memories):
        """Test bulk memory retrieval performance."""
        # Store multiple memories
        for memory in sample_memories:
            memory_repo.create_memory(memory, confirm=False)
        
        # Measure bulk retrieval performance
        start_time = time.perf_counter()
        retrieved = memory_repo.get_memories_by_user("test_user_123", limit=50)
        execution_time = time.perf_counter() - start_time
        
        assert len(retrieved) == len(sample_memories)
        assert execution_time < 0.010  # Sub-10ms requirement for bulk retrieval
        
        # Verify ordering (newest first)
        for i in range(len(retrieved) - 1):
            assert retrieved[i].created_at >= retrieved[i + 1].created_at
    
    def test_get_memories_by_type_filter(self, memory_repo, sample_memories):
        """Test memory retrieval with type filtering."""
        # Store memories
        for memory in sample_memories:
            memory_repo.create_memory(memory, confirm=False)
        
        # Test SHORT_TERM filter
        short_term = memory_repo.get_memories_by_user(
            "test_user_123", 
            memory_type=MemoryType.SHORT_TERM
        )
        
        # Test LONG_TERM filter
        long_term = memory_repo.get_memories_by_user(
            "test_user_123", 
            memory_type=MemoryType.LONG_TERM
        )
        
        # Verify filtering
        assert all(m.memory_type == MemoryType.SHORT_TERM for m in short_term)
        assert all(m.memory_type == MemoryType.LONG_TERM for m in long_term)
        assert len(short_term) + len(long_term) == len(sample_memories)
    
    def test_search_memories_functionality(self, memory_repo, sample_memories):
        """Test memory search functionality."""
        # Store memories
        for memory in sample_memories:
            memory_repo.create_memory(memory, confirm=False)
        
        # Search for specific content
        results = memory_repo.search_memories("test_user_123", "Business goal")
        
        assert len(results) > 0
        assert all("Business goal" in memory.content for memory in results)
        
        # Verify results are ordered by confidence (descending)
        for i in range(len(results) - 1):
            assert results[i].confidence >= results[i + 1].confidence
    
    def test_update_memory_success(self, memory_repo, sample_memory):
        """Test memory update functionality."""
        # Store original memory
        memory_repo.create_memory(sample_memory, confirm=False)
        
        # Update memory content
        sample_memory.content = "Updated: User wants to raise Series B funding"
        sample_memory.confidence = 0.95
        
        result = memory_repo.update_memory(sample_memory, confirm=False)
        assert result is True
        
        # Verify update
        retrieved = memory_repo.get_memory_by_id(sample_memory.id)
        assert retrieved.content == "Updated: User wants to raise Series B funding"
        assert retrieved.confidence == 0.95
    
    def test_delete_memory_success(self, memory_repo, sample_memory):
        """Test memory deletion with confirmation."""
        # Store memory
        memory_repo.create_memory(sample_memory, confirm=False)
        
        # Verify it exists
        assert memory_repo.get_memory_by_id(sample_memory.id) is not None
        
        # Delete memory
        result = memory_repo.delete_memory(sample_memory.id, sample_memory.user_id, confirm=False)
        assert result is True
        
        # Verify deletion
        assert memory_repo.get_memory_by_id(sample_memory.id) is None
    
    def test_delete_memory_security(self, memory_repo, sample_memory):
        """Test memory deletion security (wrong user)."""
        # Store memory
        memory_repo.create_memory(sample_memory, confirm=False)
        
        # Try to delete with wrong user_id
        result = memory_repo.delete_memory(sample_memory.id, "wrong_user", confirm=False)
        assert result is False
        
        # Verify memory still exists
        assert memory_repo.get_memory_by_id(sample_memory.id) is not None
    
    def test_delete_user_memories_bulk(self, memory_repo, sample_memories):
        """Test bulk deletion of user memories."""
        # Store memories
        for memory in sample_memories:
            memory_repo.create_memory(memory, confirm=False)
        
        # Delete all memories for user
        deleted_count = memory_repo.delete_user_memories("test_user_123", confirm=False)
        
        assert deleted_count == len(sample_memories)
        
        # Verify all memories are deleted
        remaining = memory_repo.get_memories_by_user("test_user_123")
        assert len(remaining) == 0
    
    def test_delete_user_memories_by_type(self, memory_repo, sample_memories):
        """Test bulk deletion with type filter."""
        # Store memories
        for memory in sample_memories:
            memory_repo.create_memory(memory, confirm=False)
        
        # Count SHORT_TERM memories
        short_term_count = len([m for m in sample_memories if m.memory_type == MemoryType.SHORT_TERM])
        
        # Delete only SHORT_TERM memories
        deleted_count = memory_repo.delete_user_memories(
            "test_user_123", 
            memory_type=MemoryType.SHORT_TERM, 
            confirm=False
        )
        
        assert deleted_count == short_term_count
        
        # Verify only LONG_TERM memories remain
        remaining = memory_repo.get_memories_by_user("test_user_123")
        assert all(m.memory_type == MemoryType.LONG_TERM for m in remaining)
    
    def test_cleanup_expired_memories(self, memory_repo):
        """Test cleanup of expired memories."""
        # Create expired memory with proper timing
        past_time = datetime.now() - timedelta(hours=2)
        expired_memory = Memory(
            user_id="test_user_123",
            content="This memory has expired",
            memory_type=MemoryType.SHORT_TERM,
            confidence=0.8,
            created_at=past_time,
            expires_at=past_time + timedelta(minutes=30)  # Expired 1.5 hours ago
        )
        
        # Create non-expired memory
        active_memory = Memory(
            user_id="test_user_123",
            content="This memory is still active",
            memory_type=MemoryType.LONG_TERM,
            confidence=0.9,
            expires_at=datetime.now() + timedelta(days=1)  # Expires tomorrow
        )
        
        # Store both memories
        memory_repo.create_memory(expired_memory, confirm=False)
        memory_repo.create_memory(active_memory, confirm=False)
        
        # Run cleanup
        cleaned_count = memory_repo.cleanup_expired_memories()
        
        assert cleaned_count == 1
        
        # Verify expired memory is gone, active memory remains
        assert memory_repo.get_memory_by_id(expired_memory.id) is None
        assert memory_repo.get_memory_by_id(active_memory.id) is not None
    
    def test_get_memory_stats(self, memory_repo, sample_memories):
        """Test memory statistics generation."""
        # Store memories
        for memory in sample_memories:
            memory_repo.create_memory(memory, confirm=False)
        
        stats = memory_repo.get_memory_stats("test_user_123")
        
        assert stats["user_id"] == "test_user_123"
        assert stats["total_memories"] == len(sample_memories)
        assert "by_type" in stats
        assert "SHORT_TERM" in stats["by_type"]
        assert "LONG_TERM" in stats["by_type"]
        assert "avg_retrieval_time_ms" in stats
        assert stats["performance_threshold_ms"] == 10.0  # 10ms threshold
    
    def test_performance_tracking(self, memory_repo, sample_memory):
        """Test performance tracking functionality."""
        # Store memory
        memory_repo.create_memory(sample_memory, confirm=False)
        
        # Perform several operations to generate performance data
        for _ in range(5):
            memory_repo.get_memory_by_id(sample_memory.id)
        
        # Check performance metrics
        avg_performance = memory_repo.get_avg_performance()
        assert avg_performance > 0
        assert avg_performance < 0.010  # Should meet performance target
        
        # Test reset functionality
        memory_repo.reset_performance_tracking()
        assert memory_repo.get_avg_performance() == 0.0
    
    def test_expired_memory_detection(self, memory_repo):
        """Test detection of expired memories during retrieval."""
        # Create expired memory with proper timing
        past_time = datetime.now() - timedelta(hours=1)
        expired_memory = Memory(
            user_id="test_user_123",
            content="This memory has expired",
            memory_type=MemoryType.SHORT_TERM,
            confidence=0.8,
            created_at=past_time,
            expires_at=past_time + timedelta(minutes=30)  # Expired 30 minutes ago
        )
        
        memory_repo.create_memory(expired_memory, confirm=False)
        
        # Retrieve expired memory (should still return it but log warning)
        retrieved = memory_repo.get_memory_by_id(expired_memory.id)
        assert retrieved is not None
        assert retrieved.is_expired() is True
        
        # Test filtering in user queries
        memories = memory_repo.get_memories_by_user("test_user_123", include_expired=False)
        assert len(memories) == 0  # Should exclude expired memory
        
        memories_with_expired = memory_repo.get_memories_by_user("test_user_123", include_expired=True)
        assert len(memories_with_expired) == 1  # Should include expired memory
    
    def test_memory_repository_error_handling(self, memory_repo):
        """Test error handling in memory repository operations."""
        # Test retrieval of non-existent memory
        result = memory_repo.get_memory_by_id("non_existent_id")
        assert result is None
        
        # Test deletion of non-existent memory
        result = memory_repo.delete_memory("non_existent_id", "test_user", confirm=False)
        assert result is False
        
        # Test search with empty user
        results = memory_repo.search_memories("", "test")
        assert len(results) == 0
    
    def test_concurrent_access_simulation(self, memory_repo, sample_memories):
        """Test repository behavior under simulated concurrent access."""
        # Store memories
        for memory in sample_memories:
            memory_repo.create_memory(memory, confirm=False)
        
        # Simulate concurrent reads
        results = []
        for _ in range(10):
            memories = memory_repo.get_memories_by_user("test_user_123")
            results.append(len(memories))
        
        # All reads should return consistent results
        assert all(count == len(sample_memories) for count in results)
        
        # Performance should remain consistent
        avg_performance = memory_repo.get_avg_performance()
        assert avg_performance < 0.010


if __name__ == "__main__":
    pytest.main([__file__, "-v"])