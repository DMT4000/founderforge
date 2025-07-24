"""
Tests for VectorStore and ContextRetriever functionality.
Validates FAISS integration, embedding generation, and semantic search capabilities.
"""

import pytest
import tempfile
import shutil
import os
import numpy as np
from datetime import datetime, timedelta
from typing import List

from src.vector_store import VectorStore, ContextRetriever
from src.models import Memory, MemoryType


class TestVectorStore:
    """Test suite for VectorStore functionality."""
    
    @pytest.fixture
    def temp_index_path(self):
        """Create temporary directory for vector index."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def vector_store(self, temp_index_path):
        """Create VectorStore with temporary index path."""
        return VectorStore(
            model_name="all-MiniLM-L6-v2",  # Small model for testing
            index_path=temp_index_path
        )
    
    @pytest.fixture
    def sample_memories(self):
        """Create sample memories for testing."""
        memories = [
            Memory(
                user_id="user_123",
                content="I want to raise Series A funding for my SaaS startup",
                memory_type=MemoryType.LONG_TERM,
                confidence=0.9
            ),
            Memory(
                user_id="user_123",
                content="My startup focuses on AI-powered customer service automation",
                memory_type=MemoryType.LONG_TERM,
                confidence=0.85
            ),
            Memory(
                user_id="user_123",
                content="I need help with product-market fit validation",
                memory_type=MemoryType.SHORT_TERM,
                confidence=0.8
            ),
            Memory(
                user_id="user_456",
                content="Looking for co-founder with technical background",
                memory_type=MemoryType.LONG_TERM,
                confidence=0.9
            ),
            Memory(
                user_id="user_123",
                content="Revenue is growing 20% month over month",
                memory_type=MemoryType.SHORT_TERM,
                confidence=0.95
            )
        ]
        return memories
    
    def test_vector_store_initialization(self, vector_store):
        """Test vector store initialization."""
        assert vector_store.model_name == "all-MiniLM-L6-v2"
        assert vector_store.embedding_dim > 0
        assert vector_store.encoder is not None
        assert vector_store.next_id == 0
        assert len(vector_store.document_metadata) == 0
    
    def test_add_memory_success(self, vector_store, sample_memories):
        """Test adding memories to vector store."""
        memory = sample_memories[0]
        
        result = vector_store.add_memory(memory)
        assert result is True
        
        # Verify memory was added
        assert vector_store.next_id == 1
        assert len(vector_store.document_metadata) == 1
        assert vector_store.index is not None
        assert vector_store.index.ntotal == 1
        
        # Verify metadata
        metadata = vector_store.document_metadata[0]
        assert metadata["memory_id"] == memory.id
        assert metadata["user_id"] == memory.user_id
        assert metadata["content"] == memory.content
    
    def test_add_multiple_memories(self, vector_store, sample_memories):
        """Test adding multiple memories."""
        for memory in sample_memories:
            result = vector_store.add_memory(memory)
            assert result is True
        
        assert vector_store.next_id == len(sample_memories)
        assert len(vector_store.document_metadata) == len(sample_memories)
        assert vector_store.index.ntotal == len(sample_memories)
    
    def test_search_similar_basic(self, vector_store, sample_memories):
        """Test basic semantic search functionality."""
        # Add memories
        for memory in sample_memories:
            vector_store.add_memory(memory)
        
        # Search for funding-related content
        results = vector_store.search_similar("funding investment capital", k=3)
        
        assert len(results) > 0
        assert all("similarity_score" in result for result in results)
        assert all("memory_id" in result for result in results)
        assert all("content" in result for result in results)
        
        # Results should be ordered by similarity (highest first)
        for i in range(len(results) - 1):
            assert results[i]["similarity_score"] >= results[i + 1]["similarity_score"]
    
    def test_search_similar_user_filter(self, vector_store, sample_memories):
        """Test search with user ID filtering."""
        # Add memories
        for memory in sample_memories:
            vector_store.add_memory(memory)
        
        # Search for user_123 only
        results = vector_store.search_similar("startup business", k=10, user_id="user_123")
        
        assert len(results) > 0
        assert all(result["user_id"] == "user_123" for result in results)
        
        # Search for user_456 only
        results_456 = vector_store.search_similar("startup business", k=10, user_id="user_456")
        
        assert len(results_456) > 0
        assert all(result["user_id"] == "user_456" for result in results_456)
        
        # Results should be different
        user_123_ids = {r["memory_id"] for r in results}
        user_456_ids = {r["memory_id"] for r in results_456}
        assert user_123_ids.isdisjoint(user_456_ids)
    
    def test_search_similar_confidence_filter(self, vector_store, sample_memories):
        """Test search with confidence filtering."""
        # Add memories
        for memory in sample_memories:
            vector_store.add_memory(memory)
        
        # Search with high confidence threshold
        results = vector_store.search_similar("startup", k=10, min_confidence=0.9)
        
        assert all(result["confidence"] >= 0.9 for result in results)
        
        # Search with lower confidence threshold should return more results
        results_low = vector_store.search_similar("startup", k=10, min_confidence=0.7)
        
        assert len(results_low) >= len(results)
    
    def test_search_empty_index(self, vector_store):
        """Test search on empty index."""
        results = vector_store.search_similar("test query", k=5)
        assert len(results) == 0
    
    def test_remove_memory(self, vector_store, sample_memories):
        """Test memory removal (soft delete)."""
        memory = sample_memories[0]
        vector_store.add_memory(memory)
        
        # Verify memory exists in search
        results = vector_store.search_similar(memory.content, k=1)
        assert len(results) == 1
        
        # Remove memory
        result = vector_store.remove_memory(memory.id)
        assert result is True
        
        # Verify memory is marked as deleted
        metadata = vector_store.document_metadata[0]
        assert metadata.get("deleted") is True
        
        # Memory should not appear in search results
        results_after = vector_store.search_similar(memory.content, k=1)
        assert len(results_after) == 0
    
    def test_remove_user_memories(self, vector_store, sample_memories):
        """Test bulk removal of user memories."""
        # Add memories
        for memory in sample_memories:
            vector_store.add_memory(memory)
        
        # Count user_123 memories
        user_123_count = sum(1 for m in sample_memories if m.user_id == "user_123")
        
        # Remove all memories for user_123
        removed_count = vector_store.remove_user_memories("user_123")
        assert removed_count == user_123_count
        
        # Verify user_123 memories don't appear in search
        results = vector_store.search_similar("startup", k=10, user_id="user_123")
        assert len(results) == 0
        
        # Verify other users' memories still exist
        results_456 = vector_store.search_similar("startup", k=10, user_id="user_456")
        assert len(results_456) > 0
    
    def test_get_embedding(self, vector_store):
        """Test embedding generation."""
        text = "This is a test sentence for embedding"
        embedding = vector_store.get_embedding(text)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (vector_store.embedding_dim,)
        assert not np.isnan(embedding).any()
        
        # Test that same text produces same embedding
        embedding2 = vector_store.get_embedding(text)
        np.testing.assert_array_equal(embedding, embedding2)
        
        # Test that different text produces different embedding
        embedding3 = vector_store.get_embedding("Different text")
        assert not np.array_equal(embedding, embedding3)
    
    def test_get_stats(self, vector_store, sample_memories):
        """Test statistics generation."""
        # Test empty stats
        stats = vector_store.get_stats()
        assert stats["total_documents"] == 0
        assert stats["active_documents"] == 0
        assert stats["deleted_documents"] == 0
        
        # Add memories
        for memory in sample_memories:
            vector_store.add_memory(memory)
        
        stats = vector_store.get_stats()
        assert stats["total_documents"] == len(sample_memories)
        assert stats["active_documents"] == len(sample_memories)
        assert stats["deleted_documents"] == 0
        assert stats["model_name"] == "all-MiniLM-L6-v2"
        assert stats["embedding_dimension"] > 0
        
        # Remove some memories and check stats
        vector_store.remove_memory(sample_memories[0].id)
        stats_after = vector_store.get_stats()
        assert stats_after["active_documents"] == len(sample_memories) - 1
        assert stats_after["deleted_documents"] == 1
    
    def test_save_and_load_index(self, vector_store, sample_memories, temp_index_path):
        """Test saving and loading index persistence."""
        # Add memories
        for memory in sample_memories:
            vector_store.add_memory(memory)
        
        # Save index
        result = vector_store.save()
        assert result is True
        
        # Verify files exist
        index_file = vector_store.index_path / "faiss.index"
        metadata_file = vector_store.index_path / "metadata.pkl"
        assert index_file.exists()
        assert metadata_file.exists()
        
        # Create new vector store and load
        new_vector_store = VectorStore(
            model_name="all-MiniLM-L6-v2",
            index_path=temp_index_path
        )
        
        # Verify data was loaded
        assert new_vector_store.next_id == len(sample_memories)
        assert len(new_vector_store.document_metadata) == len(sample_memories)
        assert new_vector_store.index.ntotal == len(sample_memories)
        
        # Test search works on loaded index
        results = new_vector_store.search_similar("funding", k=3)
        assert len(results) > 0
    
    def test_rebuild_index(self, vector_store, sample_memories):
        """Test index rebuilding to remove deleted entries."""
        # Add memories
        for memory in sample_memories:
            vector_store.add_memory(memory)
        
        original_count = len(sample_memories)
        
        # Remove some memories
        vector_store.remove_memory(sample_memories[0].id)
        vector_store.remove_memory(sample_memories[1].id)
        
        # Verify soft deletion
        stats = vector_store.get_stats()
        assert stats["total_documents"] == original_count
        assert stats["deleted_documents"] == 2
        assert stats["active_documents"] == original_count - 2
        
        # Rebuild index
        result = vector_store.rebuild_index()
        assert result is True
        
        # Verify deleted entries are removed
        stats_after = vector_store.get_stats()
        assert stats_after["total_documents"] == original_count - 2
        assert stats_after["deleted_documents"] == 0
        assert stats_after["active_documents"] == original_count - 2
        
        # Verify search still works
        results = vector_store.search_similar("startup", k=5)
        assert len(results) == original_count - 2
    
    def test_clear_vector_store(self, vector_store, sample_memories):
        """Test clearing all data from vector store."""
        # Add memories
        for memory in sample_memories:
            vector_store.add_memory(memory)
        
        assert vector_store.next_id > 0
        assert len(vector_store.document_metadata) > 0
        
        # Clear store
        vector_store.clear()
        
        assert vector_store.next_id == 0
        assert len(vector_store.document_metadata) == 0
        assert vector_store.index is None
        
        # Search should return empty results
        results = vector_store.search_similar("test", k=5)
        assert len(results) == 0
    
    def test_expired_memory_filtering(self, vector_store):
        """Test that expired memories are filtered out of search results."""
        # Create expired memory
        past_time = datetime.now() - timedelta(hours=2)
        expired_memory = Memory(
            user_id="user_123",
            content="This memory has expired",
            memory_type=MemoryType.SHORT_TERM,
            confidence=0.8,
            created_at=past_time,
            expires_at=past_time + timedelta(hours=1)  # Expired 1 hour ago
        )
        
        # Create active memory
        active_memory = Memory(
            user_id="user_123",
            content="This memory is still active",
            memory_type=MemoryType.LONG_TERM,
            confidence=0.9
        )
        
        # Add both memories
        vector_store.add_memory(expired_memory)
        vector_store.add_memory(active_memory)
        
        # Search should only return active memory
        results = vector_store.search_similar("memory", k=5)
        
        assert len(results) == 1
        assert results[0]["memory_id"] == active_memory.id
        assert results[0]["content"] == active_memory.content


class TestContextRetriever:
    """Test suite for ContextRetriever functionality."""
    
    @pytest.fixture
    def temp_index_path(self):
        """Create temporary directory for vector index."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def vector_store(self, temp_index_path):
        """Create VectorStore with temporary index path."""
        return VectorStore(index_path=temp_index_path)
    
    @pytest.fixture
    def context_retriever(self, vector_store):
        """Create ContextRetriever with vector store."""
        return ContextRetriever(vector_store)
    
    @pytest.fixture
    def sample_memories(self):
        """Create sample memories for context testing."""
        return [
            Memory(
                user_id="user_123",
                content="I want to raise Series A funding of $5M for my SaaS startup",
                memory_type=MemoryType.LONG_TERM,
                confidence=0.95
            ),
            Memory(
                user_id="user_123",
                content="My startup has 50,000 monthly active users and $100K MRR",
                memory_type=MemoryType.LONG_TERM,
                confidence=0.9
            ),
            Memory(
                user_id="user_123",
                content="I need help with investor pitch deck preparation",
                memory_type=MemoryType.SHORT_TERM,
                confidence=0.85
            ),
            Memory(
                user_id="user_123",
                content="My target market is small to medium businesses",
                memory_type=MemoryType.LONG_TERM,
                confidence=0.8
            )
        ]
    
    def test_retrieve_context_basic(self, context_retriever, vector_store, sample_memories):
        """Test basic context retrieval."""
        # Add memories to vector store
        for memory in sample_memories:
            vector_store.add_memory(memory)
        
        # Retrieve context for funding query
        context = context_retriever.retrieve_context(
            query="How should I prepare for fundraising?",
            user_id="user_123",
            max_results=3
        )
        
        assert len(context) > 0
        assert len(context) <= 3
        assert all(item["user_id"] == "user_123" for item in context)
        assert all("similarity_score" in item for item in context)
        assert all("content" in item for item in context)
    
    def test_retrieve_context_similarity_threshold(self, context_retriever, vector_store, sample_memories):
        """Test context retrieval with similarity threshold."""
        # Add memories
        for memory in sample_memories:
            vector_store.add_memory(memory)
        
        # Test with high similarity threshold
        context_high = context_retriever.retrieve_context(
            query="funding investment",
            user_id="user_123",
            min_similarity=0.8
        )
        
        # Test with low similarity threshold
        context_low = context_retriever.retrieve_context(
            query="funding investment",
            user_id="user_123",
            min_similarity=0.3
        )
        
        # Low threshold should return more or equal results
        assert len(context_low) >= len(context_high)
        
        # All results should meet similarity threshold
        assert all(item["similarity_score"] >= 0.8 for item in context_high)
        assert all(item["similarity_score"] >= 0.3 for item in context_low)
    
    def test_retrieve_context_empty_store(self, context_retriever):
        """Test context retrieval on empty vector store."""
        context = context_retriever.retrieve_context(
            query="test query",
            user_id="user_123"
        )
        
        assert len(context) == 0
    
    def test_get_user_context_summary(self, context_retriever, vector_store, sample_memories):
        """Test user context summary generation."""
        # Add memories
        for memory in sample_memories:
            vector_store.add_memory(memory)
        
        summary = context_retriever.get_user_context_summary("user_123")
        
        assert summary["user_id"] == "user_123"
        assert summary["total_memories"] > 0
        assert "memory_types" in summary
        assert "avg_confidence" in summary
        assert "recent_topics" in summary
        assert "context_quality" in summary
        
        # Check memory types
        assert "LONG_TERM" in summary["memory_types"]
        assert "SHORT_TERM" in summary["memory_types"]
        
        # Check context quality assessment
        assert summary["context_quality"] in ["high", "medium", "low"]
    
    def test_get_user_context_summary_empty(self, context_retriever):
        """Test context summary for user with no memories."""
        summary = context_retriever.get_user_context_summary("nonexistent_user")
        
        assert summary["user_id"] == "nonexistent_user"
        assert "No context available" in summary["summary"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])