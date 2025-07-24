"""
FAISS vector store implementation for semantic search and context retrieval.
Uses sentence-transformers for local document embedding and FAISS for efficient similarity search.
"""

import os
import pickle
import logging
from .logging_manager import get_logging_manager, LogLevel, LogCategory
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import time

try:
    import faiss
except ImportError:
    raise ImportError("FAISS not installed. Install with: pip install faiss-cpu")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("sentence-transformers not installed. Install with: pip install sentence-transformers")

from .models import Memory, ValidationError


class VectorStore:
    """FAISS-based vector store for semantic search of memories and documents."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", index_path: str = "data/vector_index"):
        """
        Initialize vector store with sentence transformer model and FAISS index.
        
        Args:
            model_name: Name of sentence-transformers model to use
            index_path: Path to store FAISS index files
        """
        self.model_name = model_name
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        self.logger = get_logging_manager().get_logger(__name__.split(".")[-1], LogCategory.SYSTEM)
        
        # Initialize sentence transformer model
        self.logger.info(f"Loading sentence transformer model: {model_name}")
        self.encoder = SentenceTransformer(model_name)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        
        # Initialize FAISS index
        self.index = None
        self.document_metadata: Dict[int, Dict[str, Any]] = {}
        self.next_id = 0
        
        # Performance tracking
        self._embedding_times: List[float] = []
        self._search_times: List[float] = []
        
        # Load existing index if available
        self._load_index()
    
    def add_memory(self, memory: Memory) -> bool:
        """
        Add a memory to the vector store with embedding.
        
        Args:
            memory: Memory object to add
            
        Returns:
            True if successfully added, False otherwise
        """
        try:
            start_time = time.perf_counter()
            
            # Generate embedding for memory content
            embedding = self._embed_text(memory.content)
            
            # Initialize index if needed
            if self.index is None:
                self._initialize_index()
            
            # Add to FAISS index
            embedding_array = np.array([embedding], dtype=np.float32)
            self.index.add(embedding_array)
            
            # Store metadata
            self.document_metadata[self.next_id] = {
                "memory_id": memory.id,
                "user_id": memory.user_id,
                "content": memory.content,
                "memory_type": memory.memory_type.value,
                "confidence": memory.confidence,
                "created_at": memory.created_at.isoformat(),
                "expires_at": memory.expires_at.isoformat() if memory.expires_at else None
            }
            
            self.next_id += 1
            
            # Save index periodically
            if self.next_id % 10 == 0:
                self._save_index()
            
            embedding_time = time.perf_counter() - start_time
            self._embedding_times.append(embedding_time)
            
            self.logger.debug(f"Added memory {memory.id} to vector store in {embedding_time:.3f}s")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add memory {memory.id} to vector store: {e}")
            return False
    
    def search_similar(self, query: str, k: int = 10, user_id: Optional[str] = None, 
                      min_confidence: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search for similar memories using semantic similarity.
        
        Args:
            query: Search query text
            k: Number of results to return
            user_id: Optional filter by user ID
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of similar memories with similarity scores
        """
        if self.index is None or self.index.ntotal == 0:
            self.logger.warning("Vector store is empty, no results to return")
            return []
        
        start_time = time.perf_counter()
        
        try:
            # Generate query embedding
            query_embedding = self._embed_text(query)
            query_array = np.array([query_embedding], dtype=np.float32)
            
            # Search FAISS index
            search_k = min(k * 2, self.index.ntotal)  # Get more results for filtering
            similarities, indices = self.index.search(query_array, search_k)
            
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx == -1:  # FAISS returns -1 for empty slots
                    continue
                
                metadata = self.document_metadata.get(idx)
                if not metadata:
                    continue
                
                # Skip deleted memories
                if metadata.get("deleted", False):
                    continue
                
                # Apply filters
                if user_id and metadata.get("user_id") != user_id:
                    continue
                
                if metadata.get("confidence", 0.0) < min_confidence:
                    continue
                
                # Check if memory has expired
                if metadata.get("expires_at"):
                    from datetime import datetime
                    expires_at = datetime.fromisoformat(metadata["expires_at"])
                    if datetime.now() > expires_at:
                        continue
                
                result = {
                    "memory_id": metadata["memory_id"],
                    "user_id": metadata["user_id"],
                    "content": metadata["content"],
                    "memory_type": metadata["memory_type"],
                    "confidence": metadata["confidence"],
                    "created_at": metadata["created_at"],
                    "similarity_score": float(similarity),
                    "rank": len(results) + 1
                }
                
                results.append(result)
                
                if len(results) >= k:
                    break
            
            search_time = time.perf_counter() - start_time
            self._search_times.append(search_time)
            
            self.logger.debug(f"Vector search completed in {search_time:.3f}s, found {len(results)} results")
            return results
            
        except Exception as e:
            self.logger.error(f"Vector search failed: {e}")
            return []
    
    def remove_memory(self, memory_id: str) -> bool:
        """
        Remove a memory from the vector store.
        Note: FAISS doesn't support efficient deletion, so we mark as deleted in metadata.
        
        Args:
            memory_id: ID of memory to remove
            
        Returns:
            True if successfully marked for deletion, False otherwise
        """
        try:
            # Find the index of the memory
            for idx, metadata in self.document_metadata.items():
                if metadata.get("memory_id") == memory_id:
                    # Mark as deleted instead of actually removing
                    metadata["deleted"] = True
                    self.logger.debug(f"Marked memory {memory_id} as deleted in vector store")
                    return True
            
            self.logger.warning(f"Memory {memory_id} not found in vector store")
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to remove memory {memory_id} from vector store: {e}")
            return False
    
    def remove_user_memories(self, user_id: str) -> int:
        """
        Remove all memories for a specific user.
        
        Args:
            user_id: User ID to remove memories for
            
        Returns:
            Number of memories marked for deletion
        """
        try:
            count = 0
            for idx, metadata in self.document_metadata.items():
                if metadata.get("user_id") == user_id and not metadata.get("deleted", False):
                    metadata["deleted"] = True
                    count += 1
            
            self.logger.info(f"Marked {count} memories for user {user_id} as deleted in vector store")
            return count
            
        except Exception as e:
            self.logger.error(f"Failed to remove memories for user {user_id}: {e}")
            return 0
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a text string.
        
        Args:
            text: Text to embed
            
        Returns:
            Numpy array containing the embedding
        """
        return self._embed_text(text)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get vector store statistics.
        
        Returns:
            Dictionary with statistics
        """
        active_count = sum(1 for metadata in self.document_metadata.values() 
                          if not metadata.get("deleted", False))
        deleted_count = sum(1 for metadata in self.document_metadata.values() 
                           if metadata.get("deleted", False))
        
        avg_embedding_time = (sum(self._embedding_times) / len(self._embedding_times) 
                             if self._embedding_times else 0.0)
        avg_search_time = (sum(self._search_times) / len(self._search_times) 
                          if self._search_times else 0.0)
        
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "total_documents": len(self.document_metadata),
            "active_documents": active_count,
            "deleted_documents": deleted_count,
            "index_size": self.index.ntotal if self.index else 0,
            "avg_embedding_time_ms": avg_embedding_time * 1000,
            "avg_search_time_ms": avg_search_time * 1000,
            "total_embeddings": len(self._embedding_times),
            "total_searches": len(self._search_times)
        }
    
    def rebuild_index(self) -> bool:
        """
        Rebuild the FAISS index, removing deleted entries.
        
        Returns:
            True if rebuild successful, False otherwise
        """
        try:
            self.logger.info("Starting vector store index rebuild")
            
            # Collect active memories
            active_memories = []
            active_metadata = {}
            new_id = 0
            
            for metadata in self.document_metadata.values():
                if not metadata.get("deleted", False):
                    active_memories.append(metadata["content"])
                    active_metadata[new_id] = metadata.copy()
                    active_metadata[new_id].pop("deleted", None)  # Remove deleted flag
                    new_id += 1
            
            if not active_memories:
                self.logger.info("No active memories to rebuild index")
                self._initialize_index()
                self.document_metadata = {}
                self.next_id = 0
                return True
            
            # Generate embeddings for all active memories
            self.logger.info(f"Generating embeddings for {len(active_memories)} active memories")
            embeddings = self.encoder.encode(active_memories, show_progress_bar=True)
            
            # Create new index
            self._initialize_index()
            embeddings_array = np.array(embeddings, dtype=np.float32)
            self.index.add(embeddings_array)
            
            # Update metadata
            self.document_metadata = active_metadata
            self.next_id = new_id
            
            # Save rebuilt index
            self._save_index()
            
            self.logger.info(f"Index rebuild completed. Active documents: {len(active_metadata)}")
            return True
            
        except Exception as e:
            self.logger.error(f"Index rebuild failed: {e}")
            return False
    
    def save(self) -> bool:
        """
        Save the current index and metadata to disk.
        
        Returns:
            True if save successful, False otherwise
        """
        return self._save_index()
    
    def clear(self) -> None:
        """Clear all data from the vector store."""
        self.index = None
        self.document_metadata = {}
        self.next_id = 0
        self._embedding_times = []
        self._search_times = []
        self.logger.info("Vector store cleared")
    
    # Private methods
    
    def _embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for text using sentence transformer."""
        if not text or not text.strip():
            raise ValidationError("Text cannot be empty for embedding")
        
        embedding = self.encoder.encode(text, convert_to_numpy=True)
        return embedding
    
    def _initialize_index(self) -> None:
        """Initialize a new FAISS index."""
        # Use IndexFlatIP for cosine similarity (inner product with normalized vectors)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.logger.debug(f"Initialized FAISS index with dimension {self.embedding_dim}")
    
    def _save_index(self) -> bool:
        """Save FAISS index and metadata to disk."""
        try:
            if self.index is None:
                return True
            
            # Save FAISS index
            index_file = self.index_path / "faiss.index"
            faiss.write_index(self.index, str(index_file))
            
            # Save metadata
            metadata_file = self.index_path / "metadata.pkl"
            with open(metadata_file, 'wb') as f:
                pickle.dump({
                    "document_metadata": self.document_metadata,
                    "next_id": self.next_id,
                    "model_name": self.model_name,
                    "embedding_dim": self.embedding_dim
                }, f)
            
            self.logger.debug(f"Saved vector store to {self.index_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save vector store: {e}")
            return False
    
    def _load_index(self) -> bool:
        """Load FAISS index and metadata from disk."""
        try:
            index_file = self.index_path / "faiss.index"
            metadata_file = self.index_path / "metadata.pkl"
            
            if not index_file.exists() or not metadata_file.exists():
                self.logger.info("No existing vector store found, starting fresh")
                return False
            
            # Load FAISS index
            self.index = faiss.read_index(str(index_file))
            
            # Load metadata
            with open(metadata_file, 'rb') as f:
                data = pickle.load(f)
                self.document_metadata = data.get("document_metadata", {})
                self.next_id = data.get("next_id", 0)
                
                # Verify model compatibility
                saved_model = data.get("model_name")
                saved_dim = data.get("embedding_dim")
                
                if saved_model != self.model_name:
                    self.logger.warning(f"Model mismatch: saved={saved_model}, current={self.model_name}")
                    return False
                
                if saved_dim != self.embedding_dim:
                    self.logger.warning(f"Dimension mismatch: saved={saved_dim}, current={self.embedding_dim}")
                    return False
            
            active_count = sum(1 for metadata in self.document_metadata.values() 
                              if not metadata.get("deleted", False))
            
            self.logger.info(f"Loaded vector store with {active_count} active documents")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load vector store: {e}")
            return False


class ContextRetriever:
    """High-level interface for context retrieval using vector search."""
    
    def __init__(self, vector_store: VectorStore):
        """
        Initialize context retriever.
        
        Args:
            vector_store: VectorStore instance to use for search
        """
        self.vector_store = vector_store
        self.logger = get_logging_manager().get_logger(__name__.split(".")[-1], LogCategory.SYSTEM)
    
    def retrieve_context(self, query: str, user_id: str, max_results: int = 5, 
                        min_similarity: float = 0.3) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context for a query.
        
        Args:
            query: User query or context
            user_id: User ID to filter results
            max_results: Maximum number of results to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of relevant context items
        """
        try:
            # Search for similar memories
            results = self.vector_store.search_similar(
                query=query,
                k=max_results * 2,  # Get more results for filtering
                user_id=user_id,
                min_confidence=0.5  # Only include confident memories
            )
            
            # Filter by similarity threshold
            filtered_results = [
                result for result in results 
                if result["similarity_score"] >= min_similarity
            ]
            
            # Limit to max_results
            context_items = filtered_results[:max_results]
            
            self.logger.debug(f"Retrieved {len(context_items)} context items for query: {query[:50]}...")
            
            return context_items
            
        except Exception as e:
            self.logger.error(f"Context retrieval failed: {e}")
            return []
    
    def get_user_context_summary(self, user_id: str, limit: int = 20) -> Dict[str, Any]:
        """
        Get a summary of user's context based on their memories.
        
        Args:
            user_id: User ID
            limit: Maximum number of memories to analyze
            
        Returns:
            Dictionary with context summary
        """
        try:
            # Get recent memories for the user by searching with a broad query
            results = self.vector_store.search_similar(
                query="business goals objectives plans",  # Broad query to get diverse results
                k=limit,
                user_id=user_id
            )
            
            if not results:
                return {"user_id": user_id, "summary": "No context available"}
            
            # Analyze memory types and confidence
            memory_types = {}
            total_confidence = 0
            recent_topics = []
            
            for result in results:
                memory_type = result["memory_type"]
                memory_types[memory_type] = memory_types.get(memory_type, 0) + 1
                total_confidence += result["confidence"]
                
                # Extract key topics (simplified)
                content = result["content"]
                if len(content) > 50:
                    recent_topics.append(content[:50] + "...")
                else:
                    recent_topics.append(content)
            
            avg_confidence = total_confidence / len(results) if results else 0
            
            summary = {
                "user_id": user_id,
                "total_memories": len(results),
                "memory_types": memory_types,
                "avg_confidence": round(avg_confidence, 3),
                "recent_topics": recent_topics[:5],  # Top 5 topics
                "context_quality": "high" if avg_confidence > 0.8 else "medium" if avg_confidence > 0.6 else "low"
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to get user context summary: {e}")
            return {"user_id": user_id, "error": str(e)}


# Global vector store instance
_vector_store: Optional[VectorStore] = None
_context_retriever: Optional[ContextRetriever] = None


def get_vector_store() -> VectorStore:
    """Get the global vector store instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


def get_context_retriever() -> ContextRetriever:
    """Get the global context retriever instance."""
    global _context_retriever
    if _context_retriever is None:
        _context_retriever = ContextRetriever(get_vector_store())
    return _context_retriever