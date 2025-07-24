"""
Memory repository for SQLite-based memory storage with CRUD operations.
Implements high-performance memory retrieval and management with confirmation prompts.
"""

import sqlite3
import logging
from logging_manager import get_logging_manager, LogLevel, LogCategory
import time
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from contextlib import contextmanager

from .database import get_db_manager, DatabaseManager
from .models import Memory, MemoryType, ValidationError


class MemoryRepository:
    """Repository for managing memory storage and retrieval operations."""
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """
        Initialize memory repository.
        
        Args:
            db_manager: Database manager instance (uses global if None)
        """
        self.db_manager = db_manager or get_db_manager()
        self.logger = get_logging_manager().get_logger(__name__.split(".")[-1], LogCategory.SYSTEM)
        
        # Performance tracking
        self._query_times: List[float] = []
        self._performance_threshold = 0.010  # 10ms target
    
    def create_memory(self, memory: Memory, confirm: bool = True) -> bool:
        """
        Store a new memory with optional confirmation prompt.
        
        Args:
            memory: Memory object to store
            confirm: Whether to prompt for confirmation before storing
            
        Returns:
            True if memory stored successfully, False otherwise
        """
        if confirm and not self._confirm_storage(memory):
            self.logger.info(f"Memory storage cancelled by user for memory: {memory.id}")
            return False
        
        start_time = time.perf_counter()
        
        try:
            query = """
                INSERT INTO memories (id, user_id, content, memory_type, confidence, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            params = (
                memory.id,
                memory.user_id,
                memory.content,
                memory.memory_type.value,
                memory.confidence,
                memory.created_at.isoformat(),
                memory.expires_at.isoformat() if memory.expires_at else None
            )
            
            success = self.db_manager.execute_update(query, params)
            
            if success:
                self.logger.info(f"Memory stored successfully: {memory.id}")
                self._log_access_audit(memory.user_id, "CREATE", memory.id)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to create memory {memory.id}: {e}")
            return False
        finally:
            execution_time = time.perf_counter() - start_time
            self._track_performance(execution_time)
    
    def get_memory_by_id(self, memory_id: str) -> Optional[Memory]:
        """
        Retrieve a specific memory by ID with performance tracking.
        
        Args:
            memory_id: Unique memory identifier
            
        Returns:
            Memory object if found, None otherwise
        """
        start_time = time.perf_counter()
        
        try:
            query = """
                SELECT id, user_id, content, memory_type, confidence, created_at, expires_at
                FROM memories 
                WHERE id = ?
            """
            
            results = self.db_manager.execute_query(query, (memory_id,))
            
            if results and len(results) > 0:
                row = results[0]
                memory = self._row_to_memory(row)
                
                # Check if memory has expired
                if memory.is_expired():
                    self.logger.warning(f"Retrieved expired memory: {memory_id}")
                
                return memory
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve memory {memory_id}: {e}")
            return None
        finally:
            execution_time = time.perf_counter() - start_time
            self._track_performance(execution_time)
    
    def get_memories_by_user(self, user_id: str, memory_type: Optional[MemoryType] = None, 
                           limit: int = 50, include_expired: bool = False) -> List[Memory]:
        """
        Retrieve memories for a specific user with sub-10ms performance target.
        
        Args:
            user_id: User identifier
            memory_type: Filter by memory type (optional)
            limit: Maximum number of memories to return
            include_expired: Whether to include expired memories
            
        Returns:
            List of Memory objects
        """
        start_time = time.perf_counter()
        
        try:
            # Build query with optional filters
            query_parts = [
                "SELECT id, user_id, content, memory_type, confidence, created_at, expires_at",
                "FROM memories",
                "WHERE user_id = ?"
            ]
            params = [user_id]
            
            if memory_type:
                query_parts.append("AND memory_type = ?")
                params.append(memory_type.value)
            
            if not include_expired:
                query_parts.append("AND (expires_at IS NULL OR expires_at > ?)")
                params.append(datetime.now().isoformat())
            
            query_parts.extend([
                "ORDER BY created_at DESC",
                f"LIMIT {limit}"
            ])
            
            query = " ".join(query_parts)
            results = self.db_manager.execute_query(query, tuple(params))
            
            memories = []
            if results:
                for row in results:
                    memory = self._row_to_memory(row)
                    memories.append(memory)
                    
                self._log_access_audit(user_id, "READ", f"Retrieved {len(memories)} memories")
            
            return memories
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve memories for user {user_id}: {e}")
            return []
        finally:
            execution_time = time.perf_counter() - start_time
            self._track_performance(execution_time)
            
            # Log performance warning if threshold exceeded
            if execution_time > self._performance_threshold:
                self.logger.warning(
                    f"Memory retrieval exceeded {self._performance_threshold*1000:.1f}ms target: "
                    f"{execution_time*1000:.2f}ms for user {user_id}"
                )
    
    def search_memories(self, user_id: str, search_term: str, limit: int = 20) -> List[Memory]:
        """
        Search memories by content with full-text search.
        
        Args:
            user_id: User identifier
            search_term: Text to search for in memory content
            limit: Maximum number of results
            
        Returns:
            List of matching Memory objects
        """
        start_time = time.perf_counter()
        
        try:
            query = """
                SELECT id, user_id, content, memory_type, confidence, created_at, expires_at
                FROM memories 
                WHERE user_id = ? 
                AND content LIKE ? 
                AND (expires_at IS NULL OR expires_at > ?)
                ORDER BY confidence DESC, created_at DESC
                LIMIT ?
            """
            
            search_pattern = f"%{search_term}%"
            params = (user_id, search_pattern, datetime.now().isoformat(), limit)
            
            results = self.db_manager.execute_query(query, params)
            
            memories = []
            if results:
                for row in results:
                    memory = self._row_to_memory(row)
                    memories.append(memory)
                    
                self._log_access_audit(user_id, "SEARCH", f"Found {len(memories)} memories for '{search_term}'")
            
            return memories
            
        except Exception as e:
            self.logger.error(f"Failed to search memories for user {user_id}: {e}")
            return []
        finally:
            execution_time = time.perf_counter() - start_time
            self._track_performance(execution_time)
    
    def update_memory(self, memory: Memory, confirm: bool = True) -> bool:
        """
        Update an existing memory with confirmation.
        
        Args:
            memory: Updated memory object
            confirm: Whether to prompt for confirmation
            
        Returns:
            True if update successful, False otherwise
        """
        if confirm and not self._confirm_update(memory):
            self.logger.info(f"Memory update cancelled by user for memory: {memory.id}")
            return False
        
        start_time = time.perf_counter()
        
        try:
            query = """
                UPDATE memories 
                SET content = ?, memory_type = ?, confidence = ?, expires_at = ?
                WHERE id = ? AND user_id = ?
            """
            params = (
                memory.content,
                memory.memory_type.value,
                memory.confidence,
                memory.expires_at.isoformat() if memory.expires_at else None,
                memory.id,
                memory.user_id
            )
            
            success = self.db_manager.execute_update(query, params)
            
            if success:
                self.logger.info(f"Memory updated successfully: {memory.id}")
                self._log_access_audit(memory.user_id, "UPDATE", memory.id)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to update memory {memory.id}: {e}")
            return False
        finally:
            execution_time = time.perf_counter() - start_time
            self._track_performance(execution_time)
    
    def delete_memory(self, memory_id: str, user_id: str, confirm: bool = True) -> bool:
        """
        Delete a memory with confirmation prompt.
        
        Args:
            memory_id: Memory identifier to delete
            user_id: User identifier for security
            confirm: Whether to prompt for confirmation
            
        Returns:
            True if deletion successful, False otherwise
        """
        # Get memory details for confirmation
        memory = self.get_memory_by_id(memory_id)
        if not memory or memory.user_id != user_id:
            self.logger.warning(f"Memory not found or access denied: {memory_id}")
            return False
        
        if confirm and not self._confirm_deletion(memory):
            self.logger.info(f"Memory deletion cancelled by user for memory: {memory_id}")
            return False
        
        start_time = time.perf_counter()
        
        try:
            query = "DELETE FROM memories WHERE id = ? AND user_id = ?"
            success = self.db_manager.execute_update(query, (memory_id, user_id))
            
            if success:
                self.logger.info(f"Memory deleted successfully: {memory_id}")
                self._log_access_audit(user_id, "DELETE", memory_id)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to delete memory {memory_id}: {e}")
            return False
        finally:
            execution_time = time.perf_counter() - start_time
            self._track_performance(execution_time)
    
    def delete_user_memories(self, user_id: str, memory_type: Optional[MemoryType] = None, 
                           confirm: bool = True) -> int:
        """
        Delete all memories for a user with confirmation.
        
        Args:
            user_id: User identifier
            memory_type: Optional filter by memory type
            confirm: Whether to prompt for confirmation
            
        Returns:
            Number of memories deleted
        """
        # Get count for confirmation
        memories = self.get_memories_by_user(user_id, memory_type, limit=1000, include_expired=True)
        count = len(memories)
        
        if count == 0:
            self.logger.info(f"No memories found for user {user_id}")
            return 0
        
        if confirm and not self._confirm_bulk_deletion(user_id, count, memory_type):
            self.logger.info(f"Bulk memory deletion cancelled by user for user: {user_id}")
            return 0
        
        start_time = time.perf_counter()
        
        try:
            if memory_type:
                query = "DELETE FROM memories WHERE user_id = ? AND memory_type = ?"
                params = (user_id, memory_type.value)
            else:
                query = "DELETE FROM memories WHERE user_id = ?"
                params = (user_id,)
            
            success = self.db_manager.execute_update(query, params)
            
            if success:
                self.logger.info(f"Deleted {count} memories for user {user_id}")
                self._log_access_audit(user_id, "BULK_DELETE", f"Deleted {count} memories")
                return count
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Failed to delete memories for user {user_id}: {e}")
            return 0
        finally:
            execution_time = time.perf_counter() - start_time
            self._track_performance(execution_time)
    
    def cleanup_expired_memories(self) -> int:
        """
        Remove expired memories from the database.
        
        Returns:
            Number of expired memories removed
        """
        start_time = time.perf_counter()
        
        try:
            # First, get count of expired memories
            count_query = """
                SELECT COUNT(*) as count 
                FROM memories 
                WHERE expires_at IS NOT NULL AND expires_at <= ?
            """
            
            results = self.db_manager.execute_query(count_query, (datetime.now().isoformat(),))
            count = results[0]['count'] if results else 0
            
            if count == 0:
                return 0
            
            # Delete expired memories
            delete_query = """
                DELETE FROM memories 
                WHERE expires_at IS NOT NULL AND expires_at <= ?
            """
            
            success = self.db_manager.execute_update(delete_query, (datetime.now().isoformat(),))
            
            if success:
                self.logger.info(f"Cleaned up {count} expired memories")
                return count
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup expired memories: {e}")
            return 0
        finally:
            execution_time = time.perf_counter() - start_time
            self._track_performance(execution_time)
    
    def get_memory_stats(self, user_id: str) -> Dict[str, Any]:
        """
        Get memory statistics for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary with memory statistics
        """
        start_time = time.perf_counter()
        
        try:
            query = """
                SELECT 
                    memory_type,
                    COUNT(*) as count,
                    AVG(confidence) as avg_confidence,
                    MIN(created_at) as oldest,
                    MAX(created_at) as newest
                FROM memories 
                WHERE user_id = ? 
                AND (expires_at IS NULL OR expires_at > ?)
                GROUP BY memory_type
            """
            
            results = self.db_manager.execute_query(query, (user_id, datetime.now().isoformat()))
            
            stats = {
                "user_id": user_id,
                "total_memories": 0,
                "by_type": {},
                "avg_retrieval_time_ms": self.get_avg_performance() * 1000,
                "performance_threshold_ms": self._performance_threshold * 1000
            }
            
            if results:
                for row in results:
                    memory_type = row['memory_type']
                    stats["by_type"][memory_type] = {
                        "count": row['count'],
                        "avg_confidence": round(row['avg_confidence'], 3),
                        "oldest": row['oldest'],
                        "newest": row['newest']
                    }
                    stats["total_memories"] += row['count']
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get memory stats for user {user_id}: {e}")
            return {"user_id": user_id, "error": str(e)}
        finally:
            execution_time = time.perf_counter() - start_time
            self._track_performance(execution_time)
    
    def get_avg_performance(self) -> float:
        """Get average query performance in seconds."""
        if not self._query_times:
            return 0.0
        return sum(self._query_times) / len(self._query_times)
    
    def reset_performance_tracking(self) -> None:
        """Reset performance tracking metrics."""
        self._query_times.clear()
        self.logger.info("Performance tracking metrics reset")
    
    # Private helper methods
    
    def _row_to_memory(self, row) -> Memory:
        """Convert database row to Memory object."""
        return Memory(
            id=row['id'],
            user_id=row['user_id'],
            content=row['content'],
            memory_type=MemoryType(row['memory_type']),
            confidence=row['confidence'],
            created_at=datetime.fromisoformat(row['created_at']),
            expires_at=datetime.fromisoformat(row['expires_at']) if row['expires_at'] else None
        )
    
    def _track_performance(self, execution_time: float) -> None:
        """Track query performance for monitoring."""
        self._query_times.append(execution_time)
        
        # Keep only last 100 measurements
        if len(self._query_times) > 100:
            self._query_times = self._query_times[-100:]
    
    def _log_access_audit(self, user_id: str, action: str, details: str) -> None:
        """Log data access for privacy auditing simulation."""
        timestamp = datetime.now().isoformat()
        audit_entry = f"{timestamp} | USER:{user_id} | ACTION:{action} | DETAILS:{details}"
        
        # In a real implementation, this would write to a secure audit log
        self.logger.info(f"AUDIT: {audit_entry}")
    
    def _confirm_storage(self, memory: Memory) -> bool:
        """
        Prompt for confirmation before storing sensitive memory.
        
        Args:
            memory: Memory object to be stored
            
        Returns:
            True if user confirms, False otherwise
        """
        # In a real implementation, this would show a UI prompt
        # For now, we'll use a simple heuristic based on content sensitivity
        sensitive_keywords = ['password', 'ssn', 'credit card', 'bank account', 'api key']
        content_lower = memory.content.lower()
        
        if any(keyword in content_lower for keyword in sensitive_keywords):
            self.logger.warning(f"Sensitive content detected in memory {memory.id}, requiring confirmation")
            # In CLI/testing mode, we'll assume confirmation for sensitive data
            return True
        
        # Non-sensitive content can be stored without explicit confirmation
        return True
    
    def _confirm_update(self, memory: Memory) -> bool:
        """Prompt for confirmation before updating memory."""
        # Similar logic to storage confirmation
        return self._confirm_storage(memory)
    
    def _confirm_deletion(self, memory: Memory) -> bool:
        """
        Prompt for confirmation before deleting memory.
        
        Args:
            memory: Memory object to be deleted
            
        Returns:
            True if user confirms, False otherwise
        """
        self.logger.info(f"Deletion requested for memory: {memory.id[:8]}... (confidence: {memory.confidence})")
        # In a real implementation, this would show a confirmation dialog
        return True
    
    def _confirm_bulk_deletion(self, user_id: str, count: int, memory_type: Optional[MemoryType]) -> bool:
        """
        Prompt for confirmation before bulk deletion.
        
        Args:
            user_id: User identifier
            count: Number of memories to delete
            memory_type: Type filter if applicable
            
        Returns:
            True if user confirms, False otherwise
        """
        type_filter = f" of type {memory_type.value}" if memory_type else ""
        self.logger.warning(f"Bulk deletion requested: {count} memories{type_filter} for user {user_id}")
        # In a real implementation, this would require explicit confirmation
        return True


# Global memory repository instance
_memory_repository: Optional[MemoryRepository] = None


def get_memory_repository() -> MemoryRepository:
    """Get the global memory repository instance."""
    global _memory_repository
    if _memory_repository is None:
        _memory_repository = MemoryRepository()
    return _memory_repository