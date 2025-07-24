"""
Database connection manager and schema utilities for FounderForge AI Cofounder.
Provides SQLite database setup, connection management, and error handling.
"""

import sqlite3
import logging
from pathlib import Path
from typing import Optional, Any, Dict
from contextlib import contextmanager


class DatabaseManager:
    """Manages SQLite database connections and schema operations."""
    
    def __init__(self, db_path: str = "data/founderforge.db"):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection: Optional[sqlite3.Connection] = None
        self.logger = logging.getLogger(__name__)
        
    def get_connection(self) -> sqlite3.Connection:
        """
        Get database connection with proper configuration.
        
        Returns:
            SQLite connection object
            
        Raises:
            sqlite3.Error: If connection fails
        """
        try:
            if self._connection is None or self._connection.execute("SELECT 1").fetchone() is None:
                self._connection = sqlite3.connect(
                    str(self.db_path),
                    check_same_thread=False,
                    timeout=30.0
                )
                # Enable foreign key constraints
                self._connection.execute("PRAGMA foreign_keys = ON")
                # Set WAL mode for better concurrency
                self._connection.execute("PRAGMA journal_mode = WAL")
                self._connection.row_factory = sqlite3.Row
                
            return self._connection
            
        except sqlite3.Error as e:
            self.logger.error(f"Database connection failed: {e}")
            raise
    
    @contextmanager
    def get_cursor(self):
        """
        Context manager for database cursor with automatic commit/rollback.
        
        Yields:
            SQLite cursor object
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Database operation failed: {e}")
            raise
        finally:
            cursor.close()
    
    def initialize_schema(self) -> bool:
        """
        Create database tables and indexes if they don't exist.
        
        Returns:
            True if schema creation successful, False otherwise
        """
        try:
            with self.get_cursor() as cursor:
                # Create users table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id TEXT PRIMARY KEY,
                        name TEXT,
                        email TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        preferences TEXT  -- JSON blob for user preferences
                    )
                """)
                
                # Create memories table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS memories (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        content TEXT NOT NULL,
                        memory_type TEXT NOT NULL CHECK (memory_type IN ('SHORT_TERM', 'LONG_TERM')),
                        confidence REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        expires_at TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                    )
                """)
                
                # Create conversations table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS conversations (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        message TEXT NOT NULL,
                        response TEXT NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        token_usage REAL DEFAULT 0.0,
                        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                    )
                """)
                
                # Create performance-optimized indexes
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_memory_user 
                    ON memories(user_id)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_memory_type 
                    ON memories(user_id, memory_type)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_memory_created 
                    ON memories(user_id, created_at DESC)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_conversation_user 
                    ON conversations(user_id, timestamp DESC)
                """)
                
                self.logger.info("Database schema initialized successfully")
                return True
                
        except sqlite3.Error as e:
            self.logger.error(f"Schema initialization failed: {e}")
            return False
    
    def execute_query(self, query: str, params: tuple = ()) -> Optional[Any]:
        """
        Execute a SELECT query and return results.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            Query results or None if error
        """
        try:
            with self.get_cursor() as cursor:
                cursor.execute(query, params)
                return cursor.fetchall()
        except sqlite3.Error as e:
            self.logger.error(f"Query execution failed: {e}")
            return None
    
    def execute_update(self, query: str, params: tuple = ()) -> bool:
        """
        Execute an INSERT, UPDATE, or DELETE query.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.get_cursor() as cursor:
                cursor.execute(query, params)
                return True
        except sqlite3.Error as e:
            self.logger.error(f"Update execution failed: {e}")
            return False
    
    def get_table_info(self, table_name: str) -> Optional[list]:
        """
        Get table schema information for debugging.
        
        Args:
            table_name: Name of table to inspect
            
        Returns:
            Table schema info or None if error
        """
        return self.execute_query(f"PRAGMA table_info({table_name})")
    
    def close(self):
        """Close database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None
            self.logger.info("Database connection closed")


# Global database manager instance
db_manager = DatabaseManager()


def get_db_manager() -> DatabaseManager:
    """Get the global database manager instance."""
    return db_manager


def initialize_database() -> bool:
    """Initialize database schema using global manager."""
    return db_manager.initialize_schema()