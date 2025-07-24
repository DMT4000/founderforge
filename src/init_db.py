#!/usr/bin/env python3
"""
Database initialization script for FounderForge AI Cofounder.
Run this script to create the database schema.
"""

import logging
from logging_manager import get_logging_manager, LogLevel, LogCategory
from database import initialize_database, get_db_manager

def main():
    """Initialize the database schema."""
    logging.basicConfig(level=logging.INFO)
    
    print("Initializing FounderForge database...")
    
    if initialize_database():
        print("✅ Database schema created successfully!")
        
        # Verify tables were created
        db = get_db_manager()
        tables = ['users', 'memories', 'conversations']
        
        for table in tables:
            info = db.get_table_info(table)
            if info:
                print(f"✅ Table '{table}' created with {len(info)} columns")
            else:
                print(f"❌ Table '{table}' not found")
    else:
        print("❌ Database initialization failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())