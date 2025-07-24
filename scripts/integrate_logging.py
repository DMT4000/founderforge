#!/usr/bin/env python3
"""
Script to integrate all FounderForge components with centralized logging.

This script updates components that are still using basic Python logging
to use the centralized logging manager instead.
"""

import os
import sys
import re
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def update_component_logging(file_path: Path) -> bool:
    """
    Update a component file to use centralized logging.
    
    Args:
        file_path: Path to the component file
        
    Returns:
        True if file was updated, False otherwise
    """
    if not file_path.exists():
        return False
    
    try:
        content = file_path.read_text(encoding='utf-8')
        original_content = content
        
        # Skip if already using centralized logging
        if 'get_logging_manager' in content or 'LoggingManager' in content:
            print(f"✓ {file_path.name} already uses centralized logging")
            return False
        
        # Skip if no logging import
        if 'import logging' not in content:
            print(f"- {file_path.name} doesn't use logging")
            return False
        
        print(f"Updating {file_path.name}...")
        
        # Add centralized logging import
        logging_import_pattern = r'import logging'
        if re.search(logging_import_pattern, content):
            # Replace basic logging import with centralized logging
            content = re.sub(
                r'import logging',
                'import logging\nfrom logging_manager import get_logging_manager, LogLevel, LogCategory',
                content,
                count=1
            )
        
        # Update logger initialization patterns
        logger_patterns = [
            (r'logging\.getLogger\(__name__\)', 'get_logging_manager().get_logger(__name__.split(".")[-1], LogCategory.SYSTEM)'),
            (r'logging\.getLogger\("([^"]+)"\)', r'get_logging_manager().get_logger("\1", LogCategory.SYSTEM)'),
            (r'logging\.getLogger\(([^)]+)\)', r'get_logging_manager().get_logger(\1, LogCategory.SYSTEM)'),
            (r'self\.logger = logging\.getLogger\(__name__\)', 'self.logger = get_logging_manager().get_logger(self.__class__.__name__, LogCategory.SYSTEM)'),
        ]
        
        for pattern, replacement in logger_patterns:
            content = re.sub(pattern, replacement, content)
        
        # Update basic logging calls to use structured logging
        basic_logging_patterns = [
            (r'logging\.info\(([^)]+)\)', r'get_logging_manager().log_structured(LogLevel.INFO, LogCategory.SYSTEM, __name__.split(".")[-1], \1)'),
            (r'logging\.error\(([^)]+)\)', r'get_logging_manager().log_structured(LogLevel.ERROR, LogCategory.ERROR, __name__.split(".")[-1], \1)'),
            (r'logging\.warning\(([^)]+)\)', r'get_logging_manager().log_structured(LogLevel.WARNING, LogCategory.SYSTEM, __name__.split(".")[-1], \1)'),
            (r'logging\.debug\(([^)]+)\)', r'get_logging_manager().log_structured(LogLevel.DEBUG, LogCategory.SYSTEM, __name__.split(".")[-1], \1)'),
        ]
        
        for pattern, replacement in basic_logging_patterns:
            content = re.sub(pattern, replacement, content)
        
        # Only write if content changed
        if content != original_content:
            file_path.write_text(content, encoding='utf-8')
            print(f"✓ Updated {file_path.name}")
            return True
        else:
            print(f"- No changes needed for {file_path.name}")
            return False
            
    except Exception as e:
        print(f"✗ Error updating {file_path.name}: {e}")
        return False


def main():
    """Main function to update all components"""
    src_dir = Path(__file__).parent.parent / "src"
    
    if not src_dir.exists():
        print("Error: src directory not found")
        return
    
    print("Integrating FounderForge components with centralized logging...")
    print("=" * 60)
    
    updated_files = []
    
    # Get all Python files in src directory
    python_files = list(src_dir.glob("*.py"))
    
    # Skip certain files
    skip_files = {
        "__init__.py",
        "logging_manager.py",  # Already the centralized logging
        "performance_monitor.py",  # Already integrated
        "system_integration.py",  # Already integrated
        "context_manager.py"  # Already integrated
    }
    
    for py_file in python_files:
        if py_file.name in skip_files:
            continue
            
        if update_component_logging(py_file):
            updated_files.append(py_file.name)
    
    print("=" * 60)
    print(f"Integration complete!")
    print(f"Updated {len(updated_files)} files: {', '.join(updated_files)}")
    
    if updated_files:
        print("\nNext steps:")
        print("1. Review the updated files for any syntax issues")
        print("2. Run tests to ensure everything works correctly")
        print("3. Initialize the system using: from src.system_integration import initialize_system")


if __name__ == "__main__":
    main()