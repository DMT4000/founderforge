#!/usr/bin/env python3
"""Setup verification script for FounderForge AI Cofounder."""

import sys
from pathlib import Path
import subprocess

def check_directory_structure():
    """Verify required directories exist."""
    required_dirs = ['src', 'tests', 'data', 'config']
    missing_dirs = []
    
    for dir_name in required_dirs:
        if not Path(dir_name).exists():
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"❌ Missing directories: {', '.join(missing_dirs)}")
        return False
    else:
        print("✅ All required directories present")
        return True

def check_required_files():
    """Verify required files exist."""
    required_files = [
        'requirements.txt',
        '.env.example',
        '.gitignore',
        'config/settings.py',
        'src/__init__.py',
        'tests/__init__.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
        return False
    else:
        print("✅ All required files present")
        return True

def check_git_repository():
    """Verify Git repository is initialized."""
    if not Path('.git').exists():
        print("❌ Git repository not initialized")
        return False
    else:
        print("✅ Git repository initialized")
        return True

def check_configuration():
    """Verify configuration system works."""
    try:
        from config.settings import settings
        
        # Test feature flags
        flags = settings.load_feature_flags()
        if not flags:
            print("❌ Feature flags not loading properly")
            return False
        
        # Test basic settings
        db_path = settings.database_path
        if not db_path:
            print("❌ Database path not configured")
            return False
        
        print("✅ Configuration system working")
        print(f"   Database path: {db_path}")
        print(f"   Feature flags loaded: {len(flags)} flags")
        return True
        
    except Exception as e:
        print(f"❌ Configuration system error: {e}")
        return False

def main():
    """Run all verification checks."""
    print("🔍 FounderForge Setup Verification")
    print("=" * 40)
    
    checks = [
        ("Directory Structure", check_directory_structure),
        ("Required Files", check_required_files),
        ("Git Repository", check_git_repository),
        ("Configuration System", check_configuration)
    ]
    
    all_passed = True
    for check_name, check_func in checks:
        print(f"\n{check_name}:")
        if not check_func():
            all_passed = False
    
    print("\n" + "=" * 40)
    if all_passed:
        print("🎉 All checks passed! Setup is complete.")
        print("\nNext steps:")
        print("1. Copy .env.example to .env and add your GEMINI_API_KEY")
        print("2. Install dependencies: pip install -r requirements.txt")
        print("3. Start implementing the next task in the spec")
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()