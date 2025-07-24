#!/usr/bin/env python3
"""CLI tool for managing feature flags at runtime."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from feature_flag_manager import get_feature_flag_manager, FeatureFlagManager

def list_flags(manager: FeatureFlagManager) -> None:
    """List all feature flags and their current values."""
    flags = manager.get_all_flags()
    
    if not flags:
        print("No feature flags found.")
        return
    
    print("Current Feature Flags:")
    print("=" * 50)
    
    for flag_name, value in sorted(flags.items()):
        status = "✓" if value is True else "✗" if value is False else "•"
        print(f"{status} {flag_name}: {value}")

def get_flag_value(manager: FeatureFlagManager, flag_name: str) -> None:
    """Get and display a specific flag value."""
    value = manager.get_flag(flag_name)
    if value is None:
        print(f"Flag '{flag_name}' not found.")
        sys.exit(1)
    
    print(f"{flag_name}: {value}")

def set_flag_value(manager: FeatureFlagManager, flag_name: str, value: str, changed_by: str, reason: str) -> None:
    """Set a feature flag value."""
    # Parse the value
    parsed_value = parse_value(value)
    
    try:
        success = manager.set_flag(flag_name, parsed_value, changed_by, reason)
        if success:
            print(f"✓ Updated '{flag_name}' to {parsed_value}")
        else:
            print(f"• '{flag_name}' already had value {parsed_value}")
    except Exception as e:
        print(f"✗ Error setting flag: {e}")
        sys.exit(1)

def toggle_flag_value(manager: FeatureFlagManager, flag_name: str, changed_by: str, reason: str) -> None:
    """Toggle a boolean feature flag."""
    try:
        new_value = manager.toggle_flag(flag_name, changed_by, reason)
        print(f"✓ Toggled '{flag_name}' to {new_value}")
    except Exception as e:
        print(f"✗ Error toggling flag: {e}")
        sys.exit(1)

def show_history(manager: FeatureFlagManager, flag_name: str = None, limit: int = 20) -> None:
    """Show change history for feature flags."""
    changes = manager.get_change_history(flag_name, limit)
    
    if not changes:
        if flag_name:
            print(f"No change history found for flag '{flag_name}'.")
        else:
            print("No change history found.")
        return
    
    title = f"Change History for '{flag_name}'" if flag_name else "Feature Flag Change History"
    print(title)
    print("=" * len(title))
    
    for change in changes:
        print(f"[{change.timestamp}] {change.flag_name}")
        print(f"  Changed by: {change.changed_by}")
        print(f"  {change.old_value} → {change.new_value}")
        if change.reason:
            print(f"  Reason: {change.reason}")
        print()

def validate_flags(manager: FeatureFlagManager) -> None:
    """Validate feature flags configuration."""
    issues = manager.validate_flags()
    
    if not issues:
        print("✓ All feature flags are valid.")
        return
    
    print("✗ Feature flag validation issues:")
    for issue in issues:
        print(f"  • {issue}")
    sys.exit(1)

def export_flags(manager: FeatureFlagManager, file_path: str) -> None:
    """Export feature flags to a file."""
    try:
        manager.export_flags(Path(file_path))
        print(f"✓ Exported feature flags to {file_path}")
    except Exception as e:
        print(f"✗ Error exporting flags: {e}")
        sys.exit(1)

def import_flags(manager: FeatureFlagManager, file_path: str, changed_by: str) -> None:
    """Import feature flags from a file."""
    try:
        changes_made = manager.import_flags(Path(file_path), changed_by, f"Import from {file_path}")
        print(f"✓ Imported {changes_made} feature flag changes from {file_path}")
    except Exception as e:
        print(f"✗ Error importing flags: {e}")
        sys.exit(1)

def parse_value(value_str: str) -> Any:
    """Parse a string value to appropriate Python type."""
    # Handle boolean values
    if value_str.lower() in ('true', 'yes', '1', 'on', 'enabled'):
        return True
    elif value_str.lower() in ('false', 'no', '0', 'off', 'disabled'):
        return False
    
    # Try to parse as number
    try:
        if '.' in value_str:
            return float(value_str)
        else:
            return int(value_str)
    except ValueError:
        pass
    
    # Try to parse as JSON (for complex values)
    try:
        return json.loads(value_str)
    except json.JSONDecodeError:
        pass
    
    # Return as string
    return value_str

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Manage FounderForge feature flags at runtime",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s list                                    # List all flags
  %(prog)s get enable_memory_system                # Get specific flag
  %(prog)s set enable_memory_system false          # Set flag value
  %(prog)s toggle enable_debug_mode                # Toggle boolean flag
  %(prog)s history                                 # Show change history
  %(prog)s history enable_memory_system            # Show history for specific flag
  %(prog)s validate                                # Validate configuration
  %(prog)s export flags_backup.json               # Export flags
  %(prog)s import flags_backup.json               # Import flags
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List command
    subparsers.add_parser('list', help='List all feature flags')
    
    # Get command
    get_parser = subparsers.add_parser('get', help='Get a specific flag value')
    get_parser.add_argument('flag_name', help='Name of the flag to get')
    
    # Set command
    set_parser = subparsers.add_parser('set', help='Set a flag value')
    set_parser.add_argument('flag_name', help='Name of the flag to set')
    set_parser.add_argument('value', help='Value to set (true/false, number, or string)')
    set_parser.add_argument('--changed-by', default='cli', help='Who made the change')
    set_parser.add_argument('--reason', help='Reason for the change')
    
    # Toggle command
    toggle_parser = subparsers.add_parser('toggle', help='Toggle a boolean flag')
    toggle_parser.add_argument('flag_name', help='Name of the flag to toggle')
    toggle_parser.add_argument('--changed-by', default='cli', help='Who made the change')
    toggle_parser.add_argument('--reason', help='Reason for the change')
    
    # History command
    history_parser = subparsers.add_parser('history', help='Show change history')
    history_parser.add_argument('flag_name', nargs='?', help='Specific flag to show history for')
    history_parser.add_argument('--limit', type=int, default=20, help='Number of changes to show')
    
    # Validate command
    subparsers.add_parser('validate', help='Validate feature flags configuration')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export flags to file')
    export_parser.add_argument('file_path', help='File path to export to')
    
    # Import command
    import_parser = subparsers.add_parser('import', help='Import flags from file')
    import_parser.add_argument('file_path', help='File path to import from')
    import_parser.add_argument('--changed-by', default='cli-import', help='Who made the change')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Initialize feature flag manager
    manager = get_feature_flag_manager()
    
    # Execute command
    try:
        if args.command == 'list':
            list_flags(manager)
        elif args.command == 'get':
            get_flag_value(manager, args.flag_name)
        elif args.command == 'set':
            set_flag_value(manager, args.flag_name, args.value, args.changed_by, args.reason)
        elif args.command == 'toggle':
            toggle_flag_value(manager, args.flag_name, args.changed_by, args.reason)
        elif args.command == 'history':
            show_history(manager, args.flag_name, args.limit)
        elif args.command == 'validate':
            validate_flags(manager)
        elif args.command == 'export':
            export_flags(manager, args.file_path)
        elif args.command == 'import':
            import_flags(manager, args.file_path, args.changed_by)
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
        sys.exit(1)

if __name__ == '__main__':
    main()