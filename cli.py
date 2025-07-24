#!/usr/bin/env python3
"""
FounderForge AI Cofounder - Command Line Interface
CLI interface for power users with data management, system configuration, and batch processing.
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import uuid
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import core components
try:
    from src.database import initialize_database, get_db_manager
    from src.memory_repository import get_memory_repository
    from src.context_manager import ContextAssembler, TokenManager
    from src.agents import AgentOrchestrator
    from src.gemini_client import GeminiClient
    from src.confidence_manager import ConfidenceManager
    from src.models import (
        UserContext, Message, Memory, MemoryType, BusinessInfo, 
        UserPreferences, Response, TokenUsage
    )
    from config.settings import settings
except ImportError as e:
    print(f"Failed to import required modules: {e}")
    sys.exit(1)


class FounderForgeCLI:
    """Command Line Interface for FounderForge AI Cofounder."""
    
    def __init__(self):
        """Initialize the CLI application."""
        self.initialize_components()
    
    def initialize_components(self):
        """Initialize core system components."""
        try:
            # Initialize database
            if not initialize_database():
                print("Failed to initialize database")
                sys.exit(1)
            
            # Initialize core components
            self.db_manager = get_db_manager()
            self.memory_repository = get_memory_repository()
            self.context_manager = ContextAssembler()
            self.token_manager = TokenManager()
            self.gemini_client = GeminiClient()
            self.confidence_manager = ConfidenceManager()
            
            # Initialize agent orchestrator
            self.agent_orchestrator = AgentOrchestrator(
                gemini_client=self.gemini_client,
                context_manager=self.context_manager,
                confidence_manager=self.confidence_manager
            )
            
            logger.info("CLI components initialized successfully")
            
        except Exception as e:
            print(f"Failed to initialize CLI: {e}")
            sys.exit(1)
    
    def run(self):
        """Run the CLI application."""
        parser = self.create_parser()
        args = parser.parse_args()
        
        # Set logging level based on verbosity
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        elif args.quiet:
            logging.getLogger().setLevel(logging.WARNING)
        
        # Execute the appropriate command
        try:
            if hasattr(args, 'func'):
                args.func(args)
            else:
                parser.print_help()
        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Command failed: {e}")
            if args.verbose:
                raise
            sys.exit(1)
    
    def create_parser(self):
        """Create the argument parser with all commands."""
        parser = argparse.ArgumentParser(
            description="FounderForge AI Cofounder CLI",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  %(prog)s chat --user-id user123 "What's my next step?"
  %(prog)s memory list --user-id user123
  %(prog)s memory delete --user-id user123 --type SHORT_TERM
  %(prog)s user create --name "John Doe" --email "john@example.com"
  %(prog)s batch process --file queries.json
  %(prog)s system status
            """
        )
        
        # Global options
        parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
        parser.add_argument('-q', '--quiet', action='store_true', help='Suppress non-error output')
        parser.add_argument('--config', help='Configuration file path')
        
        # Create subparsers
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Chat command
        self.add_chat_parser(subparsers)
        
        # Memory management commands
        self.add_memory_parser(subparsers)
        
        # User management commands
        self.add_user_parser(subparsers)
        
        # System commands
        self.add_system_parser(subparsers)
        
        # Batch processing commands
        self.add_batch_parser(subparsers)
        
        # Configuration commands
        self.add_config_parser(subparsers)
        
        return parser
    
    def add_chat_parser(self, subparsers):
        """Add chat command parser."""
        chat_parser = subparsers.add_parser('chat', help='Interactive chat with AI')
        chat_parser.add_argument('--user-id', required=True, help='User ID')
        chat_parser.add_argument('message', nargs='?', help='Message to send (if not provided, starts interactive mode)')
        chat_parser.add_argument('--context', help='Additional context file')
        chat_parser.add_argument('--workflow', choices=['general', 'funding', 'planning'], 
                               help='Use specific agent workflow')
        chat_parser.add_argument('--output', help='Save response to file')
        chat_parser.set_defaults(func=self.cmd_chat)
    
    def add_memory_parser(self, subparsers):
        """Add memory management command parser."""
        memory_parser = subparsers.add_parser('memory', help='Memory management')
        memory_subparsers = memory_parser.add_subparsers(dest='memory_action', help='Memory actions')
        
        # List memories
        list_parser = memory_subparsers.add_parser('list', help='List memories')
        list_parser.add_argument('--user-id', required=True, help='User ID')
        list_parser.add_argument('--type', choices=['SHORT_TERM', 'LONG_TERM'], help='Memory type filter')
        list_parser.add_argument('--limit', type=int, default=50, help='Maximum number of memories to show')
        list_parser.add_argument('--format', choices=['table', 'json'], default='table', help='Output format')
        list_parser.set_defaults(func=self.cmd_memory_list)
        
        # Search memories
        search_parser = memory_subparsers.add_parser('search', help='Search memories')
        search_parser.add_argument('--user-id', required=True, help='User ID')
        search_parser.add_argument('query', help='Search query')
        search_parser.add_argument('--limit', type=int, default=10, help='Maximum results')
        search_parser.set_defaults(func=self.cmd_memory_search)
        
        # Delete memories
        delete_parser = memory_subparsers.add_parser('delete', help='Delete memories')
        delete_parser.add_argument('--user-id', required=True, help='User ID')
        delete_parser.add_argument('--type', choices=['SHORT_TERM', 'LONG_TERM'], help='Memory type to delete')
        delete_parser.add_argument('--memory-id', help='Specific memory ID to delete')
        delete_parser.add_argument('--confirm', action='store_true', help='Skip confirmation prompt')
        delete_parser.set_defaults(func=self.cmd_memory_delete)
        
        # Memory stats
        stats_parser = memory_subparsers.add_parser('stats', help='Memory statistics')
        stats_parser.add_argument('--user-id', required=True, help='User ID')
        stats_parser.set_defaults(func=self.cmd_memory_stats)
    
    def add_user_parser(self, subparsers):
        """Add user management command parser."""
        user_parser = subparsers.add_parser('user', help='User management')
        user_subparsers = user_parser.add_subparsers(dest='user_action', help='User actions')
        
        # List users
        list_parser = user_subparsers.add_parser('list', help='List users')
        list_parser.add_argument('--format', choices=['table', 'json'], default='table', help='Output format')
        list_parser.set_defaults(func=self.cmd_user_list)
        
        # Create user
        create_parser = user_subparsers.add_parser('create', help='Create user')
        create_parser.add_argument('--user-id', help='User ID (auto-generated if not provided)')
        create_parser.add_argument('--name', help='User name')
        create_parser.add_argument('--email', help='User email')
        create_parser.set_defaults(func=self.cmd_user_create)
        
        # Show user details
        show_parser = user_subparsers.add_parser('show', help='Show user details')
        show_parser.add_argument('--user-id', required=True, help='User ID')
        show_parser.set_defaults(func=self.cmd_user_show)
        
        # Delete user
        delete_parser = user_subparsers.add_parser('delete', help='Delete user')
        delete_parser.add_argument('--user-id', required=True, help='User ID')
        delete_parser.add_argument('--confirm', action='store_true', help='Skip confirmation prompt')
        delete_parser.set_defaults(func=self.cmd_user_delete)
    
    def add_system_parser(self, subparsers):
        """Add system command parser."""
        system_parser = subparsers.add_parser('system', help='System management')
        system_subparsers = system_parser.add_subparsers(dest='system_action', help='System actions')
        
        # System status
        status_parser = system_subparsers.add_parser('status', help='Show system status')
        status_parser.set_defaults(func=self.cmd_system_status)
        
        # Database operations
        db_parser = system_subparsers.add_parser('database', help='Database operations')
        db_parser.add_argument('--init', action='store_true', help='Initialize database')
        db_parser.add_argument('--backup', help='Backup database to file')
        db_parser.add_argument('--restore', help='Restore database from file')
        db_parser.set_defaults(func=self.cmd_system_database)
        
        # Token usage
        tokens_parser = system_subparsers.add_parser('tokens', help='Token usage statistics')
        tokens_parser.add_argument('--user-id', help='User ID filter')
        tokens_parser.add_argument('--days', type=int, default=7, help='Number of days to show')
        tokens_parser.set_defaults(func=self.cmd_system_tokens)
    
    def add_batch_parser(self, subparsers):
        """Add batch processing command parser."""
        batch_parser = subparsers.add_parser('batch', help='Batch processing')
        batch_subparsers = batch_parser.add_subparsers(dest='batch_action', help='Batch actions')
        
        # Process batch file
        process_parser = batch_subparsers.add_parser('process', help='Process batch file')
        process_parser.add_argument('--file', required=True, help='JSON file with batch queries')
        process_parser.add_argument('--output', help='Output file for results')
        process_parser.add_argument('--parallel', type=int, default=1, help='Number of parallel processes')
        process_parser.set_defaults(func=self.cmd_batch_process)
        
        # Test scenarios
        test_parser = batch_subparsers.add_parser('test', help='Run test scenarios')
        test_parser.add_argument('--scenarios', help='Test scenarios file')
        test_parser.add_argument('--output', help='Test results output file')
        test_parser.set_defaults(func=self.cmd_batch_test)
    
    def add_config_parser(self, subparsers):
        """Add configuration command parser."""
        config_parser = subparsers.add_parser('config', help='Configuration management')
        config_subparsers = config_parser.add_subparsers(dest='config_action', help='Config actions')
        
        # Show configuration
        show_parser = config_subparsers.add_parser('show', help='Show configuration')
        show_parser.set_defaults(func=self.cmd_config_show)
        
        # Set configuration
        set_parser = config_subparsers.add_parser('set', help='Set configuration value')
        set_parser.add_argument('key', help='Configuration key')
        set_parser.add_argument('value', help='Configuration value')
        set_parser.set_defaults(func=self.cmd_config_set)
    
    # Command implementations
    
    def cmd_chat(self, args):
        """Handle chat command."""
        user_id = args.user_id
        
        if args.message:
            # Single message mode
            response = self.process_message(user_id, args.message, args.workflow, args.context)
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(response, f, indent=2)
                print(f"Response saved to {args.output}")
            else:
                print(f"\nResponse: {response['content']}")
                if args.verbose:
                    print(f"Confidence: {response['confidence']:.2f}")
                    print(f"Processing time: {response['processing_time']:.2f}s")
        else:
            # Interactive mode
            self.interactive_chat(user_id, args.workflow, args.context)
    
    def cmd_memory_list(self, args):
        """Handle memory list command."""
        try:
            memory_type = MemoryType(args.type) if args.type else None
            memories = self.memory_repository.get_memories_by_user(
                args.user_id, 
                memory_type=memory_type,
                limit=args.limit
            )
            
            if args.format == 'json':
                output = [
                    {
                        'id': m.id,
                        'content': m.content,
                        'type': m.memory_type.value,
                        'confidence': m.confidence,
                        'created_at': m.created_at.isoformat()
                    }
                    for m in memories
                ]
                print(json.dumps(output, indent=2))
            else:
                # Table format
                print(f"{'ID':<36} {'Type':<10} {'Confidence':<10} {'Created':<20} {'Content':<50}")
                print("-" * 126)
                for memory in memories:
                    content = memory.content[:47] + "..." if len(memory.content) > 50 else memory.content
                    print(f"{memory.id:<36} {memory.memory_type.value:<10} {memory.confidence:<10.2f} "
                          f"{memory.created_at.strftime('%Y-%m-%d %H:%M'):<20} {content:<50}")
                
                print(f"\nTotal: {len(memories)} memories")
                
        except Exception as e:
            print(f"Failed to list memories: {e}")
            sys.exit(1)
    
    def cmd_memory_search(self, args):
        """Handle memory search command."""
        try:
            memories = self.memory_repository.search_memories(
                args.user_id,
                args.query,
                limit=args.limit
            )
            
            print(f"Search results for '{args.query}':")
            print(f"{'Score':<8} {'Type':<10} {'Content':<70}")
            print("-" * 88)
            
            for memory, score in memories:
                content = memory.content[:67] + "..." if len(memory.content) > 70 else memory.content
                print(f"{score:<8.2f} {memory.memory_type.value:<10} {content:<70}")
            
            print(f"\nFound {len(memories)} results")
            
        except Exception as e:
            print(f"Failed to search memories: {e}")
            sys.exit(1)
    
    def cmd_memory_delete(self, args):
        """Handle memory delete command."""
        try:
            if args.memory_id:
                # Delete specific memory
                if not args.confirm:
                    confirm = input(f"Delete memory {args.memory_id}? (y/N): ")
                    if confirm.lower() != 'y':
                        print("Cancelled")
                        return
                
                if self.memory_repository.delete_memory(args.memory_id, args.user_id, confirm=False):
                    print(f"Memory {args.memory_id} deleted")
                else:
                    print(f"Failed to delete memory {args.memory_id}")
            else:
                # Delete by type or all
                memory_type = MemoryType(args.type) if args.type else None
                type_str = args.type if args.type else "all"
                
                if not args.confirm:
                    confirm = input(f"Delete {type_str} memories for user {args.user_id}? (y/N): ")
                    if confirm.lower() != 'y':
                        print("Cancelled")
                        return
                
                count = self.memory_repository.delete_user_memories(
                    args.user_id,
                    memory_type=memory_type,
                    confirm=False
                )
                print(f"Deleted {count} memories")
                
        except Exception as e:
            print(f"Failed to delete memories: {e}")
            sys.exit(1)
    
    def cmd_memory_stats(self, args):
        """Handle memory stats command."""
        try:
            stats = self.memory_repository.get_memory_stats(args.user_id)
            
            print(f"Memory Statistics for User: {args.user_id}")
            print("-" * 40)
            print(f"Total Memories: {stats.get('total_memories', 0)}")
            
            if stats.get('by_type'):
                print("\nBy Type:")
                for memory_type, type_stats in stats['by_type'].items():
                    print(f"  {memory_type}: {type_stats['count']} "
                          f"(avg confidence: {type_stats['avg_confidence']:.2f})")
            
            if stats.get('recent_activity'):
                print(f"\nRecent Activity: {stats['recent_activity']} memories in last 7 days")
                
        except Exception as e:
            print(f"Failed to get memory stats: {e}")
            sys.exit(1)
    
    def cmd_user_list(self, args):
        """Handle user list command."""
        try:
            users = self.db_manager.execute_query("SELECT * FROM users ORDER BY created_at DESC")
            
            if args.format == 'json':
                output = [
                    {
                        'id': user['id'],
                        'name': user['name'],
                        'email': user['email'],
                        'created_at': user['created_at']
                    }
                    for user in (users or [])
                ]
                print(json.dumps(output, indent=2))
            else:
                print(f"{'ID':<36} {'Name':<20} {'Email':<30} {'Created':<20}")
                print("-" * 106)
                for user in (users or []):
                    print(f"{user['id']:<36} {user['name'] or 'N/A':<20} "
                          f"{user['email'] or 'N/A':<30} {user['created_at'] or 'N/A':<20}")
                
                print(f"\nTotal: {len(users or [])} users")
                
        except Exception as e:
            print(f"Failed to list users: {e}")
            sys.exit(1)
    
    def cmd_user_create(self, args):
        """Handle user create command."""
        try:
            user_id = args.user_id or str(uuid.uuid4())
            
            query = "INSERT INTO users (id, name, email, created_at) VALUES (?, ?, ?, ?)"
            self.db_manager.execute_update(query, (
                user_id,
                args.name or "",
                args.email or "",
                datetime.now().isoformat()
            ))
            
            print(f"User created: {user_id}")
            if args.name:
                print(f"Name: {args.name}")
            if args.email:
                print(f"Email: {args.email}")
                
        except Exception as e:
            print(f"Failed to create user: {e}")
            sys.exit(1)
    
    def cmd_user_show(self, args):
        """Handle user show command."""
        try:
            user = self.db_manager.execute_query("SELECT * FROM users WHERE id = ?", (args.user_id,))
            
            if not user:
                print(f"User not found: {args.user_id}")
                return
            
            user = user[0]
            print(f"User Details:")
            print(f"ID: {user['id']}")
            print(f"Name: {user['name'] or 'N/A'}")
            print(f"Email: {user['email'] or 'N/A'}")
            print(f"Created: {user['created_at'] or 'N/A'}")
            
            # Show memory stats
            try:
                stats = self.memory_repository.get_memory_stats(args.user_id)
                print(f"\nMemory Stats:")
                print(f"Total Memories: {stats.get('total_memories', 0)}")
                
                if stats.get('by_type'):
                    for memory_type, type_stats in stats['by_type'].items():
                        print(f"  {memory_type}: {type_stats['count']}")
            except:
                pass
                
        except Exception as e:
            print(f"Failed to show user: {e}")
            sys.exit(1)
    
    def cmd_user_delete(self, args):
        """Handle user delete command."""
        try:
            if not args.confirm:
                confirm = input(f"Delete user {args.user_id} and all associated data? (y/N): ")
                if confirm.lower() != 'y':
                    print("Cancelled")
                    return
            
            # Delete user memories first
            self.memory_repository.delete_user_memories(args.user_id, confirm=False)
            
            # Delete conversations
            self.db_manager.execute_update("DELETE FROM conversations WHERE user_id = ?", (args.user_id,))
            
            # Delete user
            result = self.db_manager.execute_update("DELETE FROM users WHERE id = ?", (args.user_id,))
            
            if result:
                print(f"User {args.user_id} deleted")
            else:
                print(f"User {args.user_id} not found")
                
        except Exception as e:
            print(f"Failed to delete user: {e}")
            sys.exit(1)
    
    def cmd_system_status(self, args):
        """Handle system status command."""
        try:
            print("FounderForge System Status")
            print("=" * 30)
            
            # Database status
            users_count = self.db_manager.execute_query("SELECT COUNT(*) as count FROM users")
            memories_count = self.db_manager.execute_query("SELECT COUNT(*) as count FROM memories")
            conversations_count = self.db_manager.execute_query("SELECT COUNT(*) as count FROM conversations")
            
            print(f"Database:")
            print(f"  Users: {users_count[0]['count'] if users_count else 0}")
            print(f"  Memories: {memories_count[0]['count'] if memories_count else 0}")
            print(f"  Conversations: {conversations_count[0]['count'] if conversations_count else 0}")
            
            # Component status
            print(f"\nComponents:")
            print(f"  Database: {'🟢 Connected' if self.db_manager else '🔴 Disconnected'}")
            print(f"  Memory Repository: {'🟢 Active' if self.memory_repository else '🔴 Inactive'}")
            print(f"  Context Manager: {'🟢 Active' if self.context_manager else '🔴 Inactive'}")
            print(f"  Agent Orchestrator: {'🟢 Active' if self.agent_orchestrator else '🔴 Inactive'}")
            
            try:
                gemini_status = "🟢 Connected" if self.gemini_client.is_available() else "🔴 Disconnected"
            except:
                gemini_status = "🔴 Disconnected"
            print(f"  Gemini Client: {gemini_status}")
            
            # Performance metrics
            try:
                avg_performance = self.memory_repository.get_avg_performance()
                print(f"\nPerformance:")
                print(f"  Avg Query Time: {avg_performance*1000:.2f}ms")
            except:
                pass
                
        except Exception as e:
            print(f"Failed to get system status: {e}")
            sys.exit(1)
    
    def cmd_system_database(self, args):
        """Handle system database command."""
        try:
            if args.init:
                if initialize_database():
                    print("Database initialized successfully")
                else:
                    print("Failed to initialize database")
                    sys.exit(1)
            
            elif args.backup:
                # Simple backup by copying database file
                import shutil
                db_path = Path("data/founderforge.db")
                backup_path = Path(args.backup)
                
                if db_path.exists():
                    shutil.copy2(db_path, backup_path)
                    print(f"Database backed up to {backup_path}")
                else:
                    print("Database file not found")
                    sys.exit(1)
            
            elif args.restore:
                # Simple restore by copying backup file
                import shutil
                backup_path = Path(args.restore)
                db_path = Path("data/founderforge.db")
                
                if backup_path.exists():
                    confirm = input(f"Restore database from {backup_path}? This will overwrite current data. (y/N): ")
                    if confirm.lower() == 'y':
                        shutil.copy2(backup_path, db_path)
                        print(f"Database restored from {backup_path}")
                    else:
                        print("Cancelled")
                else:
                    print(f"Backup file not found: {backup_path}")
                    sys.exit(1)
            else:
                print("No database operation specified")
                
        except Exception as e:
            print(f"Database operation failed: {e}")
            sys.exit(1)
    
    def cmd_system_tokens(self, args):
        """Handle system tokens command."""
        try:
            print(f"Token Usage Statistics ({args.days} days)")
            print("=" * 40)
            
            if args.user_id:
                # User-specific token usage
                daily_usage = self.token_manager.get_daily_usage(args.user_id)
                if daily_usage:
                    print(f"User {args.user_id}:")
                    print(f"  Total Tokens: {daily_usage.total_tokens}")
                    print(f"  Prompt Tokens: {daily_usage.prompt_tokens}")
                    print(f"  Completion Tokens: {daily_usage.completion_tokens}")
                else:
                    print(f"No token usage found for user {args.user_id}")
            else:
                # System-wide token usage
                session_usage = self.token_manager.get_session_usage()
                print(f"Session Usage:")
                print(f"  Total Tokens: {session_usage.total_tokens}")
                print(f"  Prompt Tokens: {session_usage.prompt_tokens}")
                print(f"  Completion Tokens: {session_usage.completion_tokens}")
                
        except Exception as e:
            print(f"Failed to get token statistics: {e}")
            sys.exit(1)
    
    def cmd_batch_process(self, args):
        """Handle batch process command."""
        try:
            with open(args.file, 'r') as f:
                batch_data = json.load(f)
            
            results = []
            total = len(batch_data.get('queries', []))
            
            print(f"Processing {total} queries...")
            
            for i, query_data in enumerate(batch_data.get('queries', []), 1):
                user_id = query_data.get('user_id', 'batch_user')
                message = query_data.get('message', '')
                workflow = query_data.get('workflow')
                
                print(f"Processing {i}/{total}: {message[:50]}...")
                
                try:
                    response = self.process_message(user_id, message, workflow)
                    results.append({
                        'query': query_data,
                        'response': response,
                        'success': True
                    })
                except Exception as e:
                    results.append({
                        'query': query_data,
                        'error': str(e),
                        'success': False
                    })
            
            # Save results
            output_file = args.output or f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w') as f:
                json.dump({
                    'processed_at': datetime.now().isoformat(),
                    'total_queries': total,
                    'successful': sum(1 for r in results if r['success']),
                    'failed': sum(1 for r in results if not r['success']),
                    'results': results
                }, f, indent=2)
            
            print(f"Batch processing complete. Results saved to {output_file}")
            
        except Exception as e:
            print(f"Batch processing failed: {e}")
            sys.exit(1)
    
    def cmd_batch_test(self, args):
        """Handle batch test command."""
        try:
            scenarios_file = args.scenarios or "test_scenarios.json"
            
            # Load test scenarios
            try:
                with open(scenarios_file, 'r') as f:
                    scenarios = json.load(f)
            except FileNotFoundError:
                # Create default test scenarios
                scenarios = self.create_default_test_scenarios()
                with open(scenarios_file, 'w') as f:
                    json.dump(scenarios, f, indent=2)
                print(f"Created default test scenarios in {scenarios_file}")
            
            results = []
            total = len(scenarios.get('scenarios', []))
            
            print(f"Running {total} test scenarios...")
            
            for i, scenario in enumerate(scenarios.get('scenarios', []), 1):
                name = scenario.get('name', f'Test {i}')
                print(f"Running {i}/{total}: {name}")
                
                try:
                    start_time = time.time()
                    response = self.process_message(
                        scenario.get('user_id', 'test_user'),
                        scenario.get('query', ''),
                        scenario.get('workflow')
                    )
                    processing_time = time.time() - start_time
                    
                    # Evaluate response
                    passed = self.evaluate_test_response(scenario, response)
                    
                    results.append({
                        'scenario': scenario,
                        'response': response,
                        'processing_time': processing_time,
                        'passed': passed,
                        'success': True
                    })
                    
                    status = "✅ PASS" if passed else "❌ FAIL"
                    print(f"  {status} ({processing_time:.2f}s)")
                    
                except Exception as e:
                    results.append({
                        'scenario': scenario,
                        'error': str(e),
                        'passed': False,
                        'success': False
                    })
                    print(f"  ❌ ERROR: {e}")
            
            # Generate test report
            passed = sum(1 for r in results if r.get('passed', False))
            total_successful = sum(1 for r in results if r['success'])
            
            print(f"\nTest Results:")
            print(f"  Total: {total}")
            print(f"  Passed: {passed}")
            print(f"  Failed: {total_successful - passed}")
            print(f"  Errors: {total - total_successful}")
            print(f"  Success Rate: {(passed/total)*100:.1f}%")
            
            # Save detailed results
            output_file = args.output or f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w') as f:
                json.dump({
                    'test_run_at': datetime.now().isoformat(),
                    'summary': {
                        'total': total,
                        'passed': passed,
                        'failed': total_successful - passed,
                        'errors': total - total_successful,
                        'success_rate': (passed/total)*100
                    },
                    'results': results
                }, f, indent=2)
            
            print(f"Detailed results saved to {output_file}")
            
        except Exception as e:
            print(f"Test execution failed: {e}")
            sys.exit(1)
    
    def cmd_config_show(self, args):
        """Handle config show command."""
        try:
            print("FounderForge Configuration")
            print("=" * 30)
            
            # Show settings from config
            print(f"Database Path: {getattr(settings, 'DATABASE_PATH', 'data/founderforge.db')}")
            print(f"Vector Store Path: {getattr(settings, 'VECTOR_STORE_PATH', 'data/vector_index')}")
            print(f"Log Level: {getattr(settings, 'LOG_LEVEL', 'INFO')}")
            
            # Show environment variables
            import os
            print(f"\nEnvironment:")
            print(f"GEMINI_API_KEY: {'Set' if os.getenv('GEMINI_API_KEY') else 'Not set'}")
            
        except Exception as e:
            print(f"Failed to show configuration: {e}")
            sys.exit(1)
    
    def cmd_config_set(self, args):
        """Handle config set command."""
        print(f"Setting {args.key} = {args.value}")
        print("Note: Configuration changes require application restart")
    
    # Helper methods
    
    def process_message(self, user_id: str, message: str, workflow: Optional[str] = None, 
                       context_file: Optional[str] = None) -> Dict[str, Any]:
        """Process a single message and return response."""
        start_time = time.time()
        
        try:
            # Load additional context if provided
            additional_context = {}
            if context_file:
                with open(context_file, 'r') as f:
                    additional_context = json.load(f)
            
            # Assemble context
            context = self.context_manager.assemble_context(user_id, message)
            
            # Add additional context
            if additional_context:
                context.additional_data.update(additional_context)
            
            # Determine processing method
            if workflow or self.should_use_agent_workflow(message):
                # Use agent workflow
                workflow_type = workflow or "general"
                workflow_result = asyncio.run(
                    self.agent_orchestrator.execute_workflow(
                        workflow_type=workflow_type,
                        user_id=user_id,
                        task_data={"user_query": message, "context": context.to_dict()}
                    )
                )
                
                response_content = workflow_result.result_data.get("final_response", "No response generated")
                confidence = workflow_result.confidence_score
                sources = ["agent_workflow"]
                fallback_used = not workflow_result.success
                
            else:
                # Direct Gemini API call
                prompt = self.build_conversation_prompt(context, message)
                
                gemini_response = self.gemini_client.generate_content(
                    prompt,
                    temperature=0.7,
                    max_output_tokens=20000
                )
                
                response_content = gemini_response.content
                confidence = gemini_response.confidence
                sources = context.sources
                fallback_used = False
            
            processing_time = time.time() - start_time
            
            # Log token usage
            token_usage = TokenUsage(total_tokens=len(message + response_content) // 4)
            self.token_manager.log_token_usage(user_id, token_usage, "cli_response")
            
            return {
                "content": response_content,
                "confidence": confidence,
                "processing_time": processing_time,
                "sources": sources,
                "fallback_used": fallback_used,
                "token_usage": token_usage.total_tokens
            }
            
        except Exception as e:
            logger.error(f"Message processing failed: {e}")
            return {
                "content": "I apologize, but I encountered an error processing your request.",
                "confidence": 0.0,
                "processing_time": time.time() - start_time,
                "sources": [],
                "fallback_used": True,
                "error": str(e)
            }
    
    def interactive_chat(self, user_id: str, workflow: Optional[str] = None, 
                        context_file: Optional[str] = None):
        """Start interactive chat session."""
        print(f"FounderForge Interactive Chat (User: {user_id})")
        print("Type 'quit', 'exit', or press Ctrl+C to end the session")
        print("-" * 50)
        
        while True:
            try:
                message = input("\nYou: ").strip()
                
                if message.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not message:
                    continue
                
                print("AI: ", end="", flush=True)
                response = self.process_message(user_id, message, workflow, context_file)
                print(response['content'])
                
                if response.get('confidence', 0) < 0.5:
                    print("⚠️  Low confidence response")
                
            except KeyboardInterrupt:
                break
            except EOFError:
                break
        
        print("\nChat session ended")
    
    def should_use_agent_workflow(self, message: str) -> bool:
        """Determine if message requires agent workflow processing."""
        workflow_keywords = [
            "plan", "strategy", "funding", "analyze", "evaluate",
            "workflow", "process", "validate", "review"
        ]
        
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in workflow_keywords)
    
    def build_conversation_prompt(self, context, user_message: str) -> str:
        """Build prompt for conversation with context."""
        context_str = context.to_prompt_string()
        
        prompt = f"""
{context_str}

Current User Message: {user_message}

Please provide a helpful, accurate response based on the context above.
Focus on actionable insights and practical guidance for the user's business needs.
"""
        
        return prompt
    
    def create_default_test_scenarios(self) -> Dict[str, Any]:
        """Create default test scenarios for evaluation."""
        return {
            "scenarios": [
                {
                    "name": "Basic greeting",
                    "user_id": "test_user",
                    "query": "Hello, how can you help me?",
                    "expected_keywords": ["help", "assist", "founder"],
                    "min_confidence": 0.8
                },
                {
                    "name": "Business strategy question",
                    "user_id": "test_user", 
                    "query": "What should be my next steps for growing my startup?",
                    "workflow": "planning",
                    "expected_keywords": ["growth", "strategy", "steps"],
                    "min_confidence": 0.7
                },
                {
                    "name": "Funding guidance",
                    "user_id": "test_user",
                    "query": "How do I prepare for Series A funding?",
                    "workflow": "funding",
                    "expected_keywords": ["funding", "series", "prepare"],
                    "min_confidence": 0.7
                }
            ]
        }
    
    def evaluate_test_response(self, scenario: Dict[str, Any], response: Dict[str, Any]) -> bool:
        """Evaluate if test response meets scenario criteria."""
        try:
            # Check minimum confidence
            min_confidence = scenario.get('min_confidence', 0.5)
            if response.get('confidence', 0) < min_confidence:
                return False
            
            # Check for expected keywords
            expected_keywords = scenario.get('expected_keywords', [])
            response_content = response.get('content', '').lower()
            
            for keyword in expected_keywords:
                if keyword.lower() not in response_content:
                    return False
            
            # Check processing time (should be under 30 seconds)
            if response.get('processing_time', 0) > 30:
                return False
            
            return True
            
        except Exception:
            return False


def main():
    """Main CLI entry point."""
    cli = FounderForgeCLI()
    cli.run()


if __name__ == "__main__":
    main()