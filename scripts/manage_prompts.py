#!/usr/bin/env python3
"""
Script for managing prompts and experiments using Git integration.

This script provides a command-line interface for:
- Creating and managing experiments
- Versioning prompts
- Rolling back changes
- Analyzing experiment results
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from git_prompt_manager import GitPromptManager, ExperimentType, PromptChange

def create_experiment(args):
    """Create a new experiment."""
    manager = GitPromptManager(Path("data/prompts"))
    
    # Parse experiment type
    exp_type = ExperimentType.T1_ONE_WAY if args.type.upper() == "T1" else ExperimentType.T2_REVERSIBLE
    
    try:
        branch_name = manager.create_experiment_branch(
            experiment_id=args.id,
            experiment_type=exp_type,
            description=args.description,
            created_by=args.author or "system"
        )
        print(f"✅ Created experiment '{args.id}' on branch '{branch_name}'")
        print(f"   Type: {exp_type.value}")
        print(f"   Description: {args.description}")
        
    except Exception as e:
        print(f"❌ Error creating experiment: {e}")
        sys.exit(1)

def list_experiments(args):
    """List all experiments."""
    manager = GitPromptManager(Path("data/prompts"))
    
    experiments = manager.get_experiment_history()
    
    if not experiments:
        print("No experiments found.")
        return
    
    print(f"{'ID':<20} {'Type':<4} {'Status':<12} {'Created':<20} {'Description'}")
    print("-" * 80)
    
    for exp in experiments:
        created_date = datetime.fromisoformat(exp.created_at).strftime("%Y-%m-%d %H:%M")
        print(f"{exp.experiment_id:<20} {exp.experiment_type.value:<4} {exp.status:<12} {created_date:<20} {exp.description[:30]}")

def update_prompt(args):
    """Update a prompt in the current experiment."""
    manager = GitPromptManager(Path("data/prompts"))
    
    # Read prompt content
    if args.file:
        with open(args.file, 'r') as f:
            new_content = f.read()
    else:
        new_content = args.content
    
    # Get old content for change tracking
    old_content = manager.get_prompt_content(args.prompt_id) or ""
    
    # Save the prompt
    success = manager.save_prompt(args.prompt_id, new_content, args.description)
    
    if success:
        print(f"✅ Updated prompt '{args.prompt_id}'")
        if args.description:
            print(f"   Description: {args.description}")
    else:
        print(f"❌ Failed to update prompt '{args.prompt_id}'")
        sys.exit(1)

def commit_changes(args):
    """Commit prompt changes to an experiment."""
    manager = GitPromptManager(Path("data/prompts"))
    
    # Create prompt changes from arguments
    changes = []
    for change_spec in args.changes:
        parts = change_spec.split(":")
        if len(parts) != 3:
            print(f"❌ Invalid change format: {change_spec}")
            print("   Expected format: prompt_id:old_content:new_content")
            sys.exit(1)
        
        prompt_id, old_content, new_content = parts
        changes.append(PromptChange(
            prompt_id=prompt_id,
            old_content=old_content,
            new_content=new_content,
            change_type="manual",
            description=args.description or f"Update {prompt_id}",
            timestamp=datetime.now().isoformat()
        ))
    
    try:
        commit_hash = manager.commit_prompt_changes(args.experiment_id, changes)
        print(f"✅ Committed {len(changes)} changes to experiment '{args.experiment_id}'")
        print(f"   Commit hash: {commit_hash}")
        
    except Exception as e:
        print(f"❌ Error committing changes: {e}")
        sys.exit(1)

def rollback_experiment(args):
    """Rollback an experiment."""
    manager = GitPromptManager(Path("data/prompts"))
    
    if not args.force:
        response = input(f"Are you sure you want to rollback experiment '{args.experiment_id}'? (y/N): ")
        if response.lower() != 'y':
            print("Rollback cancelled.")
            return
    
    try:
        success = manager.rollback_experiment(args.experiment_id)
        if success:
            print(f"✅ Rolled back experiment '{args.experiment_id}'")
        else:
            print(f"❌ Failed to rollback experiment '{args.experiment_id}'")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error rolling back experiment: {e}")
        sys.exit(1)

def complete_experiment(args):
    """Complete an experiment with metrics."""
    manager = GitPromptManager(Path("data/prompts"))
    
    # Parse metrics from JSON string or file
    metrics = {}
    if args.metrics:
        try:
            metrics = json.loads(args.metrics)
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in metrics: {e}")
            sys.exit(1)
    elif args.metrics_file:
        try:
            with open(args.metrics_file, 'r') as f:
                metrics = json.load(f)
        except Exception as e:
            print(f"❌ Error reading metrics file: {e}")
            sys.exit(1)
    
    try:
        success = manager.complete_experiment(args.experiment_id, metrics)
        if success:
            print(f"✅ Completed experiment '{args.experiment_id}'")
            if metrics:
                print("   Metrics:")
                for key, value in metrics.items():
                    print(f"     {key}: {value}")
        else:
            print(f"❌ Failed to complete experiment '{args.experiment_id}'")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error completing experiment: {e}")
        sys.exit(1)

def merge_experiment(args):
    """Merge an experiment to main branch."""
    manager = GitPromptManager(Path("data/prompts"))
    
    if not args.force:
        response = input(f"Are you sure you want to merge experiment '{args.experiment_id}' to main? (y/N): ")
        if response.lower() != 'y':
            print("Merge cancelled.")
            return
    
    try:
        success = manager.merge_experiment_to_main(args.experiment_id)
        if success:
            print(f"✅ Merged experiment '{args.experiment_id}' to main branch")
        else:
            print(f"❌ Failed to merge experiment '{args.experiment_id}'")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error merging experiment: {e}")
        sys.exit(1)

def list_prompts(args):
    """List all available prompts."""
    manager = GitPromptManager(Path("data/prompts"))
    
    prompts = manager.list_prompts()
    
    if not prompts:
        print("No prompts found.")
        return
    
    print("Available prompts:")
    for prompt_id in sorted(prompts):
        print(f"  - {prompt_id}")

def show_prompt(args):
    """Show the content of a prompt."""
    manager = GitPromptManager(Path("data/prompts"))
    
    content = manager.get_prompt_content(args.prompt_id)
    
    if content is None:
        print(f"❌ Prompt '{args.prompt_id}' not found")
        sys.exit(1)
    
    print(f"=== Prompt: {args.prompt_id} ===")
    print(content)

def status(args):
    """Show current Git status and active experiments."""
    manager = GitPromptManager(Path("data/prompts"))
    
    current_branch = manager.get_current_branch()
    print(f"Current branch: {current_branch}")
    
    # Show active experiments
    experiments = manager.get_experiment_history()
    active_experiments = [exp for exp in experiments if exp.status == "active"]
    
    if active_experiments:
        print(f"\nActive experiments ({len(active_experiments)}):")
        for exp in active_experiments:
            print(f"  - {exp.experiment_id} ({exp.experiment_type.value}): {exp.description}")
    else:
        print("\nNo active experiments.")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Manage prompts and experiments")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Create experiment
    create_parser = subparsers.add_parser("create", help="Create a new experiment")
    create_parser.add_argument("id", help="Experiment ID")
    create_parser.add_argument("type", choices=["T1", "T2"], help="Experiment type (T1=one-way, T2=reversible)")
    create_parser.add_argument("description", help="Experiment description")
    create_parser.add_argument("--author", help="Experiment author")
    create_parser.set_defaults(func=create_experiment)
    
    # List experiments
    list_parser = subparsers.add_parser("list", help="List all experiments")
    list_parser.set_defaults(func=list_experiments)
    
    # Update prompt
    update_parser = subparsers.add_parser("update", help="Update a prompt")
    update_parser.add_argument("prompt_id", help="Prompt ID")
    update_parser.add_argument("--content", help="Prompt content")
    update_parser.add_argument("--file", help="File containing prompt content")
    update_parser.add_argument("--description", help="Change description")
    update_parser.set_defaults(func=update_prompt)
    
    # Commit changes
    commit_parser = subparsers.add_parser("commit", help="Commit prompt changes")
    commit_parser.add_argument("experiment_id", help="Experiment ID")
    commit_parser.add_argument("changes", nargs="+", help="Changes in format prompt_id:old:new")
    commit_parser.add_argument("--description", help="Commit description")
    commit_parser.set_defaults(func=commit_changes)
    
    # Rollback experiment
    rollback_parser = subparsers.add_parser("rollback", help="Rollback an experiment")
    rollback_parser.add_argument("experiment_id", help="Experiment ID")
    rollback_parser.add_argument("--force", action="store_true", help="Skip confirmation")
    rollback_parser.set_defaults(func=rollback_experiment)
    
    # Complete experiment
    complete_parser = subparsers.add_parser("complete", help="Complete an experiment")
    complete_parser.add_argument("experiment_id", help="Experiment ID")
    complete_parser.add_argument("--metrics", help="Metrics as JSON string")
    complete_parser.add_argument("--metrics-file", help="File containing metrics JSON")
    complete_parser.set_defaults(func=complete_experiment)
    
    # Merge experiment
    merge_parser = subparsers.add_parser("merge", help="Merge experiment to main")
    merge_parser.add_argument("experiment_id", help="Experiment ID")
    merge_parser.add_argument("--force", action="store_true", help="Skip confirmation")
    merge_parser.set_defaults(func=merge_experiment)
    
    # List prompts
    prompts_parser = subparsers.add_parser("prompts", help="List all prompts")
    prompts_parser.set_defaults(func=list_prompts)
    
    # Show prompt
    show_parser = subparsers.add_parser("show", help="Show prompt content")
    show_parser.add_argument("prompt_id", help="Prompt ID")
    show_parser.set_defaults(func=show_prompt)
    
    # Status
    status_parser = subparsers.add_parser("status", help="Show current status")
    status_parser.set_defaults(func=status)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)

if __name__ == "__main__":
    main()