"""Git integration for prompt versioning and experiment management."""

import os
import json
import logging
from .logging_manager import get_logging_manager, LogLevel, LogCategory
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

class ExperimentType(Enum):
    """Types of experiments for classification."""
    T1_ONE_WAY = "T1"  # One-way door changes (hard to reverse)
    T2_REVERSIBLE = "T2"  # Two-way door changes (easy to reverse)

@dataclass
class PromptChange:
    """Represents a change to a prompt."""
    prompt_id: str
    old_content: str
    new_content: str
    change_type: str
    description: str
    timestamp: str

@dataclass
class ExperimentMetadata:
    """Metadata for an experiment."""
    experiment_id: str
    experiment_type: ExperimentType
    description: str
    created_at: str
    created_by: str
    branch_name: str
    prompt_changes: List[PromptChange]
    metrics: Dict[str, Any]
    status: str  # "active", "completed", "rolled_back"

class GitPromptManager:
    """Manages prompt versioning and experiments using Git."""
    
    def __init__(self, prompts_dir: Path):
        self.prompts_dir = Path(prompts_dir)
        self.prompts_dir.mkdir(exist_ok=True)
        
        # Git repository path (same as prompts directory)
        self.repo_path = self.prompts_dir
        
        # Experiments metadata file
        self.experiments_file = self.prompts_dir / "experiments.json"
        
        # Initialize logging
        self.logger = get_logging_manager().get_logger(__name__.split(".")[-1], LogCategory.SYSTEM)
        
        # Initialize Git repository if it doesn't exist
        self._ensure_git_repository()
    
    def _ensure_git_repository(self) -> None:
        """Ensure Git repository is initialized."""
        git_dir = self.repo_path / ".git"
        
        if not git_dir.exists():
            try:
                # Initialize Git repository
                self._run_git_command(["init"])
                
                # Create initial commit with README
                readme_path = self.repo_path / "README.md"
                readme_content = """# FounderForge Prompt Repository

This repository contains versioned prompts and experiment tracking for the FounderForge AI Cofounder system.

## Structure

- `prompts/` - Individual prompt files
- `experiments.json` - Experiment metadata and tracking
- `templates/` - Prompt templates
- `archived/` - Archived experiments and prompts

## Experiment Types

- **T1 (One-way door)**: Changes that are difficult to reverse
- **T2 (Reversible)**: Changes that can be easily rolled back

## Usage

Use the GitPromptManager to manage prompt versions and experiments programmatically.
"""
                
                with open(readme_path, 'w') as f:
                    f.write(readme_content)
                
                # Create directory structure
                (self.repo_path / "prompts").mkdir(exist_ok=True)
                (self.repo_path / "templates").mkdir(exist_ok=True)
                (self.repo_path / "archived").mkdir(exist_ok=True)
                
                # Add and commit initial files
                self._run_git_command(["add", "."])
                self._run_git_command(["commit", "-m", "Initial commit: Setup prompt repository"])
                
                # Rename master to main if needed
                try:
                    current_branch = self._run_git_command(["branch", "--show-current"], check_output=True)
                    if current_branch == "master":
                        self._run_git_command(["branch", "-m", "master", "main"])
                except Exception:
                    # If branch renaming fails, continue with master
                    pass
                
                self.logger.info("Initialized Git repository for prompt management")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize Git repository: {e}")
                raise
    
    def _run_git_command(self, args: List[str], check_output: bool = False) -> Optional[str]:
        """Run a Git command in the repository directory."""
        try:
            cmd = ["git"] + args
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            if check_output:
                return result.stdout.strip()
            
            return None
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Git command failed: {' '.join(cmd)}")
            self.logger.error(f"Error: {e.stderr}")
            raise
    
    def _load_experiments(self) -> Dict[str, ExperimentMetadata]:
        """Load experiments metadata from file."""
        if not self.experiments_file.exists():
            return {}
        
        try:
            with open(self.experiments_file, 'r') as f:
                data = json.load(f)
            
            experiments = {}
            for exp_id, exp_data in data.items():
                # Convert prompt changes
                prompt_changes = []
                for change_data in exp_data.get('prompt_changes', []):
                    prompt_changes.append(PromptChange(**change_data))
                
                # Create experiment metadata
                exp_data['prompt_changes'] = prompt_changes
                exp_data['experiment_type'] = ExperimentType(exp_data['experiment_type'])
                experiments[exp_id] = ExperimentMetadata(**exp_data)
            
            return experiments
            
        except Exception as e:
            self.logger.error(f"Error loading experiments: {e}")
            return {}
    
    def _save_experiments(self, experiments: Dict[str, ExperimentMetadata]) -> None:
        """Save experiments metadata to file."""
        try:
            data = {}
            for exp_id, experiment in experiments.items():
                exp_dict = asdict(experiment)
                exp_dict['experiment_type'] = experiment.experiment_type.value
                data[exp_id] = exp_dict
            
            with open(self.experiments_file, 'w') as f:
                json.dump(data, f, indent=2, sort_keys=True)
                
        except Exception as e:
            self.logger.error(f"Error saving experiments: {e}")
            raise
    
    def create_experiment_branch(self, experiment_id: str, experiment_type: ExperimentType, 
                               description: str, created_by: str = "system") -> str:
        """Create a new Git branch for an experiment."""
        try:
            # Generate branch name
            branch_name = f"experiment/{experiment_type.value.lower()}/{experiment_id}"
            
            # Create and checkout new branch
            self._run_git_command(["checkout", "-b", branch_name])
            
            # Create experiment metadata
            experiment = ExperimentMetadata(
                experiment_id=experiment_id,
                experiment_type=experiment_type,
                description=description,
                created_at=datetime.now().isoformat(),
                created_by=created_by,
                branch_name=branch_name,
                prompt_changes=[],
                metrics={},
                status="active"
            )
            
            # Save experiment metadata
            experiments = self._load_experiments()
            experiments[experiment_id] = experiment
            self._save_experiments(experiments)
            
            # Commit experiment metadata
            self._run_git_command(["add", "experiments.json"])
            self._run_git_command(["commit", "-m", f"Create experiment {experiment_id}: {description}"])
            
            self.logger.info(f"Created experiment branch: {branch_name}")
            return branch_name
            
        except Exception as e:
            self.logger.error(f"Error creating experiment branch: {e}")
            raise
    
    def commit_prompt_changes(self, experiment_id: str, changes: List[PromptChange]) -> str:
        """Commit prompt changes to the current experiment branch."""
        try:
            # Load experiments
            experiments = self._load_experiments()
            
            if experiment_id not in experiments:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            experiment = experiments[experiment_id]
            
            # Apply changes to files
            for change in changes:
                prompt_file = self.repo_path / "prompts" / f"{change.prompt_id}.txt"
                prompt_file.parent.mkdir(exist_ok=True)
                
                with open(prompt_file, 'w') as f:
                    f.write(change.new_content)
            
            # Update experiment metadata
            experiment.prompt_changes.extend(changes)
            experiments[experiment_id] = experiment
            self._save_experiments(experiments)
            
            # Commit changes
            self._run_git_command(["add", "."])
            
            commit_message = f"Update prompts for experiment {experiment_id}\n\n"
            for change in changes:
                commit_message += f"- {change.prompt_id}: {change.description}\n"
            
            self._run_git_command(["commit", "-m", commit_message])
            
            # Get commit hash
            commit_hash = self._run_git_command(["rev-parse", "HEAD"], check_output=True)
            
            self.logger.info(f"Committed {len(changes)} prompt changes for experiment {experiment_id}")
            return commit_hash
            
        except Exception as e:
            self.logger.error(f"Error committing prompt changes: {e}")
            raise
    
    def rollback_experiment(self, experiment_id: str) -> bool:
        """Rollback an experiment by switching to main branch and marking as rolled back."""
        try:
            # Load experiments
            experiments = self._load_experiments()
            
            if experiment_id not in experiments:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            experiment = experiments[experiment_id]
            
            # Try to switch to main branch, fallback to master
            try:
                self._run_git_command(["checkout", "main"])
            except subprocess.CalledProcessError:
                try:
                    self._run_git_command(["checkout", "master"])
                except subprocess.CalledProcessError:
                    # If both fail, stay on current branch but mark as rolled back
                    pass
            
            # Update experiment status
            experiment.status = "rolled_back"
            experiments[experiment_id] = experiment
            self._save_experiments(experiments)
            
            # Commit status update
            self._run_git_command(["add", "experiments.json"])
            self._run_git_command(["commit", "-m", f"Rollback experiment {experiment_id}"])
            
            self.logger.info(f"Rolled back experiment {experiment_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error rolling back experiment: {e}")
            return False
    
    def complete_experiment(self, experiment_id: str, metrics: Dict[str, Any]) -> bool:
        """Mark an experiment as completed and optionally merge to main."""
        try:
            # Load experiments
            experiments = self._load_experiments()
            
            if experiment_id not in experiments:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            experiment = experiments[experiment_id]
            
            # Update experiment with metrics and status
            experiment.metrics = metrics
            experiment.status = "completed"
            experiments[experiment_id] = experiment
            self._save_experiments(experiments)
            
            # Commit final status
            self._run_git_command(["add", "experiments.json"])
            self._run_git_command(["commit", "-m", f"Complete experiment {experiment_id}"])
            
            self.logger.info(f"Completed experiment {experiment_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error completing experiment: {e}")
            return False
    
    def merge_experiment_to_main(self, experiment_id: str) -> bool:
        """Merge a completed experiment to the main branch."""
        try:
            # Load experiments
            experiments = self._load_experiments()
            
            if experiment_id not in experiments:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            experiment = experiments[experiment_id]
            
            if experiment.status != "completed":
                raise ValueError(f"Experiment {experiment_id} is not completed")
            
            # Try to switch to main branch, fallback to master
            try:
                self._run_git_command(["checkout", "main"])
            except subprocess.CalledProcessError:
                self._run_git_command(["checkout", "master"])
            
            # Merge experiment branch
            self._run_git_command(["merge", experiment.branch_name, "--no-ff"])
            
            self.logger.info(f"Merged experiment {experiment_id} to main branch")
            return True
            
        except Exception as e:
            self.logger.error(f"Error merging experiment to main: {e}")
            return False
    
    def get_experiment_history(self, experiment_id: Optional[str] = None) -> List[ExperimentMetadata]:
        """Get experiment history, optionally filtered by experiment ID."""
        experiments = self._load_experiments()
        
        if experiment_id:
            return [experiments[experiment_id]] if experiment_id in experiments else []
        
        # Sort by creation date (most recent first)
        return sorted(experiments.values(), key=lambda x: x.created_at, reverse=True)
    
    def get_current_branch(self) -> str:
        """Get the current Git branch name."""
        try:
            return self._run_git_command(["branch", "--show-current"], check_output=True)
        except Exception as e:
            self.logger.error(f"Error getting current branch: {e}")
            return "unknown"
    
    def get_prompt_content(self, prompt_id: str) -> Optional[str]:
        """Get the current content of a prompt."""
        prompt_file = self.repo_path / "prompts" / f"{prompt_id}.txt"
        
        if not prompt_file.exists():
            return None
        
        try:
            with open(prompt_file, 'r') as f:
                return f.read()
        except Exception as e:
            self.logger.error(f"Error reading prompt {prompt_id}: {e}")
            return None
    
    def save_prompt(self, prompt_id: str, content: str, description: str = "") -> bool:
        """Save a prompt to the repository."""
        try:
            prompt_file = self.repo_path / "prompts" / f"{prompt_id}.txt"
            prompt_file.parent.mkdir(exist_ok=True)
            
            with open(prompt_file, 'w') as f:
                f.write(content)
            
            # Add and commit if on a branch
            current_branch = self.get_current_branch()
            if current_branch not in ["main", "master"]:
                # Use relative path from repo directory
                relative_path = prompt_file.relative_to(self.repo_path)
                self._run_git_command(["add", str(relative_path)])
                commit_msg = f"Update prompt {prompt_id}"
                if description:
                    commit_msg += f": {description}"
                self._run_git_command(["commit", "-m", commit_msg])
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving prompt {prompt_id}: {e}")
            return False
    
    def list_prompts(self) -> List[str]:
        """List all available prompt IDs."""
        prompts_dir = self.repo_path / "prompts"
        
        if not prompts_dir.exists():
            return []
        
        prompt_files = prompts_dir.glob("*.txt")
        return [f.stem for f in prompt_files]
    
    def archive_experiment(self, experiment_id: str) -> bool:
        """Archive an experiment by moving it to the archived directory."""
        try:
            experiments = self._load_experiments()
            
            if experiment_id not in experiments:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            experiment = experiments[experiment_id]
            
            # Create archive directory
            archive_dir = self.repo_path / "archived" / experiment_id
            archive_dir.mkdir(parents=True, exist_ok=True)
            
            # Save experiment metadata to archive
            with open(archive_dir / "metadata.json", 'w') as f:
                json.dump(asdict(experiment), f, indent=2, default=str)
            
            # Copy prompt files to archive
            for change in experiment.prompt_changes:
                prompt_file = self.repo_path / "prompts" / f"{change.prompt_id}.txt"
                if prompt_file.exists():
                    archive_file = archive_dir / f"{change.prompt_id}.txt"
                    archive_file.write_text(prompt_file.read_text())
            
            # Remove from active experiments
            del experiments[experiment_id]
            self._save_experiments(experiments)
            
            # Commit archive
            self._run_git_command(["add", "."])
            self._run_git_command(["commit", "-m", f"Archive experiment {experiment_id}"])
            
            self.logger.info(f"Archived experiment {experiment_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error archiving experiment: {e}")
            return False

# Global Git prompt manager instance
_git_prompt_manager: Optional[GitPromptManager] = None

def get_git_prompt_manager() -> GitPromptManager:
    """Get the global Git prompt manager instance."""
    global _git_prompt_manager
    if _git_prompt_manager is None:
        from pathlib import Path
        project_root = Path(__file__).parent.parent
        prompts_dir = project_root / "data" / "prompts"
        _git_prompt_manager = GitPromptManager(prompts_dir)
    return _git_prompt_manager