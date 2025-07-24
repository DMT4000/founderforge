"""Tests for Git prompt manager functionality."""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

from src.git_prompt_manager import GitPromptManager, ExperimentType, PromptChange


class TestGitPromptManager:
    """Test cases for GitPromptManager."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        # Handle Windows permission issues
        try:
            shutil.rmtree(temp_dir)
        except PermissionError:
            # On Windows, Git files might be locked, try to handle gracefully
            import time
            time.sleep(0.1)
            try:
                shutil.rmtree(temp_dir)
            except PermissionError:
                # If still failing, just pass - temp files will be cleaned up eventually
                pass
    
    @pytest.fixture
    def manager(self, temp_dir):
        """Create a GitPromptManager instance for testing."""
        return GitPromptManager(temp_dir / "prompts")
    
    def test_initialization(self, manager, temp_dir):
        """Test that Git repository is properly initialized."""
        prompts_dir = temp_dir / "prompts"
        
        # Check that directories are created
        assert prompts_dir.exists()
        assert (prompts_dir / ".git").exists()
        assert (prompts_dir / "prompts").exists()
        assert (prompts_dir / "templates").exists()
        assert (prompts_dir / "archived").exists()
        
        # Check that README exists
        assert (prompts_dir / "README.md").exists()
        
        # Check that we're on main or master branch
        current_branch = manager.get_current_branch()
        assert current_branch in ["main", "master"]
    
    def test_create_experiment_branch(self, manager):
        """Test creating experiment branches."""
        # Test T1 experiment
        branch_name = manager.create_experiment_branch(
            experiment_id="test-t1-exp",
            experiment_type=ExperimentType.T1_ONE_WAY,
            description="Test T1 experiment",
            created_by="test_user"
        )
        
        assert branch_name == "experiment/t1/test-t1-exp"
        assert manager.get_current_branch() == branch_name
        
        # Check experiment metadata
        experiments = manager.get_experiment_history()
        assert len(experiments) == 1
        
        exp = experiments[0]
        assert exp.experiment_id == "test-t1-exp"
        assert exp.experiment_type == ExperimentType.T1_ONE_WAY
        assert exp.description == "Test T1 experiment"
        assert exp.created_by == "test_user"
        assert exp.status == "active"
        
        # Switch back to main/master and create T2 experiment
        try:
            manager._run_git_command(["checkout", "main"])
        except:
            manager._run_git_command(["checkout", "master"])
        
        branch_name_t2 = manager.create_experiment_branch(
            experiment_id="test-t2-exp",
            experiment_type=ExperimentType.T2_REVERSIBLE,
            description="Test T2 experiment"
        )
        
        assert branch_name_t2 == "experiment/t2/test-t2-exp"
        assert manager.get_current_branch() == branch_name_t2
    
    def test_save_and_get_prompt(self, manager):
        """Test saving and retrieving prompts."""
        # Create an experiment first
        manager.create_experiment_branch(
            experiment_id="prompt-test",
            experiment_type=ExperimentType.T2_REVERSIBLE,
            description="Test prompt operations"
        )
        
        # Save a prompt
        prompt_content = "You are a helpful AI assistant. Please help the user with their request."
        success = manager.save_prompt("system_prompt", prompt_content, "Initial system prompt")
        assert success
        
        # Retrieve the prompt
        retrieved_content = manager.get_prompt_content("system_prompt")
        assert retrieved_content == prompt_content
        
        # Test non-existent prompt
        assert manager.get_prompt_content("non_existent") is None
    
    def test_commit_prompt_changes(self, manager):
        """Test committing prompt changes to experiments."""
        # Create experiment
        manager.create_experiment_branch(
            experiment_id="commit-test",
            experiment_type=ExperimentType.T2_REVERSIBLE,
            description="Test commit functionality"
        )
        
        # Create prompt changes
        changes = [
            PromptChange(
                prompt_id="system_prompt",
                old_content="Old system prompt",
                new_content="New system prompt with improvements",
                change_type="enhancement",
                description="Improved system prompt clarity",
                timestamp=datetime.now().isoformat()
            ),
            PromptChange(
                prompt_id="user_prompt",
                old_content="",
                new_content="Please be concise in your responses.",
                change_type="new",
                description="Added user prompt template",
                timestamp=datetime.now().isoformat()
            )
        ]
        
        # Commit changes
        commit_hash = manager.commit_prompt_changes("commit-test", changes)
        assert commit_hash is not None
        assert len(commit_hash) > 0
        
        # Verify changes are reflected in experiment metadata
        experiments = manager.get_experiment_history("commit-test")
        assert len(experiments) == 1
        
        exp = experiments[0]
        assert len(exp.prompt_changes) == 2
        assert exp.prompt_changes[0].prompt_id == "system_prompt"
        assert exp.prompt_changes[1].prompt_id == "user_prompt"
        
        # Verify prompt files exist
        assert manager.get_prompt_content("system_prompt") == "New system prompt with improvements"
        assert manager.get_prompt_content("user_prompt") == "Please be concise in your responses."
    
    def test_rollback_experiment(self, manager):
        """Test rolling back experiments."""
        # Create and work on experiment
        manager.create_experiment_branch(
            experiment_id="rollback-test",
            experiment_type=ExperimentType.T2_REVERSIBLE,
            description="Test rollback functionality"
        )
        
        # Make some changes
        manager.save_prompt("test_prompt", "Test content", "Test change")
        
        # Rollback experiment
        success = manager.rollback_experiment("rollback-test")
        assert success
        
        # Verify we're back on main/master branch
        current_branch = manager.get_current_branch()
        assert current_branch in ["main", "master"]
        
        # Verify experiment status is updated
        experiments = manager.get_experiment_history("rollback-test")
        assert len(experiments) == 1
        assert experiments[0].status == "rolled_back"
    
    def test_complete_experiment(self, manager):
        """Test completing experiments with metrics."""
        # Create experiment
        manager.create_experiment_branch(
            experiment_id="complete-test",
            experiment_type=ExperimentType.T1_ONE_WAY,
            description="Test completion functionality"
        )
        
        # Complete with metrics
        metrics = {
            "accuracy": 0.95,
            "response_time": 1.2,
            "user_satisfaction": 4.5
        }
        
        success = manager.complete_experiment("complete-test", metrics)
        assert success
        
        # Verify experiment status and metrics
        experiments = manager.get_experiment_history("complete-test")
        assert len(experiments) == 1
        
        exp = experiments[0]
        assert exp.status == "completed"
        assert exp.metrics == metrics
    
    def test_merge_experiment_to_main(self, manager):
        """Test merging completed experiments to main branch."""
        # Create and complete experiment
        manager.create_experiment_branch(
            experiment_id="merge-test",
            experiment_type=ExperimentType.T2_REVERSIBLE,
            description="Test merge functionality"
        )
        
        # Add some content
        manager.save_prompt("merge_prompt", "Merged content", "Content for merge test")
        
        # Complete experiment
        manager.complete_experiment("merge-test", {"test_metric": 1.0})
        
        # Merge to main
        success = manager.merge_experiment_to_main("merge-test")
        assert success
        
        # Verify we're on main/master branch
        current_branch = manager.get_current_branch()
        assert current_branch in ["main", "master"]
        
        # Verify content is available on main
        assert manager.get_prompt_content("merge_prompt") == "Merged content"
    
    def test_list_prompts(self, manager):
        """Test listing available prompts."""
        # Initially should be empty
        prompts = manager.list_prompts()
        assert prompts == []
        
        # Create experiment and add prompts
        manager.create_experiment_branch(
            experiment_id="list-test",
            experiment_type=ExperimentType.T2_REVERSIBLE,
            description="Test prompt listing"
        )
        
        manager.save_prompt("prompt1", "Content 1")
        manager.save_prompt("prompt2", "Content 2")
        manager.save_prompt("prompt3", "Content 3")
        
        # List prompts
        prompts = manager.list_prompts()
        assert set(prompts) == {"prompt1", "prompt2", "prompt3"}
    
    def test_archive_experiment(self, manager):
        """Test archiving experiments."""
        # Create and work on experiment
        manager.create_experiment_branch(
            experiment_id="archive-test",
            experiment_type=ExperimentType.T1_ONE_WAY,
            description="Test archive functionality"
        )
        
        # Add content and commit changes
        changes = [
            PromptChange(
                prompt_id="archived_prompt",
                old_content="",
                new_content="This will be archived",
                change_type="new",
                description="Prompt for archiving",
                timestamp=datetime.now().isoformat()
            )
        ]
        
        manager.commit_prompt_changes("archive-test", changes)
        
        # Archive experiment
        success = manager.archive_experiment("archive-test")
        assert success
        
        # Verify archive directory exists
        archive_dir = manager.repo_path / "archived" / "archive-test"
        assert archive_dir.exists()
        assert (archive_dir / "metadata.json").exists()
        assert (archive_dir / "archived_prompt.txt").exists()
        
        # Verify experiment is removed from active experiments
        experiments = manager.get_experiment_history("archive-test")
        assert len(experiments) == 0
    
    def test_experiment_history(self, manager):
        """Test getting experiment history."""
        # Initially empty
        history = manager.get_experiment_history()
        assert len(history) == 0
        
        # Create multiple experiments
        manager.create_experiment_branch("exp1", ExperimentType.T1_ONE_WAY, "First experiment")
        try:
            manager._run_git_command(["checkout", "main"])
        except:
            manager._run_git_command(["checkout", "master"])
        
        manager.create_experiment_branch("exp2", ExperimentType.T2_REVERSIBLE, "Second experiment")
        try:
            manager._run_git_command(["checkout", "main"])
        except:
            manager._run_git_command(["checkout", "master"])
        
        # Get all history
        history = manager.get_experiment_history()
        assert len(history) == 2
        
        # Should be sorted by creation date (most recent first)
        assert history[0].experiment_id == "exp2"
        assert history[1].experiment_id == "exp1"
        
        # Get specific experiment
        specific = manager.get_experiment_history("exp1")
        assert len(specific) == 1
        assert specific[0].experiment_id == "exp1"
        
        # Non-existent experiment
        none_exp = manager.get_experiment_history("non-existent")
        assert len(none_exp) == 0
    
    def test_error_handling(self, manager):
        """Test error handling for various scenarios."""
        # Test rollback non-existent experiment
        result = manager.rollback_experiment("non-existent")
        assert result == False  # Should return False, not raise exception
        
        # Test complete non-existent experiment
        with pytest.raises(ValueError, match="Experiment non-existent not found"):
            manager.complete_experiment("non-existent", {})
        
        # Test merge non-completed experiment
        manager.create_experiment_branch("incomplete", ExperimentType.T2_REVERSIBLE, "Incomplete")
        
        with pytest.raises(ValueError, match="Experiment incomplete is not completed"):
            manager.merge_experiment_to_main("incomplete")
        
        # Test commit to non-existent experiment
        changes = [PromptChange("test", "", "content", "new", "desc", datetime.now().isoformat())]
        
        with pytest.raises(ValueError, match="Experiment non-existent not found"):
            manager.commit_prompt_changes("non-existent", changes)


def test_global_manager_instance():
    """Test the global manager instance function."""
    from src.git_prompt_manager import get_git_prompt_manager
    
    manager1 = get_git_prompt_manager()
    manager2 = get_git_prompt_manager()
    
    # Should return the same instance
    assert manager1 is manager2
    assert isinstance(manager1, GitPromptManager)