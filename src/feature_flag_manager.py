"""Feature flag management system with runtime toggles and change tracking."""

import json
import logging
from logging_manager import get_logging_manager, LogLevel, LogCategory
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from threading import Lock
import os

@dataclass
class FeatureFlagChange:
    """Represents a feature flag change event."""
    timestamp: str
    flag_name: str
    old_value: Any
    new_value: Any
    changed_by: str
    reason: Optional[str] = None

class FeatureFlagManager:
    """Enhanced feature flag manager with runtime toggles and change tracking."""
    
    def __init__(self, config_dir: Path):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        self.feature_flags_path = self.config_dir / "feature_flags.json"
        self.change_log_path = self.config_dir / "feature_flag_changes.json"
        
        # Thread safety for concurrent access
        self._lock = Lock()
        
        # Cache for feature flags
        self._flags_cache: Optional[Dict[str, Any]] = None
        self._cache_timestamp: Optional[float] = None
        
        # Initialize logging
        self.logger = get_logging_manager().get_logger(__name__.split(".")[-1], LogCategory.SYSTEM)
        
        # Ensure feature flags file exists
        self._ensure_feature_flags_exist()
    
    def _ensure_feature_flags_exist(self) -> None:
        """Ensure feature flags file exists with default values."""
        if not self.feature_flags_path.exists():
            default_flags = {
                "enable_memory_system": True,
                "enable_multi_agent": True,
                "enable_context_summarization": True,
                "enable_confidence_fallback": True,
                "enable_git_versioning": True,
                "enable_security_filters": True,
                "enable_vector_search": True,
                "enable_token_monitoring": True,
                "enable_performance_logging": True,
                "enable_experiment_tracking": True,
                "max_context_tokens": 16000,
                "confidence_threshold": 0.8,
                "memory_retention_days": 30,
                "max_memory_retrieval_time_ms": 10,
                "agent_workflow_timeout_seconds": 30,
                "gemini_api_retry_attempts": 3,
                "gemini_api_timeout_seconds": 30
            }
            self._save_flags_to_file(default_flags)
            self.logger.info("Created default feature flags configuration")
    
    def _load_flags_from_file(self) -> Dict[str, Any]:
        """Load feature flags from JSON file."""
        try:
            with open(self.feature_flags_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.logger.error(f"Error loading feature flags: {e}")
            return {}
    
    def _save_flags_to_file(self, flags: Dict[str, Any]) -> None:
        """Save feature flags to JSON file."""
        try:
            with open(self.feature_flags_path, 'w') as f:
                json.dump(flags, f, indent=2, sort_keys=True)
        except Exception as e:
            self.logger.error(f"Error saving feature flags: {e}")
            raise
    
    def _get_file_modification_time(self) -> float:
        """Get the modification time of the feature flags file."""
        try:
            return self.feature_flags_path.stat().st_mtime
        except FileNotFoundError:
            return 0.0
    
    def _should_refresh_cache(self) -> bool:
        """Check if cache should be refreshed based on file modification time."""
        if self._flags_cache is None or self._cache_timestamp is None:
            return True
        
        current_mtime = self._get_file_modification_time()
        return current_mtime > self._cache_timestamp
    
    def get_all_flags(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Get all feature flags with caching support."""
        with self._lock:
            if force_refresh or self._should_refresh_cache():
                self._flags_cache = self._load_flags_from_file()
                self._cache_timestamp = self._get_file_modification_time()
                self.logger.debug("Refreshed feature flags cache")
            
            return self._flags_cache.copy() if self._flags_cache else {}
    
    def get_flag(self, flag_name: str, default: Any = None) -> Any:
        """Get a specific feature flag value."""
        flags = self.get_all_flags()
        value = flags.get(flag_name, default)
        self.logger.debug(f"Retrieved flag '{flag_name}': {value}")
        return value
    
    def is_enabled(self, flag_name: str) -> bool:
        """Check if a boolean feature flag is enabled."""
        return bool(self.get_flag(flag_name, False))
    
    def set_flag(self, flag_name: str, value: Any, changed_by: str = "system", reason: Optional[str] = None) -> bool:
        """Set a feature flag value with change tracking."""
        with self._lock:
            try:
                # Load flags directly to avoid nested lock acquisition
                if self._should_refresh_cache():
                    self._flags_cache = self._load_flags_from_file()
                    self._cache_timestamp = self._get_file_modification_time()
                
                current_flags = self._flags_cache.copy() if self._flags_cache else {}
                old_value = current_flags.get(flag_name)
                
                # Only update if value actually changed
                if old_value == value:
                    self.logger.debug(f"Flag '{flag_name}' already has value {value}, no change needed")
                    return False
                
                # Update the flag
                current_flags[flag_name] = value
                self._save_flags_to_file(current_flags)
                
                # Log the change
                self._log_flag_change(flag_name, old_value, value, changed_by, reason)
                
                # Invalidate cache
                self._flags_cache = None
                self._cache_timestamp = None
                
                self.logger.info(f"Updated flag '{flag_name}' from {old_value} to {value} by {changed_by}")
                return True
                
            except Exception as e:
                self.logger.error(f"Error setting flag '{flag_name}': {e}")
                raise
    
    def toggle_flag(self, flag_name: str, changed_by: str = "system", reason: Optional[str] = None) -> bool:
        """Toggle a boolean feature flag."""
        current_value = self.get_flag(flag_name, False)
        if not isinstance(current_value, bool):
            raise ValueError(f"Cannot toggle non-boolean flag '{flag_name}' (current value: {current_value})")
        
        new_value = not current_value
        self.set_flag(flag_name, new_value, changed_by, reason)
        return new_value
    
    def _log_flag_change(self, flag_name: str, old_value: Any, new_value: Any, changed_by: str, reason: Optional[str]) -> None:
        """Log a feature flag change to the change log file."""
        try:
            change = FeatureFlagChange(
                timestamp=datetime.now().isoformat(),
                flag_name=flag_name,
                old_value=old_value,
                new_value=new_value,
                changed_by=changed_by,
                reason=reason
            )
            
            # Load existing changes
            changes = self._load_change_log()
            changes.append(asdict(change))
            
            # Keep only last 1000 changes to prevent file from growing too large
            if len(changes) > 1000:
                changes = changes[-1000:]
            
            # Save updated changes
            with open(self.change_log_path, 'w') as f:
                json.dump(changes, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error logging flag change: {e}")
    
    def _load_change_log(self) -> List[Dict[str, Any]]:
        """Load the feature flag change log."""
        try:
            if self.change_log_path.exists():
                with open(self.change_log_path, 'r') as f:
                    return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.logger.warning(f"Could not load change log: {e}")
        
        return []
    
    def get_change_history(self, flag_name: Optional[str] = None, limit: int = 100) -> List[FeatureFlagChange]:
        """Get change history for feature flags."""
        changes = self._load_change_log()
        
        # Filter by flag name if specified
        if flag_name:
            changes = [c for c in changes if c.get('flag_name') == flag_name]
        
        # Sort by timestamp (most recent first) and limit
        changes.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        changes = changes[:limit]
        
        # Convert to FeatureFlagChange objects
        result = []
        for change_dict in changes:
            try:
                result.append(FeatureFlagChange(**change_dict))
            except Exception as e:
                self.logger.warning(f"Invalid change log entry: {e}")
        
        return result
    
    def export_flags(self, file_path: Path) -> None:
        """Export current feature flags to a file."""
        flags = self.get_all_flags()
        with open(file_path, 'w') as f:
            json.dump(flags, f, indent=2, sort_keys=True)
        self.logger.info(f"Exported feature flags to {file_path}")
    
    def import_flags(self, file_path: Path, changed_by: str = "import", reason: str = "Bulk import") -> int:
        """Import feature flags from a file."""
        try:
            with open(file_path, 'r') as f:
                imported_flags = json.load(f)
            
            changes_made = 0
            for flag_name, value in imported_flags.items():
                if self.set_flag(flag_name, value, changed_by, f"{reason} - {flag_name}"):
                    changes_made += 1
            
            self.logger.info(f"Imported {changes_made} feature flag changes from {file_path}")
            return changes_made
            
        except Exception as e:
            self.logger.error(f"Error importing flags from {file_path}: {e}")
            raise
    
    def validate_flags(self) -> List[str]:
        """Validate feature flags and return list of issues."""
        issues = []
        flags = self.get_all_flags()
        
        # Check for required flags
        required_flags = [
            "enable_memory_system",
            "enable_multi_agent", 
            "confidence_threshold",
            "max_context_tokens"
        ]
        
        for flag in required_flags:
            if flag not in flags:
                issues.append(f"Missing required flag: {flag}")
        
        # Validate value types and ranges
        if "confidence_threshold" in flags:
            threshold = flags["confidence_threshold"]
            if not isinstance(threshold, (int, float)) or not (0 <= threshold <= 1):
                issues.append("confidence_threshold must be a number between 0 and 1")
        
        if "max_context_tokens" in flags:
            tokens = flags["max_context_tokens"]
            if not isinstance(tokens, int) or tokens <= 0:
                issues.append("max_context_tokens must be a positive integer")
        
        return issues

# Global feature flag manager instance
_feature_flag_manager: Optional[FeatureFlagManager] = None

def get_feature_flag_manager() -> FeatureFlagManager:
    """Get the global feature flag manager instance."""
    global _feature_flag_manager
    if _feature_flag_manager is None:
        # Use default config directory to avoid circular imports
        from pathlib import Path
        project_root = Path(__file__).parent.parent
        config_dir = project_root / "config"
        _feature_flag_manager = FeatureFlagManager(config_dir)
    return _feature_flag_manager

# Convenience functions
def get_flag(flag_name: str, default: Any = None) -> Any:
    """Get a feature flag value."""
    return get_feature_flag_manager().get_flag(flag_name, default)

def is_enabled(flag_name: str) -> bool:
    """Check if a feature flag is enabled."""
    return get_feature_flag_manager().is_enabled(flag_name)

def set_flag(flag_name: str, value: Any, changed_by: str = "system", reason: Optional[str] = None) -> bool:
    """Set a feature flag value."""
    return get_feature_flag_manager().set_flag(flag_name, value, changed_by, reason)

def toggle_flag(flag_name: str, changed_by: str = "system", reason: Optional[str] = None) -> bool:
    """Toggle a boolean feature flag."""
    return get_feature_flag_manager().toggle_flag(flag_name, changed_by, reason)