"""Configuration management for FounderForge AI Cofounder."""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    """Central configuration management class."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.data_dir = self.project_root / "data"
        self.config_dir = self.project_root / "config"
        
        # Ensure directories exist
        self.data_dir.mkdir(exist_ok=True)
        self.config_dir.mkdir(exist_ok=True)
        
    # Environment Variables
    @property
    def gemini_api_key(self) -> Optional[str]:
        """Get Gemini API key from environment."""
        return os.getenv("GEMINI_API_KEY")
    
    @property
    def debug_mode(self) -> bool:
        """Check if debug mode is enabled."""
        return os.getenv("DEBUG", "false").lower() == "true"
    
    @property
    def log_level(self) -> str:
        """Get logging level."""
        return os.getenv("LOG_LEVEL", "INFO")
    
    # Database Configuration
    @property
    def database_path(self) -> Path:
        """Get SQLite database path."""
        return self.data_dir / "founderforge.db"
    
    @property
    def faiss_index_path(self) -> Path:
        """Get FAISS index storage path."""
        return self.data_dir / "faiss_index"
    
    # Feature Flags (delegated to FeatureFlagManager)
    @property
    def feature_flag_manager(self):
        """Get the feature flag manager instance."""
        # Import here to avoid circular imports
        from src.feature_flag_manager import get_feature_flag_manager
        return get_feature_flag_manager()
    
    def load_feature_flags(self) -> Dict[str, Any]:
        """Load feature flags from JSON configuration."""
        return self.feature_flag_manager.get_all_flags()
    
    def save_feature_flags(self, flags: Dict[str, Any]) -> None:
        """Save feature flags to JSON configuration."""
        # This method is deprecated - use feature_flag_manager.set_flag() instead
        for flag_name, value in flags.items():
            self.feature_flag_manager.set_flag(flag_name, value, "settings", "Bulk update via save_feature_flags")
    
    def get_feature_flag(self, flag_name: str, default: Any = False) -> Any:
        """Get a specific feature flag value."""
        return self.feature_flag_manager.get_flag(flag_name, default)
    
    def update_feature_flag(self, flag_name: str, value: Any) -> None:
        """Update a specific feature flag."""
        self.feature_flag_manager.set_flag(flag_name, value, "settings", f"Updated via settings.update_feature_flag")

# Global settings instance
settings = Settings()