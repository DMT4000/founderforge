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
    
    # Feature Flags
    def load_feature_flags(self) -> Dict[str, Any]:
        """Load feature flags from JSON configuration."""
        feature_flags_path = self.config_dir / "feature_flags.json"
        
        if not feature_flags_path.exists():
            # Create default feature flags
            default_flags = {
                "enable_memory_system": True,
                "enable_multi_agent": True,
                "enable_context_summarization": True,
                "enable_confidence_fallback": True,
                "enable_git_versioning": True,
                "enable_security_filters": True,
                "max_context_tokens": 16000,
                "confidence_threshold": 0.8,
                "memory_retention_days": 30
            }
            self.save_feature_flags(default_flags)
            return default_flags
        
        with open(feature_flags_path, 'r') as f:
            return json.load(f)
    
    def save_feature_flags(self, flags: Dict[str, Any]) -> None:
        """Save feature flags to JSON configuration."""
        feature_flags_path = self.config_dir / "feature_flags.json"
        with open(feature_flags_path, 'w') as f:
            json.dump(flags, f, indent=2)
    
    def get_feature_flag(self, flag_name: str, default: Any = False) -> Any:
        """Get a specific feature flag value."""
        flags = self.load_feature_flags()
        return flags.get(flag_name, default)
    
    def update_feature_flag(self, flag_name: str, value: Any) -> None:
        """Update a specific feature flag."""
        flags = self.load_feature_flags()
        flags[flag_name] = value
        self.save_feature_flags(flags)

# Global settings instance
settings = Settings()