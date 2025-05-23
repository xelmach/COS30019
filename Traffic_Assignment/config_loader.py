#105106819 Suman Sutparai
# Config loader for TBRGS

import json
import os
from pathlib import Path

class ConfigLoader:
    """Configuration loader for TBRGS app."""
    
    def __init__(self):
        """Initialize the configuration loader."""
        self.config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        self.config = self.load_config()
    
    def load_config(self):
        """Load configuration from JSON file."""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in config file: {self.config_path}")
    
    def get(self, key, default=None):
        """Get a configuration value."""
        return self.config.get(key, default)
    
    def get_model_config(self, model_type):
        """Get model-specific configuration."""
        return self.config.get('model', {}).get(model_type, {})
    
    def get_ml_config(self):
        """Get machine learning configuration."""
        return self.config.get('ml', {})
    
    def get_gui_config(self):
        """Get GUI configuration."""
        return self.config.get('gui', {})
        
    def update(self, section, key, value):
        """Update configuration value."""
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
        
        # Save updated config
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=4) 