import yaml
from pathlib import Path
from typing import Dict, Any
#TODO: write a test for this function.
def load_config(config_path: str = "config.yml") -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    # Try current directory
    p = Path(config_path)
    if not p.exists():
        # Try parent directory (common when running from src/)
        p = Path("../") / config_path
    
    if not p.exists():
        # Return a default or handle error
        print(f"Warning: Config file not found at {config_path}. Using empty config.")
        return {"data": {"resting": "./data/raw/ds004504"}, "split_seed": 42}
        
    with open(p, "r") as file:
        return yaml.safe_load(file)
