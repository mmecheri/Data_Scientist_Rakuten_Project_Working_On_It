import os
import sys
from pathlib import Path

def get_project_root(max_levels=2):
    """
    Automatically detects the project root directory.
    
    - It goes up a maximum of `max_levels` directories (default: 3).
    - If `config.py` is found, it assumes it's the root.
    - Otherwise, raises an error if no root is detected.

    Args:
        max_levels (int): Maximum levels to go up when searching.

    Returns:
        Path: The detected project root directory.
    """
    current_dir = Path(os.getcwd()).resolve()
    
    for _ in range(max_levels):
        if (current_dir / "config.py").exists():
            return current_dir
        current_dir = current_dir.parent  # Move one level up

    raise RuntimeError(f"[X] Project root not found after {max_levels} levels! Ensure `config.py` is at the root.")

# Automatically detect project root
PROJECT_ROOT = get_project_root()

# Add project root to sys.path for module imports
sys.path.append(str(PROJECT_ROOT))

def get_relative_path(absolute_path):
    """Returns the relative path from the project root."""
    return str(Path(absolute_path).relative_to(PROJECT_ROOT))

# Debug mode: Print root directory when running this script directly
if __name__ == "__main__":
    print(f"✔ Project Root Detected: {PROJECT_ROOT}")
