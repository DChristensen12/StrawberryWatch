import os
import sys

# Lets this run as `python -m scripts.clear_cache` from the project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config


def main():
    """Deletes the local data cache CSV if it exists."""
    path = Config.DATA_FILE
    if os.path.exists(path):
        size_kb = os.path.getsize(path) / 1024
        os.remove(path)
        print(f"Cleared {path} ({size_kb:.1f} KB)")
    else:
        print(f"No cache file at {path} — nothing to clear.")


if __name__ == "__main__":
    main()
    