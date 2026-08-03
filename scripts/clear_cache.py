import os

from strawberrywatch.config import Config


def main():
    """Deletes the local data cache CSV if it exists."""
    path = Config.data_file()
    if os.path.exists(path):
        size_kb = os.path.getsize(path) / 1024
        os.remove(path)
        print(f"Cleared {path} ({size_kb:.1f} KB)")
    else:
        print(f"No cache file at {path}, nothing to clear.")


if __name__ == "__main__":
    main()
