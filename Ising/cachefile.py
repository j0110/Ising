"""
cachefile.py
JSON caching context manager with automatic NumPy array handling.
"""

import json
from pathlib import Path
from contextlib import contextmanager
import numpy as np


# ------------------------------
# Serialization helpers
# ------------------------------
def to_jsonable(obj):
    """Convert NumPy arrays recursively into JSON-serializable lists."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj  # numbers, strings, bools, None are fine


def from_jsonable(obj):
    """Optionally convert lists back to NumPy arrays if desired."""
    # For now, leave arrays as lists (plotting etc usually accepts lists)
    # You can convert back explicitly in your code if needed.
    return obj


# ------------------------------
# Cache classes
# ------------------------------
class CacheObject:
    def __init__(self, is_cache: bool, value=None):
        self.is_cache = is_cache
        self.value = value


class CacheFile:
    """JSON-only caching with NumPy array support."""
    
    def __init__(self, path):
        self.path = Path(path)

    def _load(self):
        if not self.path.exists():
            return CacheObject(is_cache=False)
        with open(self.path, "r") as f:
            data = json.load(f)
        return CacheObject(is_cache=True, value=from_jsonable(data))

    def _save(self, obj: CacheObject):
        if obj.value is None:
            raise ValueError(
                f"CacheFile: No value stored in cache block for file {self.path}"
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(to_jsonable(obj.value), f)

    def __enter__(self):
        self.obj = self._load()
        return self.obj

    def __exit__(self, exc_type, exc_value, traceback):
        # Don't save if an error occurred
        if exc_type is not None:
            return False

        if not self.obj.is_cache:
            self._save(self.obj)

        return False
