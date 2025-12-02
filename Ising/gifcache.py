"""
gifcache.py

Simple GIF caching context manager:
- If the GIF already exists, do nothing (is_cache=True)
- If it doesn't exist, let the user generate it inside the context
"""

from pathlib import Path


class CacheObject:
    def __init__(self, is_cache: bool):
        self.is_cache = is_cache


class GifCache:
    """Caching context manager for GIF files."""

    def __init__(self, path):
        self.path = Path(path)

    def _load(self):
        # If file is here → GIF already cached
        return CacheObject(is_cache=self.path.exists())

    def _save(self):
        # Saving happens implicitly: if GIF wasn’t in cache, user code creates it.
        # We just verify it exists.
        if not self.path.exists():
            raise FileNotFoundError(
                f"GifCache: expected GIF '{self.path}' to be created inside the context block."
            )

    def __enter__(self):
        self.obj = self._load()
        return self.obj

    def __exit__(self, exc_type, exc_value, traceback):
        # Don’t save on error
        if exc_type is not None:
            return False

        # If not cached, we expect the user to have generated the GIF inside the block
        if not self.obj.is_cache:
            self._save()

        return False
