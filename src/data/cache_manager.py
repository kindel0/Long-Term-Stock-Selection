"""
Cache management for data files.

Provides utilities for caching processed data to improve
loading performance on subsequent runs.
"""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import pandas as pd

from ..config import CACHE_DIR

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Manages caching of DataFrames and other data.

    Features:
    - Parquet caching for fast loading
    - Cache invalidation based on source file changes
    - Metadata tracking for cache entries
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the cache manager.

        Args:
            cache_dir: Directory for cache files (default from config)
        """
        self.cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self._metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Load cache metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _save_metadata(self) -> None:
        """Save cache metadata to disk."""
        with open(self.metadata_file, "w") as f:
            json.dump(self._metadata, f, indent=2, default=str)

    def _get_cache_key(self, name: str, params: Optional[Dict] = None) -> str:
        """Generate a cache key from name and parameters."""
        key_parts = [name]
        if params:
            # Sort params for consistent hashing
            param_str = json.dumps(params, sort_keys=True)
            key_parts.append(hashlib.md5(param_str.encode()).hexdigest()[:8])
        return "_".join(key_parts)

    def _get_file_hash(self, filepath: Path) -> str:
        """Get a hash of a file's contents (first 10MB only for large files)."""
        filepath = Path(filepath)
        if not filepath.exists():
            return ""

        hasher = hashlib.md5()
        max_bytes = 10 * 1024 * 1024  # 10MB

        with open(filepath, "rb") as f:
            data = f.read(max_bytes)
            hasher.update(data)

        return hasher.hexdigest()

    def get(
        self,
        name: str,
        source_files: Optional[list] = None,
        params: Optional[Dict] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Get a cached DataFrame if valid.

        Args:
            name: Cache entry name
            source_files: List of source file paths to check for changes
            params: Parameters used to generate the cached data

        Returns:
            Cached DataFrame if valid, None otherwise
        """
        cache_key = self._get_cache_key(name, params)
        cache_path = self.cache_dir / f"{cache_key}.parquet"

        if not cache_path.exists():
            logger.debug(f"Cache miss (not found): {name}")
            return None

        # Check if source files have changed
        if source_files:
            current_hashes = {
                str(f): self._get_file_hash(Path(f)) for f in source_files
            }
            cached_hashes = self._metadata.get(cache_key, {}).get("source_hashes", {})

            if current_hashes != cached_hashes:
                logger.info(f"Cache invalidated (source changed): {name}")
                return None

        # Load cached data
        try:
            df = pd.read_parquet(cache_path)
            logger.info(f"Cache hit: {name} ({len(df)} rows)")
            return df
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return None

    def put(
        self,
        name: str,
        df: pd.DataFrame,
        source_files: Optional[list] = None,
        params: Optional[Dict] = None,
    ) -> Path:
        """
        Cache a DataFrame.

        Args:
            name: Cache entry name
            df: DataFrame to cache
            source_files: List of source file paths
            params: Parameters used to generate the data

        Returns:
            Path to the cached file
        """
        cache_key = self._get_cache_key(name, params)
        cache_path = self.cache_dir / f"{cache_key}.parquet"

        # Save DataFrame
        df.to_parquet(cache_path, index=False)

        # Update metadata
        self._metadata[cache_key] = {
            "name": name,
            "created": datetime.now().isoformat(),
            "shape": list(df.shape),
            "params": params,
            "source_hashes": {},
        }

        if source_files:
            self._metadata[cache_key]["source_hashes"] = {
                str(f): self._get_file_hash(Path(f)) for f in source_files
            }

        self._save_metadata()
        logger.info(f"Cached: {name} ({len(df)} rows)")

        return cache_path

    def invalidate(self, name: str, params: Optional[Dict] = None) -> bool:
        """
        Invalidate a cache entry.

        Args:
            name: Cache entry name
            params: Parameters used to generate the cached data

        Returns:
            True if cache was invalidated, False if not found
        """
        cache_key = self._get_cache_key(name, params)
        cache_path = self.cache_dir / f"{cache_key}.parquet"

        if cache_path.exists():
            cache_path.unlink()
            if cache_key in self._metadata:
                del self._metadata[cache_key]
                self._save_metadata()
            logger.info(f"Invalidated cache: {name}")
            return True

        return False

    def clear_all(self) -> int:
        """
        Clear all cache entries.

        Returns:
            Number of entries cleared
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.parquet"):
            cache_file.unlink()
            count += 1

        self._metadata = {}
        self._save_metadata()
        logger.info(f"Cleared {count} cache entries")

        return count

    def get_or_compute(
        self,
        name: str,
        compute_fn: Callable[[], pd.DataFrame],
        source_files: Optional[list] = None,
        params: Optional[Dict] = None,
        force_recompute: bool = False,
    ) -> pd.DataFrame:
        """
        Get cached data or compute and cache it.

        Args:
            name: Cache entry name
            compute_fn: Function to compute the data if not cached
            source_files: Source files to track for invalidation
            params: Parameters for the computation
            force_recompute: If True, bypass cache and recompute

        Returns:
            DataFrame (from cache or freshly computed)
        """
        if not force_recompute:
            cached = self.get(name, source_files, params)
            if cached is not None:
                return cached

        # Compute fresh
        logger.info(f"Computing: {name}")
        df = compute_fn()

        # Cache result
        self.put(name, df, source_files, params)

        return df

    def list_entries(self) -> list:
        """
        List all cache entries.

        Returns:
            List of cache entry metadata dicts
        """
        entries = []
        for key, meta in self._metadata.items():
            cache_path = self.cache_dir / f"{key}.parquet"
            entries.append({
                "key": key,
                "name": meta.get("name"),
                "created": meta.get("created"),
                "shape": meta.get("shape"),
                "exists": cache_path.exists(),
                "size_mb": cache_path.stat().st_size / 1e6 if cache_path.exists() else 0,
            })
        return entries

    def print_status(self) -> None:
        """Print cache status to console."""
        entries = self.list_entries()

        print("\n" + "=" * 60)
        print("CACHE STATUS")
        print("=" * 60)
        print(f"Cache directory: {self.cache_dir}")
        print(f"Entries: {len(entries)}")

        if entries:
            total_size = sum(e["size_mb"] for e in entries)
            print(f"Total size: {total_size:.1f} MB")
            print("\nEntries:")
            for entry in entries:
                status = "OK" if entry["exists"] else "MISSING"
                print(
                    f"  {entry['name']}: {entry['shape']} "
                    f"({entry['size_mb']:.1f} MB) [{status}]"
                )

        print("=" * 60)
