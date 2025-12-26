"""I/O helpers for pipeline stages."""

from pathlib import Path
import pandas as pd


def safe_to_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write parquet with a fastparquet fallback for environments where pyarrow fails."""
    try:
        df.to_parquet(path, index=False)
    except Exception:
        df.to_parquet(path, index=False, engine="fastparquet")


def safe_read_parquet(path: Path) -> pd.DataFrame:
    """Read parquet with a fastparquet fallback for environments where pyarrow fails."""
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.read_parquet(path, engine="fastparquet")
