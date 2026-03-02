"""Shared IO helpers."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


def load_pickle(path: str | Path) -> Any:
    """Load a pickled object from disk."""
    with Path(path).open("rb") as f:
        return pickle.load(f)


def save_pickle(obj: Any, path: str | Path) -> None:
    """Write an object to a pickle file, creating parent directories."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(obj, f)
