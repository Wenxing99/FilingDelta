from __future__ import annotations

from pathlib import Path

from filingdelta.core.config import get_settings


def ensure_data_dirs() -> dict[str, Path]:
    settings = get_settings()

    paths = {
        "samples": Path("data/samples").resolve(),
        "raw": Path("data/raw").resolve(),
        "parsed": Path("data/parsed").resolve(),
        "indexes": Path("data/indexes").resolve(),
        "outputs": Path("data/outputs").resolve(),
        "qdrant": settings.qdrant_path,
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths
