"""Shared fixtures for the unlock tests (local, CPU, light)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


@pytest.fixture
def project_tree(tmp_path: Path) -> Path:
    """A minimal empty project tree mirroring the real Project_Folder layout."""
    root = tmp_path / "cellmatch"
    for sub in ("hpc_unlock",
                "research/data/hpc",
                "research/data/hpc_smoke",
                "Project_2_Dataset/training",
                "Project_2_Dataset/hidden_test",
                "hpc/unlock/container",
                "hpc/unlock/models",
                "logs"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root
