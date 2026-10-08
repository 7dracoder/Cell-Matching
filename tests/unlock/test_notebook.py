"""Structure checks for CellMatch_HPC_Unlock.ipynb (task 15.2; Req 1.1, 2.1, 2.2)."""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import pytest

from hpc_unlock import report
from hpc_unlock.config import UnlockConfig

REPO = Path(__file__).resolve().parents[2]
NB = REPO / "CellMatch_HPC_Unlock.ipynb"
BUILDER = REPO / "research" / "build_unlock_notebook.py"

FORBIDDEN = ("sg8304", "ts5789", "ece_gy")
ALLOWED_HPC = {"hpc_unlock.paths", "hpc_unlock.config", "hpc_unlock.chain"}
ALLOWED_FROM_PKG = {"paths", "config", "chain"}


@pytest.fixture(scope="module")
def nb() -> dict:
    return json.loads(NB.read_text(encoding="utf-8"))


def _src(cell: dict) -> str:
    s = cell["source"]
    return "".join(s) if isinstance(s, list) else s


def _code_cells(nb: dict) -> list[str]:
    return [_src(c) for c in nb["cells"] if c["cell_type"] == "code"]


def _python_only(src: str) -> str:
    """Drop shell (``!``) and magic (``%``) lines before parsing."""
    return "\n".join("" if ln.lstrip().startswith(("!", "%")) else ln
                     for ln in src.splitlines())


def test_valid_nbformat4(nb):
    assert nb["nbformat"] == 4
    assert nb["metadata"]["kernelspec"]["name"] == "python3"
    ids = set()
    for c in nb["cells"]:
        assert c["cell_type"] in ("markdown", "code")
        assert isinstance(c["source"], (list, str))
        assert isinstance(c["metadata"], dict)
        if nb["nbformat_minor"] >= 5:
            assert c["id"] not in ids
            ids.add(c["id"])
        if c["cell_type"] == "code":
            assert c["outputs"] == [] and c["execution_count"] is None


def test_builder_is_reproducible(nb):
    sys.path.insert(0, str(BUILDER.parent))
    try:
        import build_unlock_notebook as b
    finally:
        sys.path.remove(str(BUILDER.parent))
    assert b.build() == nb


def test_instruction_cell_before_every_code_cell(nb):
    cells = nb["cells"]
    n_code = 0
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        assert i > 0 and cells[i - 1]["cell_type"] == "markdown", f"code cell {i} has no instruction"
        md = _src(cells[i - 1])
        m = re.search(r"\*\*Cell (\d+):", md)
        assert m, f"instruction before code cell {i} does not name the cell"
        assert int(m.group(1)) == n_code
        assert f"# Cell {n_code}" in _src(c)
        assert "Expected output" in md
        n_code += 1
    assert n_code == 7


def test_steps_present(nb):
    text = "\n".join(_src(c) for c in nb["cells"])
    for step in ("Step 0", "Step 1", "Step 2", "Step 3", "Step 4", "Step 5"):
        assert step in text
    assert "chain.submit_setup(cfg)" in text
    assert 'chain.submit(cfg, start="prep")' in text
    assert "chain.monitor(cfg)" in text
    assert "chain.submit(cfg, start=START_STAGE)" in text
    assert "run_unlock.py" in text and '"check"' in text
    assert "greene-dtn:/scratch/work/public/overlay-fs-ext3/overlay-15GB-500K.ext3.gz" in text
    assert "gunzip" in text
    assert "cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif" in text
    assert "kernel.json" in text and ".local/share/jupyter/kernels/cellmatch-unlock" in text
    assert report.MD_NAME in text and report.JSON_NAME in text
    assert "submission_v13_*.csv" in text
    assert "not guaranteed" in text


def test_config_defaults(nb):
    cfg_cell = next(s for s in _code_cells(nb) if "UnlockConfig(" in s)
    assert 'ENV_MODE = "singularity"' in cfg_cell
    assert 'GPU_PARTITION = "g2-standard-12"' in cfg_cell
    assert "c12m85-a100-1" in cfg_cell
    assert "DISABLE_SELFTRAIN = False" in cfg_cell
    assert "SIGMA = 2.5" in cfg_cell and "SCAN_K = 50" in cfg_cell
    assert "MATCH_RADIUS = 6.0" in cfg_cell and "SEED_RADIUS = 6.0" in cfg_cell
    assert "cfg.validate()" in cfg_cell
    assert "account=" not in cfg_cell          # account comes from UnlockConfig
    assert UnlockConfig().account == "cs_gy_6923-2026fa"


def test_config_cell_builds_valid_default_config(nb):
    """Execute Cell 1 against the real modules: defaults validate, plan has 12 Stages."""
    cfg_cell = next(s for s in _code_cells(nb) if "UnlockConfig(" in s)
    ns: dict = {}
    exec(compile(cfg_cell, "cell1", "exec"), ns)
    assert ns["CONFIG_OK"] is True and ns["CONFIG_ERRORS"] == []
    cfg = ns["cfg"]
    assert (cfg.account, cfg.gpu_partition, cfg.env_mode, cfg.disable_selftrain) == \
        ("cs_gy_6923-2026fa", "g2-standard-12", "singularity", False)
    assert len(["setup"] + ns["chain"].plan(cfg, "prep")) == 12


def test_no_netid_or_old_account(nb):
    raw = NB.read_text(encoding="utf-8") + BUILDER.read_text(encoding="utf-8")
    for s in FORBIDDEN:
        assert s not in raw
    assert not re.search(r"/scratch/[a-z]{2,}\d+", raw)


def test_code_cells_compile(nb):
    for src in _code_cells(nb):
        ast.parse(_python_only(src))


def _inside_try(tree: ast.AST) -> set[int]:
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            for sub in node.body:
                for n in ast.walk(sub):
                    ids.add(id(n))
    return ids


def test_imports_limited(nb):
    stdlib = set(sys.stdlib_module_names)
    for src in _code_cells(nb):
        tree = ast.parse(_python_only(src))
        in_try = _inside_try(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.module == "hpc_unlock":
                    assert {a.name for a in node.names} <= ALLOWED_FROM_PKG
                    continue
                mods = [node.module]
            else:
                continue
            for mod in mods:
                top = mod.split(".")[0]
                if top == "IPython":
                    assert mod == "IPython.display" and id(node) in in_try
                elif top == "hpc_unlock":
                    assert mod in ALLOWED_HPC, mod
                else:
                    assert top in stdlib, mod
