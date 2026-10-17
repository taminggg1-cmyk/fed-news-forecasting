"""The 2026-10-17 reorganisation kept every old module name importable and runnable (src/fed_news/__init__.py)."""
import importlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import fed_news

SRC = Path(fed_news.__file__).resolve().parent


def test_every_old_name_is_the_moved_module():
    for old, new in fed_news.MOVED.items():
        assert importlib.import_module(old) is importlib.import_module(new), old
        assert sys.modules[old].__spec__.name == new, old


def test_every_old_name_has_runnable_code_for_python_dash_m():
    for old, new in fed_news.MOVED.items():
        spec = fed_news._AliasFinder().find_spec(old)  # what runpy gets when the old name is not yet imported
        assert spec.origin == importlib.util.find_spec(new).origin, old
        assert spec.loader.get_code(old) is not None, old


def test_alias_map_covers_every_moved_module_and_nothing_else():
    on_disk = {p.stem for p in SRC.rglob("*.py") if p.parent != SRC and p.stem != "__init__"}
    mapped = {new.rsplit(".", 1)[1] for new in fed_news.MOVED.values()}
    assert mapped == on_disk - {"linear"}  # linear.py was models.py; fed_news.models is now its package


def test_old_models_name_still_exposes_the_linear_models():
    from fed_news import models
    from fed_news.models import linear
    assert models.OrderedLogit is linear.OrderedLogit
    assert models.choose_penalty is linear.choose_penalty


def test_old_command_line_entry_point_still_runs():
    out = subprocess.run([sys.executable, "-m", "fed_news.run_models", "--help"], capture_output=True, text=True,
                         env={**os.environ, "PYTHONPATH": str(SRC.parent)})
    assert out.returncode == 0, out.stderr
    assert "--no-text" in out.stdout
