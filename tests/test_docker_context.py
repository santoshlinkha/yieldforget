"""The Docker build context admits the package source and nothing private.

Applies .dockerignore's rules (last matching pattern wins; a matched folder takes
its contents) to synthetic paths; no image is built.
"""

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[1]


def _rules():
    out = []
    for line in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        keep, pattern = line.startswith("!"), line.lstrip("!").rstrip("/")
        regex = re.escape(pattern).replace(r"\*\*/", "(?:.*/)?").replace(r"\*", "[^/]*").replace(r"\[", "[") \
            .replace(r"\]", "]")
        out.append((keep, re.compile(regex)))
    return out


def _excluded(path: str) -> bool:
    parts = path.split("/")
    excluded = False
    for i in range(1, len(parts) + 1):          # a folder's verdict carries to what is inside it
        prefix = "/".join(parts[:i])
        for keep, rx in _rules():
            if rx.fullmatch(prefix):
                excluded = not keep
    return excluded


@pytest.mark.parametrize("path", [
    "portfolio.sqlite3", "portfolio.sqlite3-wal", "market_data.sqlite3", "my_portfolio.json", "my_universe.json",
    ".env", ".env.enterprise", ".env.local", "backups/portfolio_x.sqlite3", "bars/AAA.parquet", "data/x.csv",
    "recommendations/r.json", "results/run/out.json", "uv.lock", "p.out", ".git/config", ".venv/pyvenv.cfg",
    "tradingagents/.env", "tradingagents/webapp/my_portfolio.json", "tradingagents/x/cache.sqlite3",
    "tradingagents/__pycache__/a.cpython-313.pyc", "cli/results/x.json",
])
def test_private_records_and_credentials_stay_out(path):
    assert _excluded(path), path


@pytest.mark.parametrize("path", [
    "pyproject.toml", "README.md", "LICENSE", "tradingagents/__init__.py", "tradingagents/webapp/static/index.html",
    "tradingagents/assets/logo.svg", "cli/main.py", "cli/static/welcome.txt",
])
def test_the_package_source_goes_in(path):
    assert not _excluded(path), path
