"""
Config loader — finds, loads, and hashes config.yaml.

Usage:
    from stockgrader.config import get_config, get_config_hash, get_portfolio_config

Discovery order:
  1. STOCKGRADER_CONFIG env var (absolute path to any .yaml file)
  2. config.yaml in the current working directory
  3. config.yaml at the project root (parent of the stockgrader package)
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import yaml

_config: dict[str, Any] | None = None
_config_hash: str = ""
_config_path: Path | None = None


def _find_config() -> Path:
    env_path = os.environ.get("STOCKGRADER_CONFIG")
    if env_path:
        p = Path(env_path)
        if p.is_file():
            return p
        raise FileNotFoundError(f"STOCKGRADER_CONFIG points to missing file: {env_path}")

    # CWD first, then walk up to find the package root
    candidates = [Path.cwd() / "config.yaml"]
    pkg_root = Path(__file__).parent.parent
    candidates.append(pkg_root / "config.yaml")

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "config.yaml not found. "
        "Set STOCKGRADER_CONFIG env var or place config.yaml in the project root."
    )


def _load() -> None:
    global _config, _config_hash, _config_path
    _config_path = _find_config()
    raw = _config_path.read_text(encoding="utf-8")
    _config_hash = hashlib.sha256(raw.encode()).hexdigest()[:16]
    _config = yaml.safe_load(raw)


def get_config() -> dict[str, Any]:
    if _config is None:
        _load()
    return _config  # type: ignore[return-value]


def get_config_hash() -> str:
    if not _config_hash:
        _load()
    return _config_hash


def get_portfolio_config(name: str) -> dict[str, Any]:
    cfg = get_config()
    portfolios = cfg.get("portfolios", {})
    if name not in portfolios:
        valid = list(portfolios.keys())
        raise KeyError(f"Unknown portfolio {name!r}. Valid names: {valid}")
    return portfolios[name]


def get_engine_weights(portfolio: str | None = None) -> dict[str, float]:
    """
    Return the fundamental/technical/quantitative weight triple.
    If portfolio is None, returns the overall (portfolio-agnostic) weights.
    """
    cfg = get_config()
    if portfolio is None:
        w = cfg["weights"]["overall"]
    else:
        w = get_portfolio_config(portfolio)["weights"]
    return {k: float(v) for k, v in w.items()}


def get_cache_config() -> dict[str, Any]:
    return get_config()["data"]["cache"]


def reload() -> None:
    """Force a reload of config from disk (useful in tests)."""
    global _config, _config_hash, _config_path
    _config = None
    _config_hash = ""
    _config_path = None
    _load()
