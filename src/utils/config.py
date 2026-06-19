"""Project config loader — OmegaConf + .env, single entry point for all settings.

WHAT: load_config() reads configs/base.yaml (optionally merged with an
experiment override file) and resolves ${oc.env:VAR} placeholders against
environment variables loaded from .env.
WHY: CLAUDE.md mandates "no hardcoded values in src/" and "config comes from
configs/base.yaml via Hydra". We use plain OmegaConf here (not the
@hydra.main decorator) because pipeline scripts are invoked directly via
argparse — but base.yaml and configs/experiments/* still follow Hydra's
`defaults:` composition convention for when full Hydra is wired in later.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

# Resolve repo root relative to this file: src/utils/config.py -> repo root is 2 dirs up
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "base.yaml"


def load_config(
    config_path: str | Path = _DEFAULT_CONFIG,
    overrides: str | Path | None = None,
) -> DictConfig:
    """Load base config (and optional experiment override), resolve env interpolation.

    Args:
        config_path: path to base config YAML. Defaults to configs/base.yaml.
        overrides: optional path to an experiment override YAML
            (e.g. configs/experiments/chunking_ablation.yaml), merged on top
            of the base config.

    Returns:
        Resolved DictConfig — ${oc.env:VAR} placeholders are substituted with
        real environment variable values from .env / the shell.

    WHY: centralizes config loading so every module and scripts/*.py get
    identical, fully-resolved settings instead of each reimplementing
    dotenv + OmegaConf wiring.
    """
    # LEARN: load .env BEFORE OmegaConf.resolve so ${oc.env:SEC_USER_AGENT}
    # can see it via os.environ. Load from repo root regardless of cwd.
    load_dotenv(_REPO_ROOT / ".env")

    cfg = OmegaConf.load(config_path)

    if overrides is not None:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(overrides))

    OmegaConf.resolve(cfg)
    return cfg  # type: ignore[return-value]
