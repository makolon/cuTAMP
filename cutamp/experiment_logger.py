# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import json
import logging
from pathlib import Path

import omegaconf
import torch
import yaml

from cutamp.config import TAMPConfiguration
from cutamp.envs import TAMPEnvironment
from cutamp.envs.utils import get_env_dict

_log = logging.getLogger(__name__)


class _OmegaConfEncoder(json.JSONEncoder):
    """Encode OmegaConf objects for JSON serialization."""

    def default(self, obj):
        if isinstance(obj, (omegaconf.ListConfig, omegaconf.DictConfig)):
            return omegaconf.OmegaConf.to_container(obj, resolve=True)
        return super().default(obj)


class ExperimentLogger:
    """Simple experiment logger."""

    def __init__(self, name: str, config: TAMPConfiguration):
        self.exp_dir = Path(config.experiment_root) / name
        if self.exp_dir.exists():
            _log.warning(f"Experiment directory {self.exp_dir} already exists")
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        _log.info(f"Logging experiment to {self.exp_dir}")

        # Save the config
        with open(self.exp_dir / "config.yml", "w") as f:
            yaml.dump(config.__dict__, f, sort_keys=False)

    @staticmethod
    def _dedupe_path(path: Path) -> Path:
        """Return a unique path by appending a numeric suffix when needed."""
        if not path.exists():
            return path

        stem = path.stem
        suffix = path.suffix
        parent = path.parent
        idx = 1
        candidate = parent / f"{stem}_{idx:03d}{suffix}"
        while candidate.exists():
            idx += 1
            candidate = parent / f"{stem}_{idx:03d}{suffix}"
        _log.warning("File %s already exists, writing to %s", path, candidate)
        return candidate

    def log_dict(self, name: str, data: dict) -> Path:
        path = self._dedupe_path(self.exp_dir / f"{name}.json")
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save as JSON, YAML is too slow to load
        with open(path, "w") as f:
            json.dump(data, f, indent=2, cls=_OmegaConfEncoder)
        _log.info(f"Logged {name} to {path}")
        return path

    def save_env(self, env: TAMPEnvironment, filename: str = "tamp_env.yml") -> Path:
        """Save the TAMP environment as a YAML file."""
        env_dict = get_env_dict(env)
        env_path = self.exp_dir / filename
        with open(env_path, "w") as f:
            yaml.dump(env_dict, f, sort_keys=False)
        _log.info(f"Saved environment to {env_path}")
        return env_path

    def log_torch(self, name: str, data: dict) -> Path:
        """Save a dictionary of tensors for later warm-start retrieval."""
        path = self._dedupe_path(self.exp_dir / f"{name}.pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, path)
        _log.info(f"Logged tensor payload {name} to {path}")
        return path
