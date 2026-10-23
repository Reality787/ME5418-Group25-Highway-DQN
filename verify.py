"""Run Gymnasium's checker and a reproducible headless environment smoke test."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import tempfile

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "me5418-group25-mpl"))

import numpy as np
from gymnasium.utils.env_checker import check_env

from highway_m1 import HighwayDecisionEnv


def rollout(seed: int) -> dict:
    env = HighwayDecisionEnv(render_mode="rgb_array")
    observations = []
    rewards = []
    actions = []
    rng = np.random.default_rng(seed)
    try:
        observation, info = env.reset(seed=seed)
        observations.append(observation.copy())
        terminated = truncated = False
        while not (terminated or truncated):
            action = int(rng.choice(np.flatnonzero(info["action_mask"])))
            observation, reward, terminated, truncated, info = env.step(action)
            assert observation.shape == (34,), "Observation dimension changed"
            assert env.observation_space.contains(observation), "Observation outside declared bounds"
            assert np.isfinite(reward), "Reward must be finite"
            assert len(info["action_mask"]) == 9 and info["action_mask"][4], "Invalid mask"
            observations.append(observation.copy())
            actions.append(action)
            rewards.append(float(reward))
        frame = env.render()
        assert frame.ndim == 3 and frame.shape[2] == 3, "RGB rendering failed"
        assert frame.dtype == np.uint8, "Renderer must produce uint8 RGB"
        assert np.std(frame) > 1, "Renderer returned an empty/constant image"
        digest = hashlib.sha256()
        digest.update(np.asarray(observations, dtype=np.float32).tobytes())
        digest.update(np.asarray(rewards, dtype=np.float64).tobytes())
        digest.update(np.asarray(actions, dtype=np.int64).tobytes())
        return {"seed": seed, "steps": len(rewards), "total_reward": sum(rewards),
                "terminated": bool(terminated), "truncated": bool(truncated),
                "crashed": bool(info["crashed"]), "rgb_shape": list(frame.shape),
                "trajectory_sha256": digest.hexdigest()}
    finally:
        env.close()


def package_versions(names) -> dict:
    """只修正诊断输出：可选依赖未安装时记录 None，不伪称该功能已验证。"""
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=25)
    parser.add_argument("--output", type=Path, default=Path("artifacts/verification.json"))
    args = parser.parse_args()
    env = HighwayDecisionEnv()
    try:
        # No registry spec exists for this direct custom class; rendering is
        # independently checked in the rollout below.
        check_env(env, skip_render_check=True)
    finally:
        env.close()
    first, second = rollout(args.seed), rollout(args.seed)
    assert first == second, "The same seed and actions produced different trajectories"
    packages = ("highway-env", "gymnasium", "numpy", "pygame-ce", "matplotlib",
                "Pillow", "imageio", "imageio-ffmpeg", "pytest")
    result = {"gymnasium_checker": "passed", "same_seed_reproducibility": "passed",
              "python": platform.python_version(), "platform": platform.platform(),
              "dependencies": package_versions(packages),
              "rollout": first}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
