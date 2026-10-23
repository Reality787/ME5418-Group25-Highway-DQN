"""兼容签名及原子检查点：只加载本项目自产、可信文件，区分推理和回合边界恢复。"""
from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import tempfile

import torch

import config

RUNTIME_FILES = (
    "config.py", "environment.py", "project_environment.py", "joint_actions.py",
    "observation.py", "reward.py", "network.py", "agent.py", "replay_buffer.py",
    "scenarios.py", "metrics.py", "validation.py", "checkpoints.py", "train.py",
    # 网络委托的源码也必须参加签名，不能只检查薄适配器。
    "highway_m1/__init__.py", "highway_m1/schema.py",
    "highway_m2/__init__.py", "highway_m2/network.py", "legacy_checkpoint.py",
)


def compatibility_record() -> dict:
    """核对物理/观测/动作/奖励及实际源码，避免旧模型静默套用新环境。"""
    from environment import make_env
    from highway_env.vehicle.controller import ControlledVehicle
    from highway_env.vehicle.kinematics import Vehicle
    from highway_env.envs.common.abstract import AbstractEnv
    root = Path(__file__).resolve().parent
    env = make_env(render_mode=None, invalid_action_mode="raise")
    try:
        env.reset(seed=0)
        control = env.controller.get_low_level_parameters()
    finally:
        env.close()
    source_hashes = {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                     for name in RUNTIME_FILES}
    library_hashes = {cls.__name__: hashlib.sha256(Path(inspect.getfile(cls)).read_bytes()).hexdigest()
                      for cls in (ControlledVehicle, Vehicle, AbstractEnv)}
    fixed = {key: value for key, value in vars(config).items() if key.isupper()}
    # JSON规范化让tuple、整数动作键等在存取后保持相同表示。
    return json.loads(json.dumps({
        "format_version": 1, "config": fixed, "control": control,
        "source_hashes": source_hashes, "library_source_hashes": library_hashes,
        "python": platform.python_version(),
        "dependencies": {name: importlib.metadata.version(name)
                         for name in ("torch", "numpy", "gymnasium", "highway-env", "pygame-ce")},
    }, sort_keys=True))


def signature(record: dict) -> str:
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name+".", suffix=".tmp", delete=False) as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = stream.name
    os.replace(temporary, path)


def atomic_save(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name+".",
                                     suffix=".tmp", delete=False) as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = stream.name
    os.replace(temporary, path)


def load_checked(path, expected: dict, *, kind: str) -> dict:
    """weights_only限制反序列化；仍只允许用户信任的本项目输出。"""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("kind") != kind:
        raise ValueError(f"检查点类型不匹配，需要 {kind}")
    if payload.get("signature") != signature(expected) or payload.get("compatibility") != expected:
        raise ValueError("检查点与当前特征/归一化/动作/奖励/动力学/源码/依赖不兼容；请使用匹配版本或新run")
    return payload


def inference_payload(agent, compatibility: dict, run_config: dict, env_steps: int,
                      validation: dict | None = None) -> dict:
    return {"kind": "dqn_inference", "signature": signature(compatibility),
            "compatibility": compatibility, "run_config": run_config,
            "env_steps": env_steps, "gradient_steps": agent.gradient_steps,
            "online": agent.online.state_dict(), "validation": validation}


def load_inference(path, agent, compatibility: dict) -> dict:
    payload = load_checked(path, compatibility, kind="dqn_inference")
    agent.online.load_state_dict(payload["online"], strict=True)
    agent.online.eval()
    return payload
