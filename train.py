"""普通DQN的有界单环境CPU训练；导入不创建模型、仿真或启动运行。"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

import config


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("必须是正整数")
    return number


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("必须是非负整数")
    return number


def epsilon_at(env_steps: int, budget: int, settings: dict | None = None) -> float:
    """epsilon按累计环境决策步线性变化，不使用梯度步或物理帧计数。"""
    settings = config.DQN_CONFIG if settings is None else settings
    fraction = min(1.0, max(0.0, env_steps / (budget * settings["epsilon_decay_fraction"])))
    return settings["epsilon_start"] + fraction * (settings["epsilon_end"] - settings["epsilon_start"])


def append_json(path: Path, record: dict):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()


def run_training(*, mode: str, seed: int, steps: int, run_dir: Path,
                 device: str = "cpu", resume: Path | None = None,
                 protocol: Path | None = None, stop_after_boundary: int | None = None,
                 settings_override: dict | None = None) -> dict:
    """正式CLI冻结参数；settings_override仅供小型恢复测试，不暴露为正式运行调参开关。"""
    import torch
    from agent import DQNAgent
    from checkpoints import (atomic_json, atomic_save, compatibility_record, inference_payload,
                             load_checked, signature)
    from environment import make_env
    from metrics import EpisodeMetrics, is_better
    from scenarios import ScenarioSampler, reset_scenario
    from validation import validate_agent

    if mode not in ("smoke", "pilot", "formal") or device != "cpu":
        raise ValueError("只支持smoke/pilot/formal及CPU单环境")
    if steps <= 0 or seed < 0:
        raise ValueError("steps必须为正整数，seed必须非负")
    torch.set_num_threads(config.TRAINING_CONFIG["torch_threads"])
    compatibility = compatibility_record()
    dqn_settings = {**config.DQN_CONFIG, **(settings_override or {})}
    protocol_data = None
    if mode == "formal":
        if protocol is None or settings_override:
            raise ValueError("正式训练必须提供冻结协议，且不能覆盖超参数")
        protocol_data = json.loads(Path(protocol).read_text())
        if (protocol_data.get("status") != "frozen"
                or protocol_data.get("signature") != signature(compatibility)
                or protocol_data.get("compatibility") != compatibility
                or protocol_data.get("budget") != steps
                or seed not in protocol_data.get("seeds", [])):
            raise ValueError("正式训练与冻结协议的源码/环境/预算/种子不一致")
    run_config = {"mode": mode, "seed": seed, "steps": steps, "device": device,
                  "dqn": dqn_settings, "training": deepcopy(config.TRAINING_CONFIG),
                  "scenario": deepcopy(config.SCENARIO_CONFIG),
                  "protocol_signature": None if protocol_data is None else protocol_data["signature"]}
    # 固定JSON表示用于存取兼容比较。
    run_config = json.loads(json.dumps(run_config))
    run_dir = Path(run_dir).resolve()
    if resume is None:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError("run-dir已有文件；请使用新目录或显式--resume")
        run_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(run_dir/"config.json", {"run_config": run_config, "compatibility": compatibility,
            "signature": signature(compatibility), "interpreter": sys.executable,
            "initialization": "fresh independent online/target/optimizer/replay/RNG"})
    else:
        if not run_dir.is_dir():
            raise ValueError("恢复需要已有run-dir")
    agent = DQNAgent(seed=seed, device=device, settings=dqn_settings)
    split = "train" if mode == "formal" else "development"
    sampler = ScenarioSampler(split=split, seed=seed)
    state = {"env_steps": 0, "episodes": 0, "physics_frames": 0, "best_record": None,
             "next_validation_step": config.TRAINING_CONFIG["validation_every"],
             "last_resume_env_steps": 0, "segment": 0}
    if resume is not None:
        payload = load_checked(resume, compatibility, kind="dqn_resume_boundary")
        if payload["run_config"] != run_config:
            raise ValueError("恢复的mode/seed/预算/epsilon方案/超参数不匹配")
        agent.load_state_dict(payload["agent"])
        sampler.load_state_dict(payload["sampler"])
        state = deepcopy(payload["training_state"])
        state["segment"] += 1
        append_json(run_dir/"events.jsonl", {"event": "resume_boundary", **state,
            "note": "仅从此完整回合边界恢复；旧日志中之后的未提交尾段保留为审计记录"})
    started = time.monotonic()
    env = make_env(render_mode=None, invalid_action_mode="raise")
    meters = None
    scenario = None
    obs = info = None
    last_validation_step = None
    update_records = []

    def checkpoint_boundary():
        state["last_resume_env_steps"] = state["env_steps"]
        atomic_save(run_dir/"resume.pt", {
            "kind": "dqn_resume_boundary", "signature": signature(compatibility),
            "compatibility": compatibility, "run_config": run_config,
            "agent": agent.state_dict(), "sampler": sampler.state_dict(),
            "training_state": deepcopy(state), "episode_boundary": True,
        })

    def validate_now():
        nonlocal last_validation_step
        record = validate_agent(agent, episodes=config.TRAINING_CONFIG["validation_episodes"],
                                seed=config.TRAINING_CONFIG["validation_seed"])
        record.update({"env_steps": state["env_steps"], "gradient_steps": agent.gradient_steps,
                       "epsilon": epsilon_at(state["env_steps"], steps, dqn_settings),
                       "segment": state["segment"]})
        append_json(run_dir/"validation.jsonl", record)
        last_validation_step = state["env_steps"]
        if is_better(record, state["best_record"]):
            state["best_record"] = deepcopy(record)
            atomic_save(run_dir/"best.pt", inference_payload(
                agent, compatibility, run_config, state["env_steps"], record))
        atomic_save(run_dir/"last.pt", inference_payload(
            agent, compatibility, run_config, state["env_steps"], record))
        summary = {key: record[key] for key in ("env_steps", "failure_rate", "collision_rate",
                   "completion_rate", "mean_normalized_progress", "score")}
        print("validation " + json.dumps(summary, ensure_ascii=False), flush=True)

    def status(label, error=None):
        data = {"status": label, "mode": mode, "seed": seed, "planned_steps": steps,
                "env_steps": state["env_steps"], "gradient_steps": agent.gradient_steps,
                "episodes": state["episodes"], "physics_frames": state["physics_frames"],
                "last_resume_env_steps": state["last_resume_env_steps"],
                "epsilon": epsilon_at(state["env_steps"], steps, dqn_settings),
                "best": state["best_record"], "elapsed_seconds_this_segment": time.monotonic()-started,
                "segment": state["segment"], "error": error,
                "resume_command": f'python train.py --mode {mode} --seed {seed} --steps {steps} '
                    f'--run-dir "{run_dir}" --resume "{run_dir / "resume.pt"}"'
                    + (f' --protocol "{Path(protocol).resolve()}"' if protocol else ""),
                "final_test_executed": False}
        atomic_json(run_dir/"status.json", data)
        return data

    try:
        if resume is None:
            validate_now()  # 预先记录初始策略；允许选模规则如实选择step0。
            checkpoint_boundary()
        status("running")
        while state["env_steps"] < steps:
            if meters is None:
                scenario = sampler.next()
                obs, info = reset_scenario(env, scenario)
                append_json(run_dir/"scenarios.jsonl", {"episode": state["episodes"],
                    "env_steps": state["env_steps"], "segment": state["segment"], **info["scenario"]})
                meters = EpisodeMetrics(env)
            epsilon = epsilon_at(state["env_steps"], steps, dqn_settings)
            action = agent.select_action(obs, info["action_mask"], epsilon=epsilon)
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            if next_info["invalid_action"] or next_info["executed_action"] != action:
                raise RuntimeError("控制接口缺陷：实际动作与所选动作不一致/触发hold回退，停止运行")
            # 先保存当前转移的最终观测和掩码，之后才可能reset；时间截断仍bootstrap。
            agent.replay.add(obs, action, reward, next_obs, terminated, truncated,
                             next_info["action_mask"])
            meters.update(env, reward, terminated, truncated, next_info)
            state["env_steps"] += 1
            dt = next_info["reward_diagnostics"]["dt"]
            state["physics_frames"] += round(dt * config.SIMULATION_FREQUENCY_HZ)
            obs, info = next_obs, next_info
            if (state["env_steps"] >= dqn_settings["learning_starts"]
                    and state["env_steps"] % dqn_settings["train_every"] == 0
                    and len(agent.replay) >= dqn_settings["batch_size"]):
                result = agent.learn()
                if result is None:
                    raise RuntimeError("达到学习条件但未执行更新")
                if agent.gradient_steps % config.TRAINING_CONFIG["log_every_updates"] == 0:
                    append_json(run_dir/"updates.jsonl", {"env_steps": state["env_steps"],
                        "epsilon": epsilon, "segment": state["segment"], **result})
                update_records.append(result["loss"])
            # 固定环境步间隔验证：独立环境/RNG，不打断当前训练回合或重置它。
            if state["env_steps"] >= state["next_validation_step"]:
                validate_now()
                while state["next_validation_step"] <= state["env_steps"]:
                    state["next_validation_step"] += config.TRAINING_CONFIG["validation_every"]
            if terminated or truncated:
                record = meters.finish()
                record.update({"env_steps": state["env_steps"], "episode": state["episodes"],
                               "scenario_id": scenario["scenario_id"], "segment": state["segment"],
                               "epsilon": epsilon, "partial": False})
                append_json(run_dir/"episodes.jsonl", record)
                state["episodes"] += 1
                meters = None
                if state["env_steps"] - state["last_resume_env_steps"] >= config.TRAINING_CONFIG["checkpoint_every"]:
                    checkpoint_boundary()
                    status("running")
                if stop_after_boundary is not None and state["env_steps"] >= stop_after_boundary:
                    checkpoint_boundary()
                    atomic_save(run_dir/"last.pt", inference_payload(agent, compatibility, run_config,
                                                                      state["env_steps"]))
                    return status("stopped_at_boundary")
                if state["env_steps"] == steps:
                    checkpoint_boundary()
            # 正常长运行定期可见进度；不依赖后台无人监管调度。
            if state["env_steps"] % config.TRAINING_CONFIG["progress_every"] == 0:
                print(f"train mode={mode} seed={seed} env_steps={state['env_steps']}/{steps} "
                      f"updates={agent.gradient_steps} epsilon={epsilon:.4f} "
                      f"mean_recent_loss={sum(update_records[-250:])/max(1,len(update_records[-250:])):.6f}",
                      flush=True)
                status("running")
        if meters is not None:
            record = meters.finish()
            record.update({"env_steps": state["env_steps"], "episode": state["episodes"],
                           "scenario_id": scenario["scenario_id"], "segment": state["segment"],
                           "partial": True, "note": "预算在回合中结束，不计为完成回合"})
            append_json(run_dir/"episodes.jsonl", record)
        if last_validation_step != state["env_steps"]:
            validate_now()
        if meters is None:
            checkpoint_boundary()
        result = status("completed")
        print(f"completed seed={seed} env_steps={state['env_steps']} updates={agent.gradient_steps}", flush=True)
        return result
    except BaseException as error:
        # 中途没有完整仿真快照：不伪称逐步精确恢复；保留最近完整边界resume。
        atomic_save(run_dir/"last.pt", inference_payload(agent, compatibility, run_config,
                                                          state["env_steps"]))
        status("interrupted" if isinstance(error, KeyboardInterrupt) else "failed", repr(error))
        raise
    finally:
        env.close()


def build_parser():
    parser = argparse.ArgumentParser(description="ME5418普通DQN：smoke/pilot/冻结formal，CPU单环境无渲染")
    parser.add_argument("--mode", choices=("smoke", "pilot", "formal"), required=True)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--seed", type=nonnegative_int, required=True)
    parser.add_argument("--steps", type=positive_int, default=None)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path, help="只接受本项目可信的回合边界resume.pt")
    parser.add_argument("--protocol", type=Path, help="formal必须使用已冻结JSON协议")
    parser.add_argument("--stop-after-boundary", type=positive_int,
                        help="达到指定步数后的首个回合边界停止，仅用于恢复验证/有界人工分段")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.steps is None:
        if args.mode == "formal":
            parser.error("formal必须显式提供冻结的--steps预算")
        args.steps = config.TRAINING_CONFIG[f"{args.mode}_steps"]
    try:
        run_training(mode=args.mode, seed=args.seed, steps=args.steps, run_dir=args.run_dir,
                     device=args.device, resume=args.resume, protocol=args.protocol,
                     stop_after_boundary=args.stop_after_boundary)
    except KeyboardInterrupt:
        print("运行已中断；查看status.json和最近完整边界resume.pt。", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"训练失败：{error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
