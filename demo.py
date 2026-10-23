"""Record a functional environment demonstration; no learning is performed.

The scripted policy deliberately exercises joint lane/speed actions. It is not a
traffic-aware controller and its return is not a learned-policy benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def json_value(value: Any) -> Any:
    """Convert NumPy values in logged information into ordinary JSON values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    return value


def load_font(size: int) -> ImageFont.ImageFont:
    for filename in (
        "C:/Windows/Fonts/consola.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ):
        try:
            return ImageFont.truetype(filename, size=size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def overlay(frame: np.ndarray, info: dict, reward: float, policy: str,
            seed: int, speed_targets: Any) -> np.ndarray:
    """Append the action legend and measured state without altering the scene."""
    scene = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
    width, height = scene.size
    result = Image.new("RGB", (width, height + 248), "#101725")
    result.paste(scene, (0, 0))
    draw = ImageDraw.Draw(result)
    font = load_font(17)
    small = load_font(15)
    text_colour = "#e5edf7"
    action = info.get("executed_action")
    action_text = "reset" if action is None else str(action)
    requested = info.get("requested_action")
    requested_text = "reset" if requested is None else str(requested)
    elapsed = float(info.get("elapsed_time", 0))
    draw.text((16, height + 10),
              f"Group 25 | {policy} | seed {seed} | t={elapsed:.1f}s | "
              f"req={requested_text} exec={action_text} "
              f"invalid={info.get('invalid_action', False)} | r={reward:+.3f}",
              font=font, fill=text_colour)
    draw.text((16, height + 38),
              f"speed={info.get('ego_speed', 0):.2f} m/s  "
              f"target={info.get('target_speed', 0):.1f} m/s  "
              f"lane={info.get('current_lane', '?')} -> "
              f"{info.get('target_lane', '?')}  "
              f"changing={info.get('manoeuvre_in_progress', False)}",
              font=font, fill=text_colour)
    draw.text((16, height + 66),
              f"target speeds [m/s]: {list(speed_targets)} | "
              f"sim t={info.get('simulation_time_s', 0):.2f}s | "
              f"crashed={info.get('crashed', False)} | "
              f"on_road={info.get('on_road', True)}",
              font=small, fill="#a6b7ce")
    draw.text((16, height + 91),
              "Next-step availability: OK/BLOCKED = structural mask; blue = last executed action",
              font=small, fill="#a6b7ce")
    mask = np.asarray(info.get("action_mask", np.ones(9)), dtype=bool)
    lateral_labels = ("Left", "Keep", "Right")
    speed_labels = ("Slower", "Hold", "Faster")
    column_width = width // 3
    for index in range(9):
        row, column = divmod(index, 3)
        left = column * column_width + 12
        top = height + 120 + row * 29
        if action == index:
            draw.rounded_rectangle((left, top - 2, left + column_width - 20,
                                    top + 24), radius=4, fill="#294b75")
        status = "OK" if mask[index] else "BLOCKED"
        label = f"{index}: {lateral_labels[row]}/{speed_labels[column]} [{status}]"
        draw.text((left + 4, top), label, font=small,
                  fill=text_colour if mask[index] else "#8590a0")
    draw.text((16, height + 218),
              "Scripted/random policies are untrained; this video demonstrates API and dynamics.",
              font=small, fill="#f4c57b")
    return np.asarray(result)


class DemonstrationPolicy:
    """Deterministic scripted, uniformly random valid, or hold-only actions."""

    def __init__(self, name: str, seed: int):
        self.name = name
        self.rng = np.random.default_rng(seed)
        self.stage = 0

    def __call__(self, info: dict) -> int:
        mask = np.asarray(info["action_mask"], dtype=bool)
        if self.name == "random":
            return int(self.rng.choice(np.flatnonzero(mask)))
        if self.name == "hold":
            return 4
        # Keep/Hold preserves an unfinished lane-change target in this wrapper.
        if info.get("manoeuvre_in_progress", False):
            return 4
        elapsed = float(info.get("elapsed_time", 0))
        scheduled = ((0.0, 5), (4.0, 0), (10.0, 8))
        if self.stage < len(scheduled):
            trigger, requested = scheduled[self.stage]
            if elapsed + 1e-9 >= trigger and mask[requested]:
                self.stage += 1
                return requested
        return 4


def record_episode(args: argparse.Namespace, episode: int, destination: Path) -> dict:
    from highway_m1 import HighwayDecisionEnv, OBSERVATION_NAMES, SPEED_TARGETS

    seed = args.seed + episode
    destination.mkdir(parents=True, exist_ok=True)
    env = HighwayDecisionEnv(render_mode="rgb_array")
    frames: list[np.ndarray] = []
    rows: list[dict] = []
    total_reward = 0.0
    terminated = truncated = False
    interrupted = False
    interruption_reason = None
    policy = DemonstrationPolicy(args.policy, seed)
    display = clock = None
    try:
        observation, info = env.reset(seed=seed)
        fps = int(env.config.get("policy_frequency", 5))
        first = overlay(env.render(), info, 0.0, args.policy, seed, SPEED_TARGETS)
        frames.append(first)
        Image.fromarray(first).save(destination / "initial.png")
        if args.human:
            import pygame
            pygame.display.init()
            display = pygame.display.set_mode((first.shape[1], first.shape[0]))
            pygame.display.set_caption("ME5418 Group 25 - functional demo")
            clock = pygame.time.Clock()
        while not (terminated or truncated):
            action = policy(info)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            row = {
                "step": info.get("step_count", len(rows) + 1),
                "elapsed_time": info["elapsed_time"],
                "simulation_time_s": info["simulation_time_s"],
                "simulation_steps": info["simulation_steps"],
                "requested_action": action,
                "executed_action": info.get("executed_action", action),
                "reward": float(reward),
                "cumulative_reward": total_reward,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "ego_speed": info["ego_speed"],
                "target_speed": info["target_speed"],
                "current_lane": info["current_lane"],
                "target_lane": info["target_lane"],
                "manoeuvre_in_progress": info["manoeuvre_in_progress"],
                "crashed": info["crashed"],
                "on_road": info["on_road"],
                "progress_m": info["progress_m"],
            }
            row.update({"reward_" + key: value
                        for key, value in info["reward_components"].items()})
            row.update({"obs_" + str(index).zfill(2) + "_" + name: float(value)
                        for index, (name, value) in enumerate(
                            zip(OBSERVATION_NAMES, observation, strict=True))})
            row["next_action_mask"] = "".join("1" if valid else "0"
                                               for valid in info["action_mask"])
            rows.append(row)
            frame = overlay(env.render(), info, float(reward), args.policy,
                            seed, SPEED_TARGETS)
            frames.append(frame)
            if display is not None:
                import pygame
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        interrupted = True
                        interruption_reason = "Demo window closed by user"
                if interrupted:
                    break
                surface = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
                display.blit(surface, (0, 0))
                pygame.display.flip()
                clock.tick(fps)
        Image.fromarray(frames[-1]).save(destination / "final.png")
        with (destination / "trajectory.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        import imageio.v2 as imageio
        video_name = "demo.gif" if args.gif else "demo.mp4"
        video_warning = None
        if args.gif:
            imageio.mimsave(destination / video_name, frames,
                            duration=1000 / fps, loop=0)
        else:
            try:
                with imageio.get_writer(destination / video_name, fps=fps,
                                        codec="libx264", macro_block_size=1) as writer:
                    for frame in frames:
                        writer.append_data(frame)
            except (OSError, RuntimeError, ValueError) as error:
                video_warning = f"MP4 unavailable ({error}); wrote GIF instead."
                partial = destination / video_name
                if partial.exists():
                    partial.unlink()
                video_name = "demo.gif"
                imageio.mimsave(destination / video_name, frames,
                                duration=1000 / fps, loop=0)
        summary = {
            "policy": args.policy,
            "purpose": "Untrained functional demonstration, not performance evaluation",
            "seed": seed,
            "steps": len(rows),
            "elapsed_time_s": info["elapsed_time"],
            "simulation_time_s": info["simulation_time_s"],
            "simulation_steps": info["simulation_steps"],
            "total_reward": total_reward,
            "progress_m": info["progress_m"],
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "crashed": info["crashed"],
            "on_road": info["on_road"],
            "interrupted": interrupted,
            "interruption_reason": interruption_reason,
            "final_info": json_value(info),
            "requested_action_counts": {str(action): sum(row["requested_action"] == action
                                                         for row in rows)
                                        for action in range(9)},
            "video": video_name,
            "video_fps": fps,
            "video_frames": len(frames),
            "video_warning": video_warning,
            "config": json_value(env.config),
        }
        (destination / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        print(f"Episode {episode + 1}: seed={seed}, steps={len(rows)}, "
              f"return={total_reward:.3f}, crashed={info['crashed']}, "
              f"interrupted={interrupted}, video={destination / video_name}")
        if video_warning:
            print(video_warning)
        return summary
    finally:
        env.close()
        if display is not None:
            import pygame
            pygame.display.quit()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("scripted", "random", "hold"), default="scripted")
    parser.add_argument("--seed", type=int, default=25)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("artifacts/demo"))
    parser.add_argument("--human", action="store_true", help="Show the recorded RGB frames in a window")
    parser.add_argument("--gif", action="store_true", help="Write an animated GIF instead of an MP4")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be at least 1")
    if not args.human:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "me5418-group25-mpl"))
    args.output.mkdir(parents=True, exist_ok=True)
    for episode in range(args.episodes):
        destination = (args.output if args.episodes == 1 else
                       args.output / f"episode_{episode + 1:03d}")
        record_episode(args, episode, destination)


if __name__ == "__main__":
    main()
