"""从真实JSONL记录绘制训练/验证曲线；不运行环境或补造缺失实验数据。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def records(path):
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    # 恢复可能保留旧尾段审计行；同一环境步使用最后一个segment的记录。
    return sorted({row["env_steps"]: row for row in rows}.values(), key=lambda row: row["env_steps"])


def main(argv=None):
    parser = argparse.ArgumentParser(description="绘制真实DQN数值和物理验证曲线，不执行训练/测试")
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for run in args.runs:
        metadata = json.loads((run/"config.json").read_text())["run_config"]
        label = f'{metadata["mode"]} seed {metadata["seed"]}'
        updates = records(run/"updates.jsonl")
        validation = records(run/"validation.jsonl")
        if not validation:
            raise ValueError(f"{run}没有实际验证记录，不能生成虚假曲线")
        axes[0, 0].plot([r["env_steps"] for r in updates], [r["loss"] for r in updates], label=label)
        for axis, key in ((axes[0, 1], "score"), (axes[0, 2], "failure_rate"),
                          (axes[1, 0], "mean_normalized_progress"),
                          (axes[1, 1], "mean_stopped_fraction"), (axes[1, 2], "mean_legal_max_q")):
            axis.plot([r["env_steps"] for r in validation], [r[key] for r in validation],
                      marker="o", markersize=3, label=label)
    titles = ["Training SmoothL1 loss (logged updates)", "Validation selection score",
              "Validation failure rate", "Validation distance / 1112 m",
              "Validation stopped-time fraction", "Validation mean max legal Q"]
    for axis, title in zip(axes.flat, titles):
        axis.set_title(title)
        axis.set_xlabel("Environment decision steps")
        axis.grid(alpha=.25)
        axis.legend(fontsize=8)
    axes[0, 2].set_ylim(-.03, 1.03)
    axes[1, 1].set_ylim(-.03, 1.03)
    fig.suptitle("ME5418 ordinary masked DQN — fixed validation, not final-test", fontsize=14)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(f"已从实际记录保存曲线：{args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
