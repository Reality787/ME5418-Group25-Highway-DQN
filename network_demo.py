"""阶段 2：Q 网络数值检查、state_dict 往返与真实环境 reset 输入对接。

这里的合成监督优化仅验证梯度和参数更新，不是 DQN 或驾驶训练。
查看 --help 时只使用标准库；导入本模块不会创建网络或仿真环境。
"""

from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path
import sys
import tempfile


def nonnegative_int(value: str) -> int:
    """命令行随机种子必须是非负整数。"""
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("必须是非负整数")
    return number


def positive_int(value: str) -> int:
    """合成检查执行有限次更新，不接受零或负数。"""
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("必须是正整数")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="阶段 2：网络数值验证，不是驾驶训练结果。默认同时验证正式环境 reset 输入。"
    )
    parser.add_argument("--seed", type=nonnegative_int, default=0, help="固定随机种子，默认 0")
    parser.add_argument(
        "--steps", type=positive_int, default=30, help="合成监督检查的有限优化次数，默认 30"
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="可选 state_dict 保存路径；拒绝覆盖已有文件，默认使用并清理临时文件",
    )
    parser.add_argument(
        "--skip-environment", action="store_true",
        help="仅检查网络，并明确标记正式环境输入集成未验证",
    )
    return parser


def verify_environment_input(seed: int) -> bool:
    """只读取真实正式环境 reset 的观测，不推进仿真，不执行网络策略。"""
    import numpy as np
    import torch

    import config
    from network import DQNNetwork

    # 缺失真实环境依赖时只报告未验证；实现错误必须暴露，不能以跳过掩盖。
    missing = []
    for distribution in ("gymnasium", "highway-env", "numpy"):
        try:
            importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            missing.append(distribution)
    if missing:
        print(f"正式环境输入集成：未验证，缺少依赖 {', '.join(missing)}")
        return False

    from environment import make_env

    env = make_env(render_mode=None)
    try:
        observation, info = env.reset(seed=seed)
        if not isinstance(observation, np.ndarray):
            raise TypeError("正式环境 reset 必须返回 numpy.ndarray 观测")
        if observation.shape != (config.NETWORK_INPUT_DIM,) or observation.dtype != np.float32:
            raise ValueError(f"正式环境观测不符合 float32[34]：{observation.shape}, {observation.dtype}")
        if not np.isfinite(observation).all() or not env.observation_space.contains(observation):
            raise ValueError("正式环境观测不属于声明空间或含非有限值")
        if env.action_space.n != config.ACTION_COUNT:
            raise ValueError("必须使用正式九动作环境，不能用 stock 环境代替")
        mask = info["action_mask"]
        if mask.shape != (config.ACTION_COUNT,) or mask.dtype != np.bool_:
            raise ValueError("正式环境结构动作掩码必须为 bool[9]")

        # 另建随机初始化网络；不能把合成回归优化过的网络说成初始网络。
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            untrained_network = DQNNetwork().cpu().eval()
        with torch.no_grad():
            q_values = untrained_network(torch.from_numpy(observation))
        if q_values.shape != (config.ACTION_COUNT,) or not torch.isfinite(q_values).all():
            raise ValueError("真实观测的网络输出应为有限的 9 个 Q 值")
        print("正式环境输入集成：通过，make_env().reset() -> float32[34] -> Q[9]")
        print("以下是单独随机初始化、尚未训练网络的输出，不代表有效驾驶策略：")
        for action, q_value in enumerate(q_values.tolist()):
            print(f"  {action}: {config.ACTION_NAMES[action]:<14} Q={q_value:+.8f}")
        print(f"结构动作掩码（独立诊断，未用于修改 Q 值）：{mask.tolist()}")
        print("本次环境输入检查只调用 reset，未调用 step，未选择或执行驾驶动作。")
        return True
    finally:
        env.close()


def verify_checkpoint(model, sample, checkpoint: Path) -> float:
    """只保存参数；显式 CPU 和 weights_only 安全加载，拒绝覆盖已有文件。"""
    import torch

    from network import DQNNetwork

    # 用独占创建避免“先 exists 再 save”的并发覆盖窗口。
    with checkpoint.open("xb") as output_file:
        torch.save(model.state_dict(), output_file)
    restored = DQNNetwork().cpu()
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    restored.load_state_dict(state_dict, strict=True)
    model.eval()
    restored.eval()
    with torch.no_grad():
        expected = model(sample)
        actual = restored(sample)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    difference = float(torch.max(torch.abs(actual - expected)).item())
    print(f"state_dict 保存及 CPU 加载：通过；同一输入输出最大差={difference:.9g}")
    return difference


def run_demo(args: argparse.Namespace) -> None:
    import torch

    import config
    from network import DQNNetwork

    # 仅本演示进程的小矩阵计算使用一个 CPU 线程，避免线程调度主导运行时间。
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    model = DQNNetwork().cpu()
    print("网络数值验证，不是驾驶训练结果")
    print(f"解释器：{sys.executable}")
    print(f"Python：{sys.version.split()[0]}；torch：{torch.__version__}；设备：cpu；dtype：float32")
    print(f"随机种子：{args.seed}；CPU 计算线程：{torch.get_num_threads()}")
    print(model)
    count = sum(parameter.numel() for parameter in model.parameters())
    # 独立手算固定规格，包含三层 bias，不复用模型参数列表生成预期值。
    expected_count = (34 * 128 + 128) + (128 * 128 + 128) + (128 * 9 + 9)
    if count != expected_count or count != 22153:
        raise AssertionError(f"参数量不符：实际 {count}，独立公式 {expected_count}")
    print(f"参数量：{count}；独立核对：(34*128+128)+(128*128+128)+(128*9+9)={expected_count}")

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    inputs = torch.randn(32, config.NETWORK_INPUT_DIM, generator=generator, dtype=torch.float32)
    # 固定的小型监督目标没有驾驶含义，不来自奖励、环境回合或 TD 更新。
    targets = 0.25 * inputs[:, :config.NETWORK_OUTPUT_DIM] + 0.1
    with torch.no_grad():
        single_output = model(inputs[0])
        batch_output = model(inputs)
    print(f"单输入 {tuple(inputs[0].shape)} -> {tuple(single_output.shape)}")
    print(f"批量输入 {tuple(inputs.shape)} -> {tuple(batch_output.shape)}")
    if not torch.isfinite(single_output).all() or not torch.isfinite(batch_output).all():
        raise ValueError("网络初始输出含 NaN/Inf")

    # Adam、1e-3、32 条样本只用于本次数值实验，不是尚未确定的 DQN 训练超参数。
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_function = torch.nn.MSELoss()
    before_parameters = [parameter.detach().clone() for parameter in model.parameters()]
    with torch.no_grad():
        before_loss = float(loss_function(model(inputs), targets).item())
    model.train()
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(model(inputs), targets)
        if not torch.isfinite(loss):
            raise ValueError("合成监督检查 loss 含 NaN/Inf")
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise ValueError("反向传播未产生有限梯度")
        optimizer.step()
    model.eval()
    with torch.no_grad():
        after_loss = float(loss_function(model(inputs), targets).item())
        parameter_change = float(torch.sqrt(sum(
            torch.sum((parameter - previous) ** 2)
            for parameter, previous in zip(model.parameters(), before_parameters)
        )).item())
    if not parameter_change > 0.0:
        raise AssertionError("优化后参数未发生有效变化")
    if not after_loss < before_loss:
        raise AssertionError("固定合成任务误差没有降低，请检查数值或步数设置")
    print(f"合成监督检查：Adam lr=0.001，样本数=32，更新次数={args.steps}（非 DQN 超参数）")
    print(f"MSE：{before_loss:.9f} -> {after_loss:.9f}；参数差 L2={parameter_change:.9f}")

    if args.checkpoint is None:
        with tempfile.TemporaryDirectory(prefix="me5418-network-") as temporary_directory:
            checkpoint = Path(temporary_directory) / "q_network_state.pt"
            verify_checkpoint(model, inputs, checkpoint)
            print("默认检查点使用临时文件，本次读回验证后自动清理。")
    else:
        checkpoint = args.checkpoint.expanduser()
        verify_checkpoint(model, inputs, checkpoint)
        print(f"保留 state_dict 文件：{checkpoint.resolve()}")
    if args.skip_environment:
        print("正式环境输入集成：未验证（用户指定 --skip-environment）。")
    else:
        verify_environment_input(args.seed)
    print("阶段 2 数值检查结束；网络已验证，但 DQN 和驾驶训练尚未完成。")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_demo(args)
    except ModuleNotFoundError as error:
        print(f"检查未完成：缺少依赖 {error.name}；请使用已配置的项目解释器。", file=sys.stderr)
        return 1
    except (OSError, ValueError, TypeError, AssertionError, RuntimeError) as error:
        print(f"网络验证失败：{error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
