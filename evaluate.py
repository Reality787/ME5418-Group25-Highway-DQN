"""评测命令入口；导入本模块不会加载模型或运行评测。"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """解析命令行；后续实现 DQN 与 gap-based controller 的公平对照。"""
    parser = argparse.ArgumentParser(
        description="ME5418 DQN 与基线评测入口（阶段 0：项目骨架）。",
        epilog="评测计划已记录在规格文档；具体实现待完成，本阶段不生成评测结果。",
    )
    parser.parse_args(argv)
    print("当前仅完成项目骨架，功能待实现：DQN 与基线评测。", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
