"""按明确文件清单打包 Learning Agent 作业；不训练、不包含旧实验产物。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile


SOURCE_FILES = (
    "config.py", "environment.py", "project_environment.py", "joint_actions.py",
    "observation.py", "reward.py", "network.py", "agent.py", "replay_buffer.py",
    "train.py", "scenarios.py", "metrics.py", "validation.py", "checkpoints.py",
    "demo.py", "network_demo.py", "validate_env.py", "plot_training.py",
    "baseline.py", "evaluate.py", "environment.yml", "requirements.txt",
    "requirements-network.txt",
)
ARCHIVE_ROOT = "ME5418_Group25_LearningAgent_CR"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--commit", help="最终包必须提供已审核的 Git commit；省略只生成预检包")
    parser.add_argument("--verification", type=Path, help="实际解压检查的 JSON 记录")
    args = parser.parse_args()
    root = args.project.resolve()
    if args.stage.exists() or args.output.exists():
        raise FileExistsError("暂存目录或 ZIP 已存在；请使用新路径，不覆盖既有交付物")
    sources = {name: name for name in SOURCE_FILES}
    sources.update({str(path.relative_to(root)): str(path.relative_to(root))
                    for path in sorted((root / "tests").glob("test_*.py"))})
    sources["README.md"] = "docs/learning_agent_submission_README.md"
    sources["Learning_Agent_CR_Report.md"] = "docs/learning_agent_cr_report.md"
    if not any(name.startswith("tests/") for name in sources):
        raise ValueError("缺少测试文件，拒绝打包")
    for archive_name, source_name in sources.items():
        path = root / source_name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"缺少源文件或发现符号链接：{source_name}")
        if args.commit:
            # 每个源文件必须与指定提交逐字节一致，避免未确认的混合版本。
            saved = subprocess.check_output(["git", "show", f"{args.commit}:{source_name}"], cwd=root)
            if saved != path.read_bytes():
                raise ValueError(f"源文件与指定提交不一致：{source_name}")
    if args.commit:
        args.commit = subprocess.check_output(
            ["git", "rev-parse", f"{args.commit}^{{commit}}"], cwd=root, text=True
        ).strip()
    if not args.report.is_file() or args.report.is_symlink():
        raise ValueError("缺少实际 PDF 报告")
    args.stage.mkdir(parents=True)
    file_records = {}
    for archive_name, source_name in sources.items():
        destination = args.stage / archive_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / source_name, destination)
        file_records[archive_name] = {
            "source": source_name,
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        }
    pdf_name = "Learning_Agent_CR_Report.pdf"
    shutil.copyfile(args.report, args.stage / pdf_name)
    file_records[pdf_name] = {
        "source": "generated from docs/learning_agent_cr_report.md using tools/build_learning_agent_report.py",
        "sha256": hashlib.sha256(args.report.read_bytes()).hexdigest(),
    }
    record = {
        "assignment": "NUS ME5418 Learning Agent C&R", "group": 25,
        "git_commit": args.commit, "preview_only": args.commit is None,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": file_records,
        "verification": json.loads(args.verification.read_text()) if args.verification else None,
        "scope": "Existing dependencies used; no clean Conda installation; no formal retraining or final-test",
    }
    (args.stage / "SUBMISSION_RECORD.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(args.stage.rglob("*")):
            if path.is_file():
                archive.write(path, f"{ARCHIVE_ROOT}/{path.relative_to(args.stage).as_posix()}")
    with zipfile.ZipFile(args.output) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("ZIP 校验失败")
        print(json.dumps({"archive": str(args.output), "file_count": len(archive.namelist()),
                          "bytes": args.output.stat().st_size, "commit": args.commit}, ensure_ascii=False))


if __name__ == "__main__":
    main()
