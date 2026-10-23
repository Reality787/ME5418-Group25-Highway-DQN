# ME5418 第一阶段：快速运行

本目录依据新版 **9-actions Proposal** 实现环境、演示和验证，尚未训练 DQN。
详细环境定义见 [SPECIFICATION.md](SPECIFICATION.md)，完整说明见 [README.md](README.md)。

所有命令均在**解压后的项目根目录**执行，即同时包含 `environment.yml`、
`verify.py` 和 `demo.py` 的目录。已验证的运行环境为 **Windows + Python 3.12.14**；
`environment.yml` 已固定该 Python 版本及直接依赖版本，**Ubuntu 尚未验证**。

Ubuntu 推荐通过 Conda 创建环境。将下面的示例路径替换为你的项目目录：

```sh
cd /path/to/milestone1_20261016
conda env create -f environment.yml
conda activate me5418-group25-m1
python verify.py
python -m pytest -q
python demo.py --policy scripted --seed 25 --output artifacts/ubuntu_demo
```

上述 demo 无需图形桌面，会保存运行结果。在 Ubuntu 桌面会话中，如需实时窗口：

```sh
python demo.py --human --policy scripted --seed 25 --output artifacts/ubuntu_live
```

这些 Ubuntu 命令是待验证的复现步骤，不表示已经在 Ubuntu 运行成功。

Windows 也可在项目根目录运行 `conda env create -f environment.yml`，
再运行 `conda activate me5418-group25-m1`，之后使用 `python` 执行上述脚本。

如果 Windows 已安装 Python 3.12 及 `venv` 模块，也可打开 PowerShell 使用：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe verify.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe demo.py --policy scripted --seed 25
```

Ubuntu 的 `venv` 方案同样要求**已经安装 Python 3.12 及其 `venv` 模块**。
Ubuntu 22.04 默认的 `python3` 不是 Python 3.12，不能直接用它代替：

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python verify.py
python -m pytest -q
python demo.py --policy scripted --seed 25 --output artifacts/ubuntu_demo
```

默认演示无需窗口，结果位于 `artifacts/demo/`：视频、首末帧、逐步 CSV 和摘要 JSON。
加 `--human` 可显示实时窗口，加 `--gif` 可生成 GIF。

演示策略仅用于展示接口与动作功能，不会判断交通安全，也不代表模型训练效果。
提交前需由组员核对代码、报告、课程 AI 使用声明和 Canvas 的最终要求。
