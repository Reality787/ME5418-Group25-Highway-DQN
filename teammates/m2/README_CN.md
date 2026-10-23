# ME5418 第二阶段：神经网络（快速运行）

本目录在第一阶段环境（`highway_m1/`，未改动）的基础上加入 Q 网络
（34 → 128 ReLU → 128 ReLU → 9），并用真实环境数据演示前向传播、带 mask 的 TD loss、
反向传播、参数更新和模型保存/加载。**本阶段不训练驾驶策略。** 详细说明见 [README.md](README.md)。

在解压后的项目根目录执行：

```sh
conda env create -f environment.yml
conda activate me5418-group25-m2
python -m pytest -q
python nn_demo.py
```

结果保存在 `artifacts/nn_demo/`：`summary.json`、`fixed_batch_loss.png` 和演示用的 checkpoint。
依赖为 CPU 版 PyTorch 2.13.0，无需 GPU。
