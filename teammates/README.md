# 队友原代码与最小适配

`m1/` 和 `m2/` 保存原演示、测试、规格及依赖清单；实际可导入包在仓库根的 `highway_m1/`、`highway_m2/`。22 个导入文件中 19 个逐字节保留，3 个局部适配；来源、原始/当前哈希见 [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json)，三个原文件的改动见 [changes/](changes/)。新增 schema/rendering 为从 M1 提取的共用代码。

正式训练使用原项目环境与 M2 QNetwork 的薄适配接口。M1 虽然也是 34 维/9 动作，但归一化、奖励和动力学不同，保留为独立诊断环境；不能直接使用正式旧权重当作同一环境评测。M2 的数值演示和简单 checkpoint 也不替代正式 DQN/恢复协议。

从仓库根使用项目解释器运行：

```bash
python teammate_demo.py --help
python -m teammates.m2.nn_demo --help
```

完整整合结果和实际验证命令见 [整合记录](../docs/teammate_code_integration.md)。原 pytest 测试保留，但当前项目环境未安装 pytest，原测试没有重跑；原视频导出需要缺失的 imageio/imageio-ffmpeg，也未验证。新 PNG 入口和正式 unittest 已实际运行，没有安装依赖。

原 README/报告引用的 artifacts、视频、ZIP 内模型和历史日志没有导入本仓库。依赖版本是原来源记录，不要用它们覆盖项目已有环境。M2 附带的重复 M1 包、测试和规格已按字节核对去重。保留队友文档的 AI 辅助说明，不推断具体作者或成员分工。
