# 队友来源记录

M1/M2原始ZIP分别来自用户提供的本地下载文件。根目录 `highway_m1/`、`highway_m2/`、`demo.py`、`verify.py`、`nn_demo.py` 直接采用原件。M1/M2原pytest测试保留在本目录的 `m1/tests` 和 `m2/tests`，原README、规格和依赖记录也保留。

`SOURCE_MANIFEST.json` 列出每个采用文件的原始路径和SHA256。本轮22个采用文件中21个逐字节匹配原件；verify.py仅将缺少可选依赖的版本记录为null（差异见changes/verify_optional_metadata.patch），其Gym检查和rollout不变；此前的dtype/device适配、schema/render拆分和旧模型兼容补丁已撤下。M2包中重复的M1环境、规格和测试经字节比较一致，去重保留M1来源的一份。

原文档提到的artifacts、训练/数值演示权重、视频和报告只属于原提交记录；未将它们重新作为当前结果上传。原pytest和视频功能在缺少可选依赖的本机未验证。当前实际结果与入口以根README和 `docs/m1_m2_authoritative_integration.md` 为准。
