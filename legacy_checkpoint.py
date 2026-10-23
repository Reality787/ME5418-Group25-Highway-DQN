"""显式加载整合前的可信推理权重；不转换模型文件，不恢复训练状态。"""

from copy import deepcopy
import argparse
import json
from pathlib import Path

import torch

from checkpoints import compatibility_record, load_checked, signature


def legacy_compatibility_record() -> dict:
    """只接受已验证的源码对；特征、归一化、动作、物理、奖励和依赖须全部相同。"""
    profile = json.loads((Path(__file__).parent / "integration_compatibility.json").read_text())
    current = compatibility_record()
    if current["source_hashes"] != profile["integrated_source_hashes"]:
        raise ValueError("整合源码已改变，旧模型兼容证明不再适用；请重新审核，不能静默加载")
    expected = deepcopy(current)
    expected["source_hashes"] = profile["base_source_hashes"]
    if signature(expected) != profile["base_signature"]:
        raise ValueError("特征/归一化/动作/奖励/动力学/依赖与旧版本不兼容")
    return expected


def load_legacy_inference(path, agent) -> dict:
    """用户显式选择旧推理路径；六个 layers.* 键逐一严格加载，不接受 resume。"""
    expected = legacy_compatibility_record()
    payload = load_checked(path, expected, kind="dqn_inference")
    state = payload["online"]
    reference = agent.online.state_dict()
    if state.keys() != reference.keys():
        raise ValueError("旧网络权重键不一致，不能只按参数量猜测映射")
    for name, value in state.items():
        if (value.shape != reference[name].shape or value.dtype != reference[name].dtype
                or not torch.isfinite(value).all()):
            raise ValueError(f"旧网络参数形状/dtype/有限性不兼容：{name}")
    agent.online.load_state_dict(state, strict=True)
    agent.online.eval()
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="仅接受本项目自产、可信 best.pt/last.pt")
    args = parser.parse_args()
    from agent import DQNAgent
    agent = DQNAgent(seed=0, device="cpu")
    payload = load_legacy_inference(args.checkpoint, agent)
    print(json.dumps({"loaded": str(args.checkpoint), "kind": payload["kind"],
                      "env_steps": payload["env_steps"],
                      "gradient_steps": payload["gradient_steps"],
                      "parameter_count": sum(p.numel() for p in agent.online.parameters()),
                      "resume_supported": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
