"""Forward/backward demonstration of the Q-network on real environment data.

A pool of transitions is collected from the milestone-1 environment with a
uniformly random valid-action policy, and one minibatch is drawn from it as a
replay memory later would. The script then shows the forward pass,
a masked one-step TD loss against a frozen copy of the network, backpropagated
gradients, an optimizer step that changes the parameters, a fixed-batch fitting
diagnostic and checkpoint save/load. It does not train a driving policy: there
is no replay memory, exploration schedule or repeated data collection.
"""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import tempfile
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "me5418-group25-mpl"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from highway_m1 import ACTION_NAMES, HighwayDecisionEnv
from highway_m2 import QNetwork, greedy_action, load_checkpoint, masked_max, save_checkpoint


def collect_transitions(seed: int, count: int) -> dict[str, torch.Tensor]:
    """Record (s, a, r, s', terminated, truncated, mask, next_mask) tuples.

    The next observation is read before any reset, so a terminal transition
    keeps the true final observation. Later episodes continue the seeded
    environment generator, making the whole batch reproducible.
    """
    env = HighwayDecisionEnv()
    rng = np.random.default_rng(seed)
    keys = ("obs", "action", "reward", "next_obs", "terminated", "truncated", "mask", "next_mask")
    data: dict[str, list] = {key: [] for key in keys}
    episodes = 1
    try:
        observation, info = env.reset(seed=seed)
        while len(data["action"]) < count:
            action = int(rng.choice(np.flatnonzero(info["action_mask"])))
            next_observation, reward, terminated, truncated, next_info = env.step(action)
            for key, value in zip(keys, (observation, action, reward, next_observation, terminated,
                                         truncated, info["action_mask"], next_info["action_mask"])):
                data[key].append(value)
            observation, info = next_observation, next_info
            if terminated or truncated:
                observation, info = env.reset()
                episodes += 1
    finally:
        env.close()
    batch = {
        "obs": torch.as_tensor(np.asarray(data["obs"]), dtype=torch.float32),
        "action": torch.as_tensor(data["action"], dtype=torch.int64),
        "reward": torch.as_tensor(data["reward"], dtype=torch.float32),
        "next_obs": torch.as_tensor(np.asarray(data["next_obs"]), dtype=torch.float32),
        "terminated": torch.as_tensor(data["terminated"], dtype=torch.float32),
        "truncated": torch.as_tensor(data["truncated"], dtype=torch.float32),
        "mask": torch.as_tensor(np.asarray(data["mask"]), dtype=torch.bool),
        "next_mask": torch.as_tensor(np.asarray(data["next_mask"]), dtype=torch.bool),
    }
    batch["episodes"] = episodes
    return batch


def sample_batch(pool: dict, size: int, generator: torch.Generator) -> dict:
    """Draw a random minibatch, forcing in every episode ending of the pool.

    Endings are rare (at most one per episode); including them guarantees that
    this one demonstration batch exercises both terminal and truncated cases.
    """
    endings = torch.nonzero((pool["terminated"] + pool["truncated"]) > 0).flatten()[:size]
    others = torch.nonzero((pool["terminated"] + pool["truncated"]) == 0).flatten()
    chosen = others[torch.randperm(len(others), generator=generator)[:size - len(endings)]]
    index = torch.cat((endings, chosen))
    index = index[torch.randperm(len(index), generator=generator)]
    return {key: value[index] for key, value in pool.items() if torch.is_tensor(value)}


def td_targets(target_network: QNetwork, batch: dict, gamma: float) -> torch.Tensor:
    """r + gamma * (1 - terminated) * max over valid a' of Q_target(s', a').

    Time-limit truncation still bootstraps; only collision/off-road stops it.
    The frozen target network receives no gradient.
    """
    with torch.no_grad():
        next_value = masked_max(target_network(batch["next_obs"]), batch["next_mask"])
        return batch["reward"] + gamma * (1.0 - batch["terminated"]) * next_value


def td_loss(network: QNetwork, batch: dict, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    q_values = network(batch["obs"])
    chosen = q_values.gather(1, batch["action"].unsqueeze(1)).squeeze(1)
    # Huber loss: quadratic for small errors, linear for large ones such as
    # the -20 collision reward, which limits the size of a single update.
    return F.smooth_l1_loss(chosen, targets), q_values


def parameter_snapshot(network: QNetwork) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in network.named_parameters()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=25)
    parser.add_argument("--pool-size", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--fit-steps", type=int, default=300)
    parser.add_argument("--output", type=Path, default=Path("artifacts/nn_demo"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True, exist_ok=True)
    network = QNetwork()
    parameter_count = sum(p.numel() for p in network.parameters())
    print(f"[1] Network: {network}\n    trainable parameters: {parameter_count}")

    pool = collect_transitions(args.seed, args.pool_size)
    batch = sample_batch(pool, args.batch_size, torch.Generator().manual_seed(args.seed))
    print(f"[2] Collected {args.pool_size} real transitions from {pool['episodes']} episode(s); "
          f"sampled a batch of {args.batch_size}: terminated={int(batch['terminated'].sum())}, "
          f"truncated={int(batch['truncated'].sum())}, "
          f"reward range [{batch['reward'].min():.3f}, {batch['reward'].max():.3f}]")

    started = time.perf_counter()
    with torch.no_grad():
        q_values = network(batch["obs"])
    forward_ms = 1000 * (time.perf_counter() - started)
    assert q_values.shape == (args.batch_size, 9) and q_values.dtype == torch.float32
    assert torch.isfinite(q_values).all()
    masked_choice = greedy_action(q_values, batch["mask"])
    unmasked_choice = q_values.argmax(dim=1)
    unmasked_invalid = int((~batch["mask"].gather(1, unmasked_choice.unsqueeze(1))).sum())
    assert batch["mask"].gather(1, masked_choice.unsqueeze(1)).all()
    # Show a sample whose mask excludes something, so the effect is visible.
    sample = int(torch.nonzero(~batch["mask"].all(dim=1)).flatten()[0]) if not batch["mask"].all() else 0
    print(f"[3] Forward pass: input {tuple(batch['obs'].shape)} -> output {tuple(q_values.shape)} "
          f"in {forward_ms:.2f} ms. Sample {sample} (untrained, values are arbitrary):")
    for index, name in enumerate(ACTION_NAMES):
        valid = "valid" if batch["mask"][sample, index] else "MASKED"
        print(f"      {index} {name:<13} Q={q_values[sample, index]:+.4f}  {valid}")
    print(f"    masked greedy action={int(masked_choice[sample])}, unmasked argmax={int(unmasked_choice[sample])}; unmasked argmax would pick "
          f"an invalid action in {unmasked_invalid}/{args.batch_size} samples")

    target_network = copy.deepcopy(network).eval()
    for parameter in target_network.parameters():
        parameter.requires_grad_(False)
    targets = td_targets(target_network, batch, args.gamma)
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)
    optimizer.zero_grad()
    loss, q_values = td_loss(network, batch, targets)
    q_values.retain_grad()
    loss.backward()
    nonzero_q_gradients = int((q_values.grad != 0).sum())
    gradient_norms = {name: float(p.grad.norm()) for name, p in network.named_parameters()}
    assert all(np.isfinite(v) and v > 0 for v in gradient_norms.values())
    print(f"[4] Masked TD loss (Huber) = {loss.item():.6f}; dLoss/dQ is non-zero for "
          f"{nonzero_q_gradients}/{q_values.numel()} outputs (only each sample's chosen action)")
    for name, norm in gradient_norms.items():
        print(f"      grad norm {name:<16} {norm:.6f}")

    before = parameter_snapshot(network)
    optimizer.step()
    parameter_changes = {name: float((p.detach() - before[name]).norm())
                         for name, p in network.named_parameters()}
    assert all(change > 0 for change in parameter_changes.values())
    with torch.no_grad():
        loss_after_one_step = td_loss(network, batch, targets)[0].item()
    print(f"[5] One Adam step (lr={args.lr}) changed every parameter tensor; "
          f"loss on the same batch {loss.item():.6f} -> {loss_after_one_step:.6f}")
    for name, change in parameter_changes.items():
        print(f"      |delta| {name:<16} {change:.6f}")

    # Diagnostic only: the same 64 samples and frozen targets are reused, so a
    # falling curve checks the gradient path, not driving ability.
    # fit_losses[k] is the loss after k updates; the first update was step [5].
    fit_losses = [loss.item()]
    for _ in range(args.fit_steps - 1):
        optimizer.zero_grad()
        fit_loss = td_loss(network, batch, targets)[0]
        fit_losses.append(fit_loss.item())
        fit_loss.backward()
        optimizer.step()
    with torch.no_grad():
        final_loss = td_loss(network, batch, targets)[0].item()
    fit_losses.append(final_loss)
    print(f"[6] Fixed-batch fitting diagnostic: {args.fit_steps} updates, loss "
          f"{fit_losses[0]:.6f} -> {final_loss:.6f} (memorising one batch, not learning to drive)")

    figure, axis = plt.subplots(figsize=(6, 3.2))
    axis.semilogy(fit_losses)
    axis.set_xlabel("update on the same fixed batch")
    axis.set_ylabel("Huber TD loss (log)")
    axis.set_title("Diagnostic: gradient updates reduce loss on one batch")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(args.output / "fixed_batch_loss.png", dpi=150)
    plt.close(figure)

    checkpoint = args.output / "q_network_demo.pt"
    save_checkpoint(network, checkpoint, purpose="diagnostic, not a trained policy", seed=args.seed)
    restored = load_checkpoint(checkpoint)
    with torch.no_grad():
        identical = bool(torch.equal(network(batch["obs"]), restored(batch["obs"])))
    assert identical
    print(f"[7] Saved and reloaded {checkpoint}; outputs identical: {identical}")

    summary = {
        "purpose": "Network forward/backward demonstration; no policy training or evaluation",
        "architecture": "34 -> 128 ReLU -> 128 ReLU -> 9 (linear)",
        "trainable_parameters": parameter_count,
        "seed": args.seed, "batch_size": args.batch_size, "gamma": args.gamma, "learning_rate": args.lr,
        "data": {"source": "HighwayDecisionEnv, uniformly random valid actions",
                 "pool_size": args.pool_size, "episodes": pool["episodes"],
                 "terminated": int(batch["terminated"].sum()), "truncated": int(batch["truncated"].sum()),
                 "reward_min": float(batch["reward"].min()), "reward_max": float(batch["reward"].max())},
        "forward": {"input_shape": list(batch["obs"].shape), "output_shape": list(q_values.shape),
                    "dtype": str(q_values.dtype), "all_finite": True,
                    "unmasked_argmax_invalid": unmasked_invalid,
                    "masked_greedy_always_valid": True},
        "backward": {"initial_td_loss": loss.item(), "nonzero_output_gradients": nonzero_q_gradients,
                     "gradient_norms": gradient_norms},
        "optimizer_step": {"parameter_change_norms": parameter_changes,
                           "loss_after_one_step": loss_after_one_step},
        "fixed_batch_fit": {"updates": args.fit_steps, "initial_loss": fit_losses[0],
                            "final_loss": final_loss},
        "checkpoint": {"path": checkpoint.as_posix(), "reload_outputs_identical": identical},
        "python": platform.python_version(), "platform": platform.platform(),
        "dependencies": {name: importlib.metadata.version(name)
                         for name in ("torch", "highway-env", "gymnasium", "numpy", "matplotlib")},
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Summary written to {args.output / 'summary.json'}")


if __name__ == "__main__":
    main()
