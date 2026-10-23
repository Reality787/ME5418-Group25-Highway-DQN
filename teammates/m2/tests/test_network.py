"""Checks for the Q-network, masking utilities and checkpoint interface.

These tests verify data flow and gradient mechanics. They make no claim that a
network has learned a useful driving policy.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from highway_m1 import ACTION_NAMES, OBSERVATION_NAMES, HighwayDecisionEnv
from highway_m2 import (
    QNetwork, apply_action_mask, greedy_action, load_checkpoint, masked_max, save_checkpoint,
)
from nn_demo import collect_transitions, sample_batch, td_loss, td_targets


@pytest.fixture
def network():
    torch.manual_seed(25)
    return QNetwork()


@pytest.fixture(scope="module")
def pool():
    return collect_transitions(seed=25, count=300)


def test_architecture_matches_proposal(network):
    linear = [layer for layer in network.layers if isinstance(layer, torch.nn.Linear)]
    assert [(l.in_features, l.out_features) for l in linear] == [(34, 128), (128, 128), (128, 9)]
    assert sum(p.numel() for p in network.parameters()) == 34 * 128 + 128 + 128 * 128 + 128 + 128 * 9 + 9
    assert network.observation_dim == len(OBSERVATION_NAMES)
    assert network.action_dim == len(ACTION_NAMES)
    assert isinstance(network.layers[-1], torch.nn.Linear), "Q output must be unbounded"


def test_batch_and_single_observation_shapes(network):
    batch = torch.rand(5, 34) * 2 - 1
    q = network(batch)
    assert q.shape == (5, 9) and q.dtype == torch.float32
    assert torch.allclose(network(batch[0]), q[0])


def test_float64_input_is_converted(network):
    q = network(torch.zeros(2, 34, dtype=torch.float64))
    assert q.dtype == torch.float32


@pytest.mark.parametrize("shape", [(33,), (4, 35), (2, 3, 34)])
def test_wrong_input_shape_is_rejected(network, shape):
    with pytest.raises(ValueError):
        network(torch.zeros(shape))


def test_real_environment_observation_forward(network):
    env = HighwayDecisionEnv()
    try:
        observation, info = env.reset(seed=7)
        q = network(torch.as_tensor(observation))
    finally:
        env.close()
    assert q.shape == (9,)
    assert torch.isfinite(q).all()
    assert info["action_mask"][int(greedy_action(q, torch.as_tensor(info["action_mask"])))]


def test_mask_sets_invalid_actions_to_negative_infinity():
    q = torch.arange(9, dtype=torch.float32).repeat(2, 1)
    mask = torch.ones(2, 9, dtype=torch.bool)
    mask[0, 8] = mask[1, [2, 5, 8]] = False
    masked = apply_action_mask(q, mask)
    assert torch.isneginf(masked[~mask]).all()
    assert torch.equal(masked[mask], q[mask])
    assert greedy_action(q, mask).tolist() == [7, 7]
    assert masked_max(q, mask).tolist() == [7.0, 7.0]


def test_greedy_never_selects_masked_action():
    generator = torch.Generator().manual_seed(0)
    q = torch.randn(500, 9, generator=generator)
    mask = torch.rand(500, 9, generator=generator) > 0.5
    mask[:, 4] = True
    chosen = greedy_action(q, mask)
    assert mask.gather(1, chosen.unsqueeze(1)).all()
    expected = torch.where(mask, q, torch.full_like(q, -1e9)).max(dim=1).values
    assert torch.equal(masked_max(q, mask), expected)


def test_mask_errors():
    with pytest.raises(ValueError):
        apply_action_mask(torch.zeros(2, 9), torch.zeros(2, 9, dtype=torch.bool))
    with pytest.raises(ValueError):
        apply_action_mask(torch.zeros(2, 9), torch.ones(9, dtype=torch.bool))


def test_mask_does_not_block_gradient_to_valid_actions(network):
    obs = torch.zeros(1, 34)
    mask = torch.tensor([[False] * 4 + [True] + [False] * 4])
    masked_max(network(obs), mask).sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in network.parameters())
    assert network.layers[-1].weight.grad[4].abs().sum() > 0
    assert network.layers[-1].weight.grad[[0, 1, 2, 3, 5, 6, 7, 8]].abs().sum() == 0


def test_terminal_transition_does_not_bootstrap(network):
    batch = {
        "next_obs": torch.zeros(2, 34),
        "next_mask": torch.ones(2, 9, dtype=torch.bool),
        "reward": torch.tensor([-20.0, 1.0]),
        "terminated": torch.tensor([1.0, 0.0]),
    }
    targets = td_targets(network, batch, gamma=0.99)
    next_value = masked_max(network(torch.zeros(1, 34)), torch.ones(1, 9, dtype=torch.bool))
    assert targets[0].item() == pytest.approx(-20.0)
    assert targets[1].item() == pytest.approx(1.0 + 0.99 * next_value.item(), rel=1e-6)


def test_sampled_batch_contains_real_episode_endings(pool):
    batch = sample_batch(pool, 64, torch.Generator().manual_seed(1))
    assert batch["obs"].shape == (64, 34)
    assert int(batch["terminated"].sum() + batch["truncated"].sum()) >= 1
    assert batch["mask"].gather(1, batch["action"].unsqueeze(1)).all(), "Logged actions were valid"


def test_backward_and_optimizer_step_update_every_parameter(network, pool):
    batch = sample_batch(pool, 64, torch.Generator().manual_seed(1))
    target = copy.deepcopy(network)
    targets = td_targets(target, batch, gamma=0.99)
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    before = {name: p.detach().clone() for name, p in network.named_parameters()}
    loss, _ = td_loss(network, batch, targets)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    for name, parameter in network.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    optimizer.step()
    for name, parameter in network.named_parameters():
        assert not torch.equal(parameter, before[name]), name
    assert all(p.grad is None for p in target.parameters()), "Target network must stay frozen"


def test_repeated_updates_reduce_fixed_batch_loss(network, pool):
    batch = sample_batch(pool, 64, torch.Generator().manual_seed(1))
    targets = td_targets(copy.deepcopy(network), batch, gamma=0.99)
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    initial = td_loss(network, batch, targets)[0].item()
    for _ in range(100):
        optimizer.zero_grad()
        td_loss(network, batch, targets)[0].backward()
        optimizer.step()
    assert td_loss(network, batch, targets)[0].item() < 0.1 * initial


def test_only_chosen_action_receives_loss_gradient(network, pool):
    batch = sample_batch(pool, 16, torch.Generator().manual_seed(2))
    q = network(batch["obs"])
    q.retain_grad()
    chosen = q.gather(1, batch["action"].unsqueeze(1)).squeeze(1)
    F.smooth_l1_loss(chosen, torch.zeros(16)).backward()
    expected = torch.zeros_like(q, dtype=torch.bool)
    expected[torch.arange(16), batch["action"]] = True
    assert torch.equal(q.grad != 0, expected & (chosen != 0).unsqueeze(1))


def test_checkpoint_round_trip(network, tmp_path):
    path = tmp_path / "q.pt"
    save_checkpoint(network, path, note="test")
    restored = load_checkpoint(path)
    obs = torch.rand(8, 34)
    assert torch.equal(network(obs), restored(obs))
    assert restored.config() == network.config()


def test_checkpoint_with_other_interface_is_rejected(network, tmp_path):
    path = tmp_path / "q.pt"
    save_checkpoint(network, path)
    checkpoint = torch.load(path, weights_only=True)
    checkpoint["action_names"] = list(reversed(checkpoint["action_names"]))
    torch.save(checkpoint, path)
    with pytest.raises(ValueError):
        load_checkpoint(path)


def test_same_seed_gives_identical_initial_weights():
    torch.manual_seed(3)
    first = QNetwork()
    torch.manual_seed(3)
    second = QNetwork()
    for a, b in zip(first.parameters(), second.parameters()):
        assert torch.equal(a, b)
    assert np.isfinite(sum(p.sum().item() for p in first.parameters()))
