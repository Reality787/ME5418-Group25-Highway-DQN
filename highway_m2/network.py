"""Group 25 milestone 2: the proposal's Q-network and action-mask utilities.

``QNetwork`` maps one normalised 34-feature observation to nine Q-values, one per
joint lane/speed command (34 -> 128 ReLU -> 128 ReLU -> 9). The network itself
is unaware of the structural action mask: masking is applied to its outputs
when choosing a greedy action and, in the later learning agent, when taking the
maximum over next-state actions in a temporal-difference target.

No replay memory, exploration schedule or training loop is implemented here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch import nn

from highway_m1 import ACTION_NAMES, OBSERVATION_NAMES


OBSERVATION_DIM = len(OBSERVATION_NAMES)
ACTION_DIM = len(ACTION_NAMES)
HIDDEN_SIZES = (128, 128)


class QNetwork(nn.Module):
    """Fully connected action-value network, one output per joint action.

    A multilayer perceptron suits this input: the 34 features are already
    engineered numerical measurements without an image grid or sequence axis.
    Producing all nine Q-values in one forward pass lets greedy selection and
    the TD maximum use a single evaluation per observation.
    """

    def __init__(self, observation_dim: int = OBSERVATION_DIM,
                 action_dim: int = ACTION_DIM,
                 hidden_sizes: Sequence[int] = HIDDEN_SIZES):
        super().__init__()
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.hidden_sizes = tuple(int(size) for size in hidden_sizes)
        layers: list[nn.Module] = []
        width = self.observation_dim
        for size in self.hidden_sizes:
            layers += [nn.Linear(width, size), nn.ReLU()]
            width = size
        # The output layer stays linear: Q-values are unbounded signed returns,
        # and the reward contains -20 terminal penalties.
        layers.append(nn.Linear(width, self.action_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Return Q-values of shape (batch, 9), or (9,) for one observation."""
        if observation.shape[-1] != self.observation_dim:
            raise ValueError(
                f"Expected last dimension {self.observation_dim}, "
                f"received shape {tuple(observation.shape)}."
            )
        if observation.dim() not in (1, 2):
            raise ValueError("Expected an observation or a batch of observations.")
        return self.layers(observation.to(dtype=torch.float32))

    def config(self) -> dict:
        return {"observation_dim": self.observation_dim,
                "action_dim": self.action_dim,
                "hidden_sizes": list(self.hidden_sizes)}


def apply_action_mask(q_values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Replace structurally invalid actions by -inf so max/argmax ignore them.

    The mask is the environment's ``info["action_mask"]``. Every row must allow
    at least one action; the environment always allows Keep/Hold.
    """
    mask = torch.as_tensor(mask, dtype=torch.bool, device=q_values.device)
    if mask.shape != q_values.shape:
        raise ValueError(f"Mask shape {tuple(mask.shape)} differs from Q shape {tuple(q_values.shape)}.")
    if not bool(mask.any(dim=-1).all()):
        raise ValueError("Each action mask must allow at least one action.")
    return q_values.masked_fill(~mask, float("-inf"))


def greedy_action(q_values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Index of the highest-valued valid action (ties: lowest index)."""
    return apply_action_mask(q_values, mask).argmax(dim=-1)


def masked_max(q_values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Highest Q-value among valid actions, as used by a masked TD target."""
    return apply_action_mask(q_values, mask).max(dim=-1).values


def save_checkpoint(network: QNetwork, path: str | Path, **metadata) -> None:
    """Save weights with the interface they assume; enough for inference only.

    Optimizer, replay memory and random-generator state are not included, so a
    checkpoint written here does not support exact resumption of training.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": network.state_dict(),
        "network_config": network.config(),
        "observation_names": list(OBSERVATION_NAMES),
        "action_names": list(ACTION_NAMES),
        "metadata": metadata,
    }, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> QNetwork:
    """Rebuild a network, refusing checkpoints made for another interface."""
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=True)
    if (checkpoint["observation_names"] != list(OBSERVATION_NAMES)
            or checkpoint["action_names"] != list(ACTION_NAMES)):
        raise ValueError("Checkpoint feature or action names differ from this environment.")
    network = QNetwork(**checkpoint["network_config"])
    network.load_state_dict(checkpoint["state_dict"])
    return network
