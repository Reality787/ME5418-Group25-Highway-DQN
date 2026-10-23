# ME5418 Group 25 — Neural Network milestone

This package adds the revised proposal's Q-network (34 -> 128 ReLU -> 128 ReLU -> 9) to
the milestone-1 highway environment, and demonstrates its forward pass, a masked
temporal-difference loss, backpropagation, an optimizer step and checkpointing on real
environment observations. No driving policy is trained here: replay memory,
epsilon-greedy exploration and the training loop belong to the learning-agent milestone.

## Contents

| Path | Purpose |
| --- | --- |
| `highway_m2/network.py` | `QNetwork`, action-mask utilities (`apply_action_mask`, `greedy_action`, `masked_max`) and checkpoint save/load |
| `nn_demo.py` | End-to-end forward/backward demonstration; writes `artifacts/nn_demo/` |
| `tests/test_network.py` | Network, masking, TD-target, gradient and checkpoint tests |
| `highway_m1/`, `tests/test_environment.py`, `SPECIFICATION.md` | Milestone-1 environment, unchanged, and its tests |
| `artifacts/` | Recorded outputs of the commands below |
| `report/` | Two-page milestone report |

## Setup and run

Run from the extracted project root (the folder containing `environment.yml`):

```sh
conda env create -f environment.yml
conda activate me5418-group25-m2
python -m pytest -q
python nn_demo.py
```

Without conda, Python 3.12 with `venv` also works:
`python3.12 -m venv .venv`, activate it, then `python -m pip install -r requirements.txt`.
The dependency files install the CPU build of PyTorch 2.13.0; no GPU is required.
Verified on Ubuntu (Python 3.12.14). The same commands should work on Windows, where all
pinned packages, including the CPU PyTorch wheel, are available; this milestone was not re-run there.

`nn_demo.py` options: `--seed`, `--pool-size`, `--batch-size`, `--gamma`, `--lr`,
`--fit-steps`, `--output`. The default run is deterministic.

## What the demonstration shows

1. The network and its 22,153 trainable parameters.
2. 600 transitions collected with uniformly random *valid* actions, from which one
   64-sample minibatch is drawn; every episode ending in the pool (collision or
   time limit) is included so both target cases are exercised.
3. Forward pass `(64, 34) -> (64, 9)`, float32 and finite. Untrained Q-values are
   arbitrary. Without masking, argmax would select a structurally invalid action in
   about half the samples; masked greedy selection never does.
4. Huber loss between `Q(s, a)` and the one-step target
   `r + 0.99 * (1 - terminated) * max_valid Q_target(s', a')`, computed with a frozen
   copy of the network. Time-limit truncation still bootstraps; collision/off-road
   termination does not. Gradients reach every layer; only the taken action's output
   receives a loss gradient.
5. One Adam step changes every parameter tensor.
6. A fixed-batch fitting diagnostic: repeated updates on the same batch and fixed
   targets drive the loss down (`fixed_batch_loss.png`). This checks the gradient
   path only; it is memorisation, not learning to drive.
7. Checkpoint save/reload reproduces outputs exactly; checkpoints store feature and
   action names and are refused if the interface differs. They contain weights only
   and support inference, not exact training resumption.

## External code and AI use

PyTorch provides layers, autograd and the Adam optimizer; HighwayEnv and Gymnasium
provide the simulator, as in milestone 1. Claude (Anthropic) assisted in generating the
code. The use is disclosed; no human-only authorship or individual member contribution is
asserted. The team remains responsible for the content and quality of the submission.
