# NUS ME5418 — Learning Agent C&R (Group 25)

This submission contains the complete runnable learning program, regression tests,
an actual Conda environment snapshot, and the **two-page English report**
`Learning_Agent_CR_Report.pdf` (editable source: `Learning_Agent_CR_Report.md`).
No pretrained model is required. The short demo starts from new weights and learns
from real interactions with the project environment, not synthetic observations.

## Setup

Open a terminal in this extracted directory. On Ubuntu/Linux with Conda installed:

```bash
conda env create -f environment.yml
conda activate me5418-highway
python -c "import sys, torch, gymnasium, highway_env; print(sys.executable); print(torch.__version__); print(gymnasium.__version__)"
```

If this environment already exists, check it before creating another one; do not
overwrite it. `environment.yml` was exported from the actual project environment,
with the personal prefix removed. Verified versions: Python 3.11.16,
PyTorch 2.14.1+cpu, Gymnasium 1.4.0, HighwayEnv 1.12.1, NumPy 2.4.6,
pygame-ce 2.5.8 and Matplotlib 3.11.2. PyTorch is CPU-only; CUDA is not required.
HighwayEnv is installed as a Python package: no separate simulator executable
or external road assets are needed. Headless commands need no display server.

The environment file includes PyPI and the official PyTorch CPU wheel index for
the pinned `+cpu` build. The project's previously verified CPU install recipe is
`python -m pip install -r requirements-network.txt`, after the environment packages
in `requirements.txt`; do not rerun it in an already working environment.
The submission was extracted and reproduced **using existing dependencies**.
It was not freshly installed in a new Conda environment or on a new Ubuntu machine.

## Run a short, real learning demo

```bash
python train.py --mode smoke --device cpu --seed 101 --steps 1004 --run-dir outputs/training/learning_agent_cr_demo
```

This finite CPU run performs 1,004 real environment decisions and **two DQN
optimizer updates**, at steps 1,000 and 1,004. It uses the formal 34-feature,
nine-action environment and project reward. It collects copied transitions in
replay, uses legal-action masks in exploration/greedy/TD targets, and saves
checkpoints. Initial and final validation each use six independent fixed scenes;
validation does not update the network or replay. Console output reports validation, progress and completion; `config.json` records
the interpreter and dependency versions. This is a learning-pipeline check, not evidence of a
converged or safe driving policy. It does not continue the historical formal runs.

The run directory must be new. For another demonstration, change the path to an
unused directory rather than deleting or overwriting existing results.
Inspect `status.json` for `status: completed`, `env_steps: 1004` and
`gradient_steps: 2`. `last.pt` contains the final inference weights;
`best.pt` is selected by validation and can correspond to the initial model.
`resume.pt` is the most recent complete episode boundary, which may precede the
final step; it must not be confused with an exact mid-episode continuation.
Only load checkpoints you generated or otherwise trust.

## Checks and optional module demos

```bash
python -m unittest discover -s tests -v
python demo.py --mode project --render none --episodes 1 --seed 0
python network_demo.py --seed 0 --steps 30
python train.py --help
```

The full suite has 167 tests. Actual packaging results and the corresponding Git
commit are recorded in `SUBMISSION_RECORD.json`. Tests cover real simulation,
observations, actions, rewards, network gradients, masks, TD targets, replay,
validation isolation and episode-boundary resume. The network demo's synthetic
regression is explicitly a numerical check, not driving training.

## Files and implementation

| Files | Responsibility |
| --- | --- |
| `config.py` | Fixed scenario, observation, action, reward and learning settings |
| `environment.py`, `project_environment.py` | Public factories and formal Gym environment |
| `joint_actions.py`, `observation.py`, `reward.py` | Joint control/masks, real 34-D sensing, project reward |
| `network.py` | 34→128→128→9 ReLU Q network, 22,153 parameters |
| `agent.py`, `replay_buffer.py` | Ordinary masked DQN and independent transition storage |
| `train.py`, `scenarios.py` | Finite training, scenario randomization and split rules |
| `metrics.py`, `validation.py`, `checkpoints.py` | Physical metrics, isolated validation and compatible saves |
| `demo.py`, `network_demo.py`, `validate_env.py`, `plot_training.py` | Diagnostics and log plotting |
| `baseline.py`, `evaluate.py` | Explicit future interfaces; not implemented final evaluation |
| `tests/` | Existing complete regression suite |

True termination uses target `r`; otherwise, including time truncation, the target
is `r + 0.99 * max_legal Q_target(next_obs)`. Replay uses the step's final observation
before reset. HighwayEnv supplies the underlying simulation and low-level control;
the project implements the sensing/control adapter and ordinary DQN learning core.
No Stable-Baselines3 trainer is used. Structural masks do not filter traffic risk.

## Scope and authorship

Formal seeds 11 and 22 previously completed 50,000 steps each; seed 33 remains
incomplete. Final tests, generalization and the gap-based baseline comparison are
unfinished. None of those historical models or large logs are needed or included.
Some earlier-stage TODO comments remain in frozen source; the implemented code,
tests and report describe the current state. Runtime source was preserved to retain
checkpoint compatibility.

ChatGPT/Codex assisted with design organization, code generation, debugging, tests
and documentation. Group members remain responsible for understanding, reviewing,
accurately declaring their individual contributions and submitting the work under
the course rules. No unperformed human review or invented member contribution is
claimed. The PDF contains external-library references and an AI use declaration.
