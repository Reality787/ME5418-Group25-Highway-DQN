# ME5418 Group 25 — Environment code and report (Milestone 1)

This submission implements the three-lane highway environment in the revised
**9-action proposal**. The ego vehicle has nine joint lateral/speed actions,
a 34-dimensional vector observation, reward components exposed for inspection,
seeded traffic, and RGB visualization. Milestone 1 demonstrates the environment
and its API; it does not include DQN training or a trained agent.

## Recreate the environment

Run all commands from the extracted project root, the directory containing
`environment.yml`, `verify.py`, and `demo.py`. Python **3.12.14 on Windows** is
the tested runtime. `environment.yml` pins that Python version; Ubuntu has not
been tested.
The version pins in `requirements.txt` record the direct packages used for verification.
`requirements-lock-windows.txt` additionally records the complete installed
dependency closure for the tested Windows runtime, excluding unrelated packages.

With Conda on Windows:

```powershell
conda env create -f environment.yml
conda activate me5418-group25-m1
```

For Ubuntu, Conda is the recommended reproduction route because it supplies the
pinned Python version. Replace the example path with your extracted project root:

```sh
cd /path/to/milestone1_20261016
conda env create -f environment.yml
conda activate me5418-group25-m1
python verify.py
python -m pytest -q
python demo.py --policy scripted --seed 25 --output artifacts/ubuntu_demo
```

The demo above is headless and writes its outputs to `artifacts/ubuntu_demo`.
On a desktop Ubuntu session, a live window can additionally be requested with:

```sh
python demo.py --human --policy scripted --seed 25 --output artifacts/ubuntu_live
```

These Ubuntu commands are provided for reproduction; no Ubuntu run is claimed.

Alternatively, if Python 3.12 and its `venv` module are already installed on
Windows:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe verify.py
```

After activating either environment, the remaining examples use `python`.
For an unactivated Windows virtual environment, replace `python` with
`.\.venv\Scripts\python.exe`.

An Ubuntu `venv` is also an option **only if Python 3.12 and its `venv` module
are already installed**. Ubuntu 22.04 does not provide Python 3.12 as its default
`python3`; a bare `python3 -m venv` there does not recreate this runtime. From
the project root:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python verify.py
python -m pytest -q
python demo.py --policy scripted --seed 25 --output artifacts/ubuntu_demo
```

## Run and inspect

```powershell
python verify.py
python -m pytest -q
python demo.py --policy scripted --seed 25 --output artifacts/demo
```

The default demo runs without a display. It writes the measured
`trajectory.csv`, `summary.json`, `initial.png`, `final.png`, and `demo.mp4`.
If the MP4 encoder is unavailable, it writes `demo.gif` and records the fallback
in `summary.json`. The output path is relative to the current working directory.
The legend highlights the last executed action and separately labels the next
decision's structural mask. Requested and executed IDs are both displayed.
Closing the live window saves a partial rollout with `interrupted=true`; it is
not labelled as a collision, completed episode, or time-limit truncation.

```powershell
python demo.py --human
python demo.py --gif --output artifacts/demo_gif
python demo.py --policy random --seed 25 --episodes 3 --output artifacts/random
python demo.py --policy hold --seed 25 --output artifacts/hold
```

`--human` opens a window with the same RGB scene and legend used in the saved
video. For multiple episodes, artifacts are placed in `episode_001`,
`episode_002`, etc.; the seed increases by one per episode. Re-running an output
path replaces files with the same names.

The scripted policy requests Keep/Faster at the start, Left/Slower after 4 s,
and Right/Faster after 10 s when those actions are structurally available.
It uses Keep/Hold between requests and during lane changes. The random policy
samples uniformly from the structural action mask. Neither policy considers
nearby traffic when selecting actions; neither is a safety controller or a
performance baseline. A collision is an expected possible outcome in this
environment and is recorded without altering the results.

## Environment API

```python
from highway_m1 import HighwayDecisionEnv

env = HighwayDecisionEnv(render_mode="rgb_array")
observation, info = env.reset(seed=25)
observation, reward, terminated, truncated, info = env.step(4)
rgb_image = env.render()
env.close()
```

- `observation_space`: `Box`, shape `(34,)`, dtype `float32`.
- `action_space`: `Discrete(9)`.
- `reset(seed=...)` and the five-value `step(...)` follow the Gymnasium API.
- `info["action_mask"]` is a nine-entry Boolean structural mask. It identifies
  lane-boundary, manoeuvre, and speed-target constraints, not traffic safety.
- `info` exposes speed, lane targets, manoeuvre state, reward terms, traffic
  gaps, control inputs, collision/off-road state, and progress for debugging.
- `render()` in `rgb_array` mode returns a `uint8` RGB image.

| ID | Lateral command | Speed-target command |
|---:|---|---|
| 0 | Left | Slower |
| 1 | Left | Hold |
| 2 | Left | Faster |
| 3 | Keep | Slower |
| 4 | Keep | Hold |
| 5 | Keep | Faster |
| 6 | Right | Slower |
| 7 | Right | Hold |
| 8 | Right | Faster |

The full observation indices, normalization, action-mask rules, reward formula,
termination rules, and proposal-to-code choices are documented in
[`SPECIFICATION.md`](SPECIFICATION.md). `OBSERVATION_NAMES` and `ACTION_NAMES`
are also exported by `highway_m1` for programmatic inspection. A speed command
changes the target speed; the simulated vehicle reaches it through continuous
control. Keep/Hold preserves an unfinished lane-change target.

## Files and verification evidence

| File | Role |
|---|---|
| `highway_m1/` | Custom environment, observation/action/reward implementation |
| `demo.py` | Functional rollout, RGB legend, video and trajectory recording |
| `verify.py` | Gymnasium checker, seeded reproducibility, observation and RGB smoke checks |
| `tests/` | Behavioral checks for environment semantics |
| `SPECIFICATION.md` | Exact environment contract and modeling assumptions |
| `environment.yml`, `requirements.txt` | Runtime recreation and pinned dependencies |
| `artifacts/verification.json` | Verification result and runtime versions from a real run |
| `artifacts/demo/` | Demo outputs from a real rollout |
| `PROJECT_ROADMAP.md` | Current milestone acceptance and planned later deliverables |

`verify.py` compares two complete seeded rollouts, including actions,
observations and rewards, through a SHA-256 digest. This tests reproducibility
in the same runtime, not bitwise equality across operating systems or library
versions. The RGB smoke check checks shape, dtype and nonconstant pixels. Inspect the saved PNGs
and video for the visual scene. Experiment logs describe only these actual
functional runs; they do not imply a trained policy or evaluated driving skill.

## External software and AI use

This project extends the external `highway-env` simulator and uses Gymnasium,
NumPy, pygame-ce, Matplotlib, Pillow, ImageIO, and imageio-ffmpeg. Pytest is used
for verification. Refer to the report and the source comments for implementation
details and external-code attribution.

The initial implementation, documentation, tests, and draft report were
AI-assisted. Group members must inspect and understand the final submission
and make any required course AI-use declaration. No individual contribution,
human review, or training result is claimed by these files.
