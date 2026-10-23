# Learning Agent C&R
NUS ME5418 Machine Learning in Robotics | Group 25
Deep Reinforcement Learning for Highway Lane Change and Speed Decisions

## 1. Code Explanation
The learning program connects a project-specific Gymnasium environment to an ordinary, masked Deep Q-Network (DQN). HighwayEnv supplies the road, surrounding traffic, vehicle dynamics, collision detection and low-level tracking. The formal environment uses three lanes, one controlled vehicle, 20 surrounding vehicles and a 40-second limit. A decision advances 0.2 s, normally comprising three physical updates at 15 Hz. Training uses the project reward, not the stock debugging reward.

[FLOW]

The float32 observation contains 34 features: 18 lane-specific front/rear traffic features (presence, bumper gap and relative speed), seven ego/control features, and nine previous-action indicators. Visibility is filtered by vehicle-centre Euclidean distance at 100 m. Feature order and normalization are fixed; reset uses an all-zero action history. Nine actions combine Left/Keep/Right with slower/hold/faster. The speed target is one of 0, 6.95, 13.90, 20.85 and 27.80 m/s. Both targets are updated together before one simulator step; actual vehicle motion still follows HighwayEnv dynamics.

network.py implements Linear(34,128), ReLU, Linear(128,128), ReLU and Linear(128,9), with biases: 22,153 parameters. Outputs are raw Q-values, without Softmax. agent.py maintains separate online and target networks, copying online weights at initialization. Masked epsilon-greedy explores uniformly among legal actions or selects the largest legal online Q-value. The same structural mask is used in the TD target. It excludes road-edge lane changes, new lane changes while one is unfinished, and speed-index overflow; it is not a traffic-risk safety filter.

replay_buffer.py stores independent copies of observation, action, reward, next observation, terminated, truncated and next-action mask in a seeded ring buffer. Uniform batches are sampled without replacement. train.py checks that the executed action equals the selected action, stores the step's final observation before reset, and schedules learning by environment steps. The scalar reward combines progress, short-headway penalty, collision/off-road failure cost, lane-change initiation cost and target-speed-change cost.

[TD]

A true terminal transition uses y = r. Otherwise, including a time-limit truncation, y = r + 0.99 max over legal next actions of Q_target(next observation, action). Only samples requiring bootstrap are evaluated by the target network, under no_grad; this avoids zero times negative infinity for terminal masks. Current action values use gather with matching one-dimensional batch shapes. SmoothL1Loss is backpropagated through the online network, followed by gradient clipping and Adam. The target has no gradients and is hard-synchronized periodically.

[HYPERPARAMETERS]

The CPU entry point is train.py --mode smoke --seed 101 --steps 1004 --run-dir outputs/training/learning_agent_cr_demo. It starts without pretrained weights, collects genuine driving transitions and performs two scheduled optimizer updates. Separate fixed-scenario validation uses epsilon=0 and no_grad, without modifying replay or training random streams. Checkpoints distinguish inference weights from complete episode-boundary resume state. The submission README provides setup and execution commands.

[PAGEBREAK]

## 2. Existing Codes / Libraries
HighwayEnv 1.12.1 [1] provides the simulation, IDM/MOBIL traffic behaviour, tracking control and kinematics; these are reused through the installed package, not claimed as team-built simulation. Gymnasium 1.4.0 [2] provides the Env/space interfaces and reset/step termination contract. PyTorch 2.14.1+cpu [3] supplies Linear/ReLU, automatic differentiation, Adam, loss functions and serialization. NumPy 2.4.6 handles numerical arrays and seeded generators; Matplotlib plots saved logs. Python unittest supports regression testing. environment.yml records the actual dedicated Conda environment, including pip packages, without a personal prefix.

Project-specific, AI-assisted code implements the joint-action adapter, structural masks, 34-feature sensing, project reward, DQN orchestration, replay, randomized scenarios, validation metrics and compatible checkpoints. No Stable-Baselines3 trainer or other ready-made learning core is used. Installed third-party source was inspected for interface and control behaviour and was not edited. The report describes the current repository and local execution records; it does not claim access to unavailable lecture attachments or undocumented team contributions.

## 3. Reflections / Lessons Learned
Interface semantics matter more than action labels. The native five-action interface does not implement the required nine joint actions. The adapter updates both persistent targets and advances one decision interval; sequential native steps would change the timing and learning problem. Masks must constrain exploration, exploitation and bootstrap consistently. Structural legality alone cannot certify a safe traffic gap.

Termination semantics directly affect the target: collisions/off-road events stop bootstrap, whereas the 40-second limit preserves it. Saving the post-step observation before reset prevents learning from the next episode's initial state. Separate RNG streams and episode-boundary checkpoints make reproducibility testable; a 400-step regression compared continuous and resumed learning and obtained identical final weights.

Local records show 167 source regression tests passing. Smoke completed 2,000 environment steps with 251 updates; pilot completed 20,000 with 4,751. Formal seeds 11 and 22 each completed 50,000 steps with 12,251 updates. Seed 33 is incomplete: logs reach at least 19,396 steps, but its complete resume point is 15,178. These historical checkpoints are not required by, or included in, this submission.

Training is not monotonic: seed 11 had two collisions in its final six fixed validation scenarios, so the predefined selection rule retained an earlier checkpoint. Repeated speed-target changes and limited lane-changing also remain concerns. The reused speed controller applies acceleration (target speed - actual speed)/0.6 without a routine fixed acceleration cap; this strong response is a documented modelling limitation, not a realism or safety claim. Final-test scenarios, generalization and comparison with the gap-based baseline remain unfinished. Reproduction here uses existing dependencies, not a newly installed Ubuntu/Conda environment.

## 4. AI Use Declaration
ChatGPT/Codex assisted with specification organization, project-specific code generation, debugging, automated tests, execution-log inspection and document drafting/layout. Automated checks and observed results are reported explicitly. This does not establish that every group member has independently reviewed every line. Group 25 remains responsible for understanding, checking and submitting the work, confirming individual contributions and complying with course AI-use requirements. No unperformed human review or invented member reflection is asserted.

## References
[1] Farama Foundation, HighwayEnv documentation and source: https://highway-env.farama.org/ ; https://github.com/Farama-Foundation/HighwayEnv
[2] Farama Foundation, Gymnasium Env API: https://gymnasium.farama.org/api/env/
[3] PyTorch documentation and CPU installation guidance: https://docs.pytorch.org/docs/stable/ ; https://pytorch.org/get-started/locally/
