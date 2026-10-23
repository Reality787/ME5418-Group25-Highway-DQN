# ME5418 Group 25 requirements and project roadmap

This roadmap distinguishes the delivered environment milestone from planned learning and evaluation work. Dates and submission details below reflect the Canvas pages inspected on 15 October 2026; the current assignment page takes precedence. The revised proposal remains the design reference. This document is not a claim that later work is complete or a prediction of a grade.

## Assessment and deliverables

Lecture 1 pages 8 and 10 specify weekly quizzes 10%, proposal and peer feedback 10%, environment C&R 15%, neural network C&R 15%, learning agent C&R 15%, and final project 35%. These are component weights, not a detailed marking rubric.

The authenticated Canvas Grades page confirms the same group weights. Lecture 6b, *OpenAI Gym*, pages 3-9 separates the environment, neural network and learning agent, and asks the environment to define reset/step, spaces, rewards, episode randomization, internal state and rendering. The delivered environment follows that separation using the modern Gymnasium reset/step contract. Lectures 8+9, *Neural Network Design + Learning Agent*, pages 4-10 additionally require explicit data flow, loss/gradient updates, episode experience collection, training logs and model saving; those belong to the next milestones.

| Stage | Deadline Singapore time | Formal deliverable inspected on Canvas |
| --- | --- | --- |
| Environment C&R | 16 Oct 2026 23:59 | One group ZIP/RAR/7z with commented environment code, a graphical demo, environment.yml, and a common two-page report. Include run instructions in README or the report. A random/deterministic policy is allowed. |
| Neural Network C&R | 30 Oct 2026 23:59 | Network code, a working forward/backward-pass demo, environment.yml, and a common two-page report. |
| Learning Agent C&R | 6 Nov 2026 23:59 | Runnable integrated learning code, environment.yml and a common two-page report. Successful training is not yet required. |
| Final Project | 20 Nov 2026 23:59 | Training and testing code, at least one trained functional model, dependency configuration, README, group video and each member's independently written report of at most ten pages. |

For the final submission, the inspected page's deadline header was 20 November in the 2026 course; the body retained an old 2025 year. Its availability until 22 November is not the stated deadline. Verify the live assignment before submission.

## Delivered environment and remaining acceptance work

The current environment implements three lanes, initially 20 surrounding vehicles, 34 local/control/history features, nine joint target commands, the five target-speed levels, a structural mask, decomposed rewards, and collision/off-road termination separate from 40-second truncation. It exposes action masks and final observations for later learning code. The report, dependency pins, tests, video, CSV and JSON evidence are packaged together.

Windows execution, seeded replay, Gymnasium checks and the semantic tests in `tests/` have been executed. The number of passing cases and exact dependency versions are recorded in `artifacts/validation.json`. The scripted and random rollouts are functional development demonstrations, not learned-policy performance measurements.

Before submission, reproduce on Ubuntu from the extracted ZIP with a new environment, run `verify.py`, pytest and the graphical demo, and preserve the actual outputs. Ubuntu has not been verified by the delivered Windows run. Review the two-page report against the actual Ubuntu result before changing its platform claims.

The source, report and external-library choices must be understood and checked by the team. AI use is disclosed honestly. Disclosure alone does not establish the course's independent project-code contribution requirement. Keep records of real member-authored implementation, substantive fixes, modelling choices and analysis; do not invent percentages or contributors.

## Neural network milestone plan

- Implement the proposal's MLP `34 -> 128 ReLU -> 128 ReLU -> 9`, using PyTorch. Preserve `OBSERVATION_NAMES` and `ACTION_NAMES` as the interface contract.
- Demonstrate an input batch of actual environment observations, nine Q outputs, a finite scalar loss, backward gradients, an optimizer step and an actual parameter change. Label any synthetic targets as diagnostic data.
- Check shape, dtype, finite values, save/load equality and CPU execution. Add a separate optional learning dependency configuration; keep the environment-only installation lightweight.
- Explain why this numerical observation uses an MLP, why there are nine outputs, and why a forward pass alone does not demonstrate learning.

## Learning agent milestone plan

- Implement replay memory, epsilon-greedy exploration, target network, TD updates, a training loop and validation logging. Use discount 0.99 as proposed.
- Store `(observation, action, reward, next_observation, terminated, truncated, next_action_mask)` explicitly. Explore and act greedily only over valid actions. Mask the next-state maximum too.
- Use `reward + gamma * (1-terminated) * max_valid Q_target(next_observation)` for the proposed continuing task with an external 40-second cutoff. Time-limit truncation still bootstraps; a true terminal failure does not. Preserve the final observation before any reset.
- Log reward components, TD loss, epsilon, gradient/parameter changes, crash/off-road/completion outcomes and actual environment steps. Check a short real parameter-update run before launching large training budgets.
- Save model/configuration/feature/action mappings. If exact resumption is claimed, also save optimizer, RNG, replay and training-step state; weights alone support inference, not exact resumption.

## Final experiment plan

The teaching-team email prioritised a working, generalizable RL implementation before elaborate baseline or ablation work. The proposal commits to one economical non-learning gap-based reference, which can test the value of learning without implementing many competing algorithms.

1. Establish development, validation and final-test seed namespaces before tuning. All scenarios already viewed in this package are development data. Final-test seeds must remain fresh until policies and rule parameters are frozen.
2. Train three independent seeds, randomizing density, initial gaps and traffic speeds in documented development ranges. The current generator starts traffic ahead of ego; deliberately varied front/rear traffic scenarios are later work, not a completed claim.
3. Select training budget, checkpoint and reward/rule parameters using development and validation only. Record selections before opening final tests.
4. Compare unfiltered DQN and a gap-based rule controller with identical 34-feature observations, nine actions, actuator limits and matched initial scenarios. Built-in IDM/MOBIL traffic behavior is not itself a matched ego baseline.
5. As proposed, run 100 matched episodes per scenario group and training seed: fresh within-range settings, withheld combinations and selected out-of-range settings. Publish the actual ranges and scenario IDs; avoid changing them after results are inspected.
6. Primary metrics: collision rate, completion rate and `max(0, x_end-x_start)/(27.8*40)` progress. The denominator remains the full horizon even for early crashes. Separately report off-road endings, mean speed/headway, lane-change count and time to failure. Acceleration/jerk should be measured at dynamics frequency if comfort claims are made.
7. Show learning curves and variability across training seeds. Distinguish episode-level collision uncertainty from training-seed uncertainty. Discuss both successful behavior and failure cases; a single attractive video is not a generalization result.
8. Add one focused ablation only after reliable learning, such as randomized versus fixed-condition training. Reward variants require retraining at matched budgets. If a safety filter is studied, compare both filtered/unfiltered DQN and a non-learning control with the same filter.

## Peer-feedback follow-through

The September peer comments concern the original five-action, 30-feature proposal. The revised proposal and this environment use nine joint actions and 34 features. Peer suggestions are useful review evidence, not additional compulsory grading rules.

| Feedback theme | Current response or later acceptance condition |
| --- | --- |
| Reward weights and frequent speed changes | Exact initial weights and the target-speed-change cost are implemented and logged. Choose later changes using development/validation only; sensitivity claims need retrained comparisons. |
| Commands during lane changes | The structural mask forbids a new lateral command until the current manoeuvre settles. Keep preserves the target lane. Requested/executed commands are logged. |
| Library versus project contribution | Report and source identify reused dynamics, IDM/MOBIL traffic, low-level control and rendering, and the custom observation/joint-action/mask/reward layer. AI generation is declared. |
| Generalization claim too broad | Evaluate density, gap and speed shifts on the same three-lane geometry. Do not claim transfer to lane closures, different road layouts or arbitrary lane counts. A different encoder would be a later scope change. |
| Simple reference may be weak | Deliver the proposed matched gap controller first. If time permits, add a stronger conventional IDM/MOBIL-inspired ego controller constrained to the same observation and actuator interface, and a keep-lane reference. Built-in NPC behavior is not automatically a fair ego comparison. Beating a simple reference alone cannot establish that RL is necessary or superior to conventional control generally. |
| Stopping target and limited sensing | Retain the revised proposal's five speed levels and 100 m sensing for this milestone. Diagnose stop frequency, failure cases and observable delayed trade-offs before changing the task; do not infer unseen future traffic from the discount factor. |

Do not add new algorithms or an attention encoder solely to increase apparent complexity before the promised DQN and evaluation are reliable. One controlled, interpretable study is the next extension priority.

## Final report and contribution evidence

The final report is individual, even though code and videos are shared. The inspected instructions prohibit copying prose from the proposal, prior reports or teammates; emphasize new/changed methods, results, limitations and lessons, and refer to earlier reports for unchanged definitions. Shared figures can be reused under those instructions. Describe actual personal contributions and acknowledge external code and AI use. Do not claim safety, superiority or generalization beyond the tested simulation conditions.

## Source entry points

- Environment assignment: https://canvas.nus.edu.sg/courses/97711/assignments/269473
- Network assignment: https://canvas.nus.edu.sg/courses/97711/assignments/269474
- Agent assignment: https://canvas.nus.edu.sg/courses/97711/assignments/269498
- Final assignment: https://canvas.nus.edu.sg/courses/97711/assignments/269500
- Approved proposal comments: https://canvas.nus.edu.sg/courses/97711/assignments/269472
- OpenAI Gym lecture: https://canvas.nus.edu.sg/courses/97711/files/folder/Lecture%20Slides?preview=10080072
- NN and Learning Agent lecture: https://canvas.nus.edu.sg/courses/97711/files/folder/Lecture%20Slides?preview=10347419
- HighwayEnv custom environment guidance: https://highway-env.farama.org/make_your_own/
- Gymnasium termination and truncation: https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/
