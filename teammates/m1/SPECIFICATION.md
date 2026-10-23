# Group 25 Environment Specification

This document freezes the implementation choices for the OpenAI Gym code and report milestone. It implements the revised nine-action proposal, with the additional numeric choices below. The choices have not been tuned against final-test performance.

## Scenario and time

- Three straight, same-direction lanes, indexed left to right as 0, 1 and 2.
- One controlled vehicle and initially 20 surrounding IDM/MOBIL vehicles.
- Ego starts in lane 1 with actual and target speed 20.85 m/s (index 3).
- Per-reset traffic density is uniformly sampled from [0.8, 1.2]; initial surrounding speeds and their desired speeds are uniformly sampled from [18, 30] m/s. Built-in seeded placement randomizes gaps. These are development ranges, not a generalization result.
- Dynamics run at 15 Hz, decisions at 5 Hz, decision interval 0.2 s. In the installed HighwayEnv 1.12.1, every action advances three physics ticks, including the terminal decision; episode termination is evaluated after that interval.
- Collision or leaving the road sets `terminated=True`. Otherwise, 40 s (200 decisions) sets `truncated=True`. Termination takes precedence if a failure occurs on the final decision. No target-speed success terminal is used. A completed episode must be reset before stepping again.
- `info.elapsed_time` is decision-clock time, `step_count / 5`; `simulation_steps` records actual dynamics ticks and `simulation_time_s` is `simulation_steps / 15`. The two clocks agree in this implementation and are both recorded to make the integration timing auditable.
- Ego control uses HighwayEnv's lane/speed tracker with acceleration clamped to [-5, 3] m/s^2. Ego speed bounds are [0, 35] m/s. These bounds are project choices; they do not certify comfort or real-vehicle feasibility.

## Nine target commands

Action ID is `3 * lateral_index + longitudinal_index`.

| Lateral | Slower | Hold target speed | Faster |
| --- | --- | --- | --- |
| Left | 0 | 1 | 2 |
| Keep target lane | 3 | 4 | 5 |
| Right | 6 | 7 | 8 |

Speed targets are `[0, 6.95, 13.90, 20.85, 27.80]` m/s. Slower and faster change the **existing target index** by -1 and +1. The actual speed tracks the target dynamically. Keep retains the target lane during an unfinished change.

A manoeuvre is in progress when physical and target lane differ, lateral offset from the target lane exceeds 0.2 m, or heading error exceeds 0.03 rad. The structural mask excludes new lateral changes during this interval, lane changes beyond lane 0/2, and speed-index under/overflow. Action 4 is always available. The mask does not read surrounding vehicles or reject dangerous gaps. A mask-invalid in-range ID maps atomically to action 4, preserving both pending targets. `info` records requested/executed IDs and `invalid_action`. Values outside Discrete(9) raise `ValueError`.

## Observation vector

The vector has dtype float32, shape `(34,)`, declared Box bounds [-1, 1], and feature order exported as `OBSERVATION_NAMES`. The raw full simulator state is not part of the observation.

| Indices | Definition and normalization |
| --- | --- |
| 0,1,2 | Lane 0 nearest front: presence, bumper gap / 100, (other vx - ego vx) / 35 |
| 3,4,5 | Lane 0 nearest rear: same three features |
| 6-11 | Lane 1 front then rear, in the same order |
| 12-17 | Lane 2 front then rear, in the same order |
| 18 | Ego longitudinal velocity / 35 |
| 19 | Ego lateral velocity / 35 |
| 20 | Offset from the physical current lane centre / half lane width |
| 21 | Physical current lane index / 2 |
| 22 | Target lane index / 2 |
| 23 | Target speed / 27.8 |
| 24 | Manoeuvre-in-progress flag, 0 or 1 |
| 25-33 | Previous **executed** action one-hot; all zero at reset because no previous action exists |

Gap values and lane encodings are clipped to [0,1] where relevant; signed velocities and offset are clipped to [-1,1]. Surrounding vehicles are sensed only inside a Euclidean centre-distance radius of 100 m. Nearest front/rear is selected by absolute longitudinal distance in each measured physical lane. Bumper gap is `max(0, abs(dx) - (ego_length + other_length)/2)`. Front includes `dx=0`; rear means `dx<0`. An absent neighbour has presence 0, normalized gap 1, and relative speed 0. Both front and rear relative-speed signs mean other minus ego. No future trajectory, target lane, desired speed or hidden intention of another vehicle is read by the encoder.

Physical nearest-lane assignment can switch during another vehicle's lane change. It is a deliberately compact local sensor model, not a complete occupancy envelope. The seven ego/control features and previous action reduce ambiguity but do not remove partial observability.

## Reward per decision

`reward = progress + collision + offroad + headway + lane_change + speed_change`

| Component | Value |
| --- | --- |
| progress | `clip(delta_x / (27.8 * 0.2), 0, 1)` at default frequency |
| collision | -20 if ego crashed |
| offroad | -20 if ego is outside the road |
| headway | `-0.5 * max(0, 1 - front_gap/(1.5*vx))` if a front vehicle is sensed and forward speed is positive; otherwise 0 |
| lane_change | -0.03 only when a valid lateral target change is initiated |
| speed_change | -0.01 only when a valid target-speed index change occurs |

Headway uses the nearest sensed front vehicle in the ego's current physical lane at the end of the decision. It is a training signal, not a traffic-risk action filter. It can miss target-lane risk before lane crossing; evaluating and improving this modelling choice is later work. Collision and off-road penalties can both apply if both events occur. Both action costs apply to a joint command. `info.reward_components` exposes every signed term. Acceleration and jerk penalties are not included in this milestone.

## Evidence and scope

`verify.py` checks Gymnasium compatibility, RGB shape and exact repeatability of a seeded random-action rollout. `tests/test_environment.py` checks physical edge cases and contract behavior. `demo.py` records observations, commands, masks and reward terms with a reproducible untrained scripted/random/hold policy. Diagnostic tests can set no surrounding vehicles or a shorter duration; the demo uses the default proposal scenario.

The package does not contain a trained agent, claim improved safety, or establish generalization. Windows execution is verified in the recorded evidence. Ubuntu execution remains to be verified in the team's environment. For later DQN, valid masks must be used in exploration, greedy choice and the next-state maximum; bootstrap through time truncation, but not true termination.
