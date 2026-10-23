"""Behavioral checks for the proposal's environment contract.

The small hand-built traffic scenes below isolate perception, action constraints,
and episode endings. They do not rely on an agent learning a successful policy.
"""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium.error import ResetNeeded
from gymnasium.utils.env_checker import check_env
from highway_env.vehicle.kinematics import Vehicle

from highway_m1 import ACTION_NAMES, OBSERVATION_NAMES, SPEED_TARGETS, HighwayDecisionEnv


@pytest.fixture
def env():
    instance = HighwayDecisionEnv(config={"vehicles_count": 0, "duration": 60})
    instance.reset(seed=25)
    yield instance
    instance.close()


def place_ego(env, lane_id=1, speed_index=2, x=100.0):
    """Place the ego at a lane center with a consistent controller state."""
    index = ("0", "1", lane_id)
    lane = env.road.network.get_lane(index)
    ego = env.vehicle
    ego.position = lane.position(x, 0)
    ego.heading = lane.heading_at(x)
    ego.speed = float(SPEED_TARGETS[speed_index])
    ego.speed_index = speed_index
    ego.target_speed = float(SPEED_TARGETS[speed_index])
    ego.target_lane_index = index
    ego.crashed = False
    ego.on_state_update()
    return ego


def add_vehicle(env, lane_id, dx, speed):
    index = ("0", "1", lane_id)
    lane = env.road.network.get_lane(index)
    x = float(env.vehicle.position[0] + dx)
    vehicle = Vehicle(env.road, lane.position(x, 0), lane.heading_at(x), speed)
    env.road.vehicles.append(vehicle)
    return vehicle


def observe(env):
    return env.observation_type.observe()


def test_public_spaces_and_names(env):
    obs = observe(env)
    assert env.action_space.n == 9
    assert len(ACTION_NAMES) == 9
    assert len(OBSERVATION_NAMES) == 34
    assert obs.shape == (34,)
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    assert np.isfinite(obs).all()
    assert np.max(np.abs(obs)) <= 1


@pytest.mark.parametrize(
    "action, expected_lane, expected_speed_index",
    [(0, 0, 1), (1, 0, 2), (2, 0, 3),
     (3, 1, 1), (4, 1, 2), (5, 1, 3),
     (6, 2, 1), (7, 2, 2), (8, 2, 3)],
)
def test_nine_actions_apply_both_control_components(
    env, action, expected_lane, expected_speed_index
):
    place_ego(env)
    env.step(action)
    assert env.vehicle.target_lane_index[2] == expected_lane
    assert env.vehicle.speed_index == expected_speed_index
    assert env.vehicle.target_speed == SPEED_TARGETS[expected_speed_index]


@pytest.mark.parametrize(
    "lane, disabled", [(0, [0, 1, 2]), (2, [6, 7, 8])]
)
def test_action_mask_excludes_lane_edges(env, lane, disabled):
    place_ego(env, lane_id=lane)
    mask = env.action_mask()
    assert mask.shape == (9,)
    assert not mask[disabled].any()
    assert mask[4]
    assert int(mask.sum()) == 6


@pytest.mark.parametrize(
    "speed_index, disabled", [(0, [0, 3, 6]), (4, [2, 5, 8])]
)
def test_action_mask_excludes_target_speed_bounds(env, speed_index, disabled):
    place_ego(env, speed_index=speed_index)
    mask = env.action_mask()
    assert not mask[disabled].any()
    assert mask[4]
    assert int(mask.sum()) == 6


def test_action_mask_uses_commanded_speed_even_before_vehicle_reaches_it(env):
    place_ego(env, speed_index=4)
    env.vehicle.speed = float(SPEED_TARGETS[2])
    assert not env.action_mask()[[2, 5, 8]].any()
    assert env.action_mask()[[0, 3, 6]].all()


def test_in_flight_mask_and_straight_action_preserve_lane_change_target(env):
    place_ego(env)
    env.step(1)  # Left + maintain speed.
    target = env.vehicle.target_lane_index
    assert target[2] == 0
    assert env.action_mask()[[3, 4, 5]].all()
    assert not env.action_mask()[[0, 1, 2, 6, 7, 8]].any()
    env.step(4)  # Straight + maintain must continue the accepted maneuver.
    assert env.vehicle.target_lane_index == target


def test_real_lane_change_completes_and_reenables_next_lateral_command(env):
    place_ego(env)
    _, _, _, _, info = env.step(1)
    assert info["manoeuvre_in_progress"]
    # Ten seconds is a generous physical settling deadline on an empty road.
    for _ in range(50):
        _, _, terminated, truncated, info = env.step(4)
        assert not terminated and not truncated
        if not info["manoeuvre_in_progress"]:
            break
    else:
        pytest.fail("An accepted empty-road lane change never settled.")
    assert env.vehicle.lane_index[2] == 0
    assert abs(env.vehicle.position[1]) <= 0.2
    assert abs(env.vehicle.heading) <= 0.03
    assert env.action_mask()[[6, 7, 8]].all()
    assert not env.action_mask()[[0, 1, 2]].any()
    env.step(7)
    assert env.vehicle.target_lane_index[2] == 1


def test_action_mask_is_feasibility_only_and_does_not_hide_traffic_risk(env):
    place_ego(env)
    before = env.action_mask().copy()
    add_vehicle(env, lane_id=0, dx=0, speed=10)
    add_vehicle(env, lane_id=2, dx=3, speed=10)
    np.testing.assert_array_equal(env.action_mask(), before)
    assert env.action_mask()[1] and env.action_mask()[7]


@pytest.mark.parametrize("lateral_offset, heading", [(0.3, 0), (0, 0.05)])
def test_new_lane_command_waits_until_center_and_heading_have_settled(
    env, lateral_offset, heading
):
    place_ego(env)
    env.vehicle.position[1] += lateral_offset
    env.vehicle.heading = heading
    env.vehicle.on_state_update()
    # Crossing the nearest-lane boundary alone does not complete a maneuver.
    assert env.vehicle.lane_index == env.vehicle.target_lane_index
    assert not env.action_mask()[[0, 1, 2, 6, 7, 8]].any()
    assert env.action_mask()[[3, 4, 5]].all()


def test_nearest_front_rear_and_relative_velocity_observation(env):
    place_ego(env)
    add_vehicle(env, lane_id=1, dx=20, speed=15)
    add_vehicle(env, lane_id=1, dx=60, speed=30)
    add_vehicle(env, lane_id=1, dx=-30, speed=30)
    add_vehicle(env, lane_id=1, dx=-80, speed=10)
    lanes = observe(env)[:18].reshape(3, 6)
    np.testing.assert_allclose(
        # Two 5 m vehicles have 15 m / 25 m bumper clearances at these
        # center separations, distinct from center-to-center radar range.
        lanes[1], [1, 0.15, (15 - SPEED_TARGETS[2]) / 35,
                   1, 0.25, (30 - SPEED_TARGETS[2]) / 35], atol=1e-6
    )
    np.testing.assert_allclose(lanes[0], [0, 1, 0, 0, 1, 0])
    np.testing.assert_allclose(lanes[2], [0, 1, 0, 0, 1, 0])


def test_all_six_neighbour_slots_keep_absolute_lane_and_front_rear_order(env):
    place_ego(env)
    for lane_id, front_dx, rear_dx, front_speed, rear_speed in (
        (0, 15, -25, 14, 20),
        (1, 35, -45, 17, 9),
        (2, 55, -65, 23, 13),
    ):
        add_vehicle(env, lane_id, front_dx, front_speed)
        add_vehicle(env, lane_id, rear_dx, rear_speed)
    np.testing.assert_allclose(observe(env)[:18].reshape(3, 6), [
        [1, .10, (14 - 13.9) / 35, 1, .20, (20 - 13.9) / 35],
        [1, .30, (17 - 13.9) / 35, 1, .40, (9 - 13.9) / 35],
        [1, .50, (23 - 13.9) / 35, 1, .60, (13 - 13.9) / 35],
    ], atol=1e-6)


def test_perception_does_not_leak_vehicles_outside_100_m_radius(env):
    place_ego(env)
    without_traffic = observe(env).copy()
    # Each object is outside the radar disc; the side-lane object also tests
    # Euclidean range rather than longitudinal distance alone.
    add_vehicle(env, lane_id=1, dx=100.01, speed=0)
    add_vehicle(env, lane_id=1, dx=-100.01, speed=35)
    add_vehicle(env, lane_id=0, dx=99.99, speed=10)
    np.testing.assert_array_equal(observe(env), without_traffic)


def test_perception_does_not_read_another_drivers_hidden_target(env):
    from highway_env.vehicle.behavior import IDMVehicle

    place_ego(env)
    vehicle = IDMVehicle(env.road, np.array([130.0, 0.0]), speed=20)
    env.road.vehicles.append(vehicle)
    before = observe(env).copy()
    vehicle.target_lane_index = ("0", "1", 2)
    vehicle.target_speed = 30
    np.testing.assert_array_equal(observe(env), before)


def test_previous_action_is_one_hot_and_observation_remains_normalized(env):
    place_ego(env)
    obs, *_ = env.step(8)
    expected = np.zeros(9, dtype=np.float32)
    expected[8] = 1
    np.testing.assert_array_equal(obs[25:34], expected)
    assert env.observation_space.contains(obs)


def test_joint_lane_and_speed_change_pay_both_costs_once(env):
    place_ego(env)
    _, reward, _, _, info = env.step(8)
    components = info["reward_components"]
    assert components["lane_change"] == pytest.approx(-0.03)
    assert components["speed_change"] == pytest.approx(-0.01)
    assert reward == pytest.approx(sum(components.values()))
    # Continuing the same lateral maneuver and speed target is cost-free.
    _, _, _, _, continuing = env.step(4)
    assert continuing["reward_components"]["lane_change"] == 0
    assert continuing["reward_components"]["speed_change"] == 0


@pytest.mark.parametrize("speed_index, center_gap, expected", [
    (2, 15.425, -0.25),  # 10.425 m clearance = half of 1.5 * 13.9 m/s.
    (2, 25.85, 0.0),    # Exactly 1.5 s clearance has no deficit.
    (2, 6.0, -397 / 834),  # 1 m clearance at 13.9 m/s.
    (0, 6.0, 0.0),      # A stopped ego must not divide by zero.
])
def test_analytical_time_headway_penalty(env, speed_index, center_gap, expected):
    place_ego(env, speed_index=speed_index)
    add_vehicle(env, lane_id=1, dx=center_gap, speed=SPEED_TARGETS[speed_index])
    _, reward, terminated, truncated, info = env.step(4)
    assert not terminated and not truncated
    assert np.isfinite(reward)
    assert info["reward_components"]["headway"] == pytest.approx(expected, abs=1e-10)


def test_no_front_neighbour_has_no_time_headway_cost(env):
    place_ego(env)
    add_vehicle(env, lane_id=1, dx=-6, speed=SPEED_TARGETS[2])
    _, _, _, _, info = env.step(4)
    assert info["reward_components"]["headway"] == 0


def test_mask_invalid_joint_command_falls_back_atomically(env):
    place_ego(env, lane_id=0)
    # A left + faster command is structurally invalid as a whole. It must
    # neither start a lane change nor silently execute its faster component.
    obs, _, _, _, info = env.step(2)
    assert info["invalid_action"]
    assert info["requested_action"] == 2
    assert info["executed_action"] == 4
    assert env.vehicle.target_lane_index[2] == 0
    assert env.vehicle.speed_index == 2
    assert info["reward_components"]["lane_change"] == 0
    assert info["reward_components"]["speed_change"] == 0
    assert obs[25 + 4] == 1
    assert obs[25 + 2] == 0


def test_reset_seed_reproduces_traffic_and_action_rollout():
    env = HighwayDecisionEnv(config={"vehicles_count": 10, "duration": 10})
    try:
        first, _ = env.reset(seed=5418)
        states = [(v.position.copy(), v.speed) for v in env.road.vehicles]
        rollout = [env.step(action)[:4] for action in [4, 5, 4, 3, 4]]
        second, _ = env.reset(seed=5418)
        np.testing.assert_array_equal(first, second)
        for (position, speed), vehicle in zip(states, env.road.vehicles):
            np.testing.assert_array_equal(position, vehicle.position)
            assert speed == vehicle.speed
        replay = [env.step(action)[:4] for action in [4, 5, 4, 3, 4]]
        for actual, expected in zip(replay, rollout):
            np.testing.assert_array_equal(actual[0], expected[0])
            assert actual[1:] == expected[1:]
        env.reset(seed=5419)
        assert any(not np.array_equal(position, vehicle.position)
                   for (position, _), vehicle in zip(states, env.road.vehicles))
    finally:
        env.close()


def test_each_decision_runs_exactly_three_physics_ticks(env, monkeypatch):
    place_ego(env)
    ticks = []
    original_step = env.road.step

    def record_step(dt):
        ticks.append(dt)
        return original_step(dt)

    monkeypatch.setattr(env.road, "step", record_step)
    before = env.time
    env.step(4)
    np.testing.assert_allclose(ticks, [1 / 15, 1 / 15, 1 / 15], atol=1e-12)
    assert env.time - before == pytest.approx(0.2)


def test_actual_acceleration_limits_target_tracking_braking_and_stop(env, monkeypatch):
    place_ego(env)
    samples = []
    original_step = env.vehicle.step

    def record_vehicle_step(dt):
        before = env.vehicle.speed
        original_step(dt)
        samples.append((before, env.vehicle.speed, dt, env.vehicle.action["acceleration"]))

    monkeypatch.setattr(env.vehicle, "step", record_vehicle_step)
    env.step(5)
    env.step(5)  # The second command advances the pending speed target again.
    assert env.vehicle.target_speed == pytest.approx(27.8)
    for _ in range(40):
        env.step(4)
    assert env.vehicle.speed == pytest.approx(27.8, abs=0.01)
    for _ in range(4):
        env.step(3)
    assert env.vehicle.target_speed == 0
    for _ in range(65):
        env.step(4)
    assert 0 <= env.vehicle.speed < 0.01
    # Check the state derivative at every physics tick, not merely telemetry
    # from the final tick of a policy decision.
    accelerations = [(after - before) / dt for before, after, dt, _ in samples]
    assert min(accelerations) == pytest.approx(-5)
    assert max(accelerations) == pytest.approx(3)
    assert all(-5 - 1e-10 <= value <= 3 + 1e-10 for value in accelerations)
    assert all(-5 <= command <= 3 for _, _, _, command in samples)
    assert all(after >= 0 for _, after, _, _ in samples)


def test_real_overlap_causes_collision_termination(env):
    place_ego(env)
    add_vehicle(env, lane_id=1, dx=0, speed=SPEED_TARGETS[2])
    _, reward, terminated, truncated, _ = env.step(4)
    assert env.vehicle.crashed
    assert terminated and not truncated
    assert reward < 0
    with pytest.raises(ResetNeeded):
        env.step(4)


@pytest.mark.parametrize("ending, expected_reward", [("collision", -20), ("offroad", -20),
                                                     ("both", -40)])
def test_terminal_penalties_have_exact_negative_values_when_stationary(
    env, ending, expected_reward
):
    place_ego(env, speed_index=0)
    if ending in ("collision", "both"):
        add_vehicle(env, lane_id=1, dx=0, speed=0)
    if ending in ("offroad", "both"):
        env.vehicle.position[1] = -20
        env.vehicle.on_state_update()
        if ending == "both":
            # Put the second stationary object at the same off-road position.
            env.road.vehicles[-1].position = env.vehicle.position.copy()
            env.road.vehicles[-1].on_state_update()
    _, reward, terminated, truncated, info = env.step(4)
    assert terminated and not truncated
    assert reward == pytest.approx(expected_reward)
    assert info["reward_components"]["collision"] == (-20 if ending != "offroad" else 0)
    assert info["reward_components"]["offroad"] == (-20 if ending != "collision" else 0)


def test_terminal_decision_still_advances_exactly_three_physics_ticks(env, monkeypatch):
    place_ego(env, speed_index=0)
    add_vehicle(env, lane_id=1, dx=0, speed=0)
    ticks = []
    original_step = env.road.step

    def record_step(dt):
        ticks.append(dt)
        return original_step(dt)

    monkeypatch.setattr(env.road, "step", record_step)
    _, _, terminated, _, info = env.step(4)
    assert terminated
    np.testing.assert_allclose(ticks, [1 / 15] * 3, atol=1e-12)
    assert info["simulation_steps"] == 3
    assert info["simulation_time_s"] == pytest.approx(0.2)
    assert info["elapsed_time"] == pytest.approx(0.2)


def test_off_road_causes_termination(env):
    place_ego(env)
    env.vehicle.position[1] = -20
    env.vehicle.on_state_update()
    _, _, terminated, truncated, _ = env.step(4)
    assert terminated and not truncated
    with pytest.raises(ResetNeeded):
        env.step(4)


def test_horizon_is_truncation_and_requires_reset():
    env = HighwayDecisionEnv(config={"vehicles_count": 0, "duration": 0.2})
    try:
        env.reset(seed=1)
        _, _, terminated, truncated, _ = env.step(4)
        assert truncated and not terminated
        with pytest.raises(ResetNeeded):
            env.step(4)
        env.reset(seed=1)
        env.step(4)
    finally:
        env.close()


def test_default_40_second_horizon_ends_at_200_decisions(env):
    env.configure({"duration": 40})
    place_ego(env)
    for _ in range(199):
        _, _, terminated, truncated, _ = env.step(4)
        assert not terminated and not truncated
    _, _, terminated, truncated, info = env.step(4)
    assert truncated and not terminated
    assert info["elapsed_time"] == pytest.approx(40)


@pytest.mark.parametrize("action", [-1, 9, 2.5, "4"])
def test_out_of_space_action_is_rejected_before_advancing(env, action):
    position = env.vehicle.position.copy()
    with pytest.raises(ValueError):
        env.step(action)
    np.testing.assert_array_equal(env.vehicle.position, position)
    assert env.time == 0


@pytest.mark.parametrize("key, value", [
    ("sensing_radius", 0), ("sensing_radius", -1),
    ("sensing_radius", np.nan), ("sensing_radius", np.inf),
    ("headway_seconds", 0), ("headway_seconds", -1),
    ("headway_seconds", np.nan), ("headway_seconds", np.inf),
    ("settled_lateral_tolerance", -0.01),
    ("settled_lateral_tolerance", np.nan), ("settled_lateral_tolerance", np.inf),
    ("settled_heading_tolerance", -0.01),
    ("settled_heading_tolerance", np.nan), ("settled_heading_tolerance", np.inf),
    ("duration", np.nan), ("duration", np.inf),
    ("vehicles_count", 1.5), ("vehicles_count", np.inf),
    ("initial_speed_index", 1.5),
    ("simulation_frequency", 15.5), ("policy_frequency", 0),
    ("traffic_speed_range", [18, np.inf]),
    ("traffic_density_range", [np.nan, 1.2]),
    ("manual_control", True),
])
def test_invalid_configuration_is_rejected(key, value):
    with pytest.raises(ValueError):
        HighwayDecisionEnv(config={"vehicles_count": 0, key: value})


def test_valid_zero_settling_tolerances_and_positive_numeric_limits():
    env = HighwayDecisionEnv(config={
        "vehicles_count": 0, "sensing_radius": 50, "headway_seconds": 2,
        "settled_lateral_tolerance": 0, "settled_heading_tolerance": 0,
    })
    try:
        observation, _ = env.reset(seed=25)
        assert env.observation_space.contains(observation)
        assert np.isfinite(env.step(4)[1])
    finally:
        env.close()


def test_rgb_render_is_an_image():
    env = HighwayDecisionEnv(config={"vehicles_count": 0}, render_mode="rgb_array")
    try:
        env.reset(seed=25)
        frame = env.render()
        assert frame.ndim == 3 and frame.shape[2] == 3
        assert frame.dtype == np.uint8
        assert frame.shape[0] >= 100 and frame.shape[1] >= 200
        assert frame.max() > frame.min()
    finally:
        env.close()


def test_gymnasium_environment_contract():
    env = HighwayDecisionEnv(config={"vehicles_count": 0})
    try:
        check_env(env, skip_render_check=True)
    finally:
        env.close()
