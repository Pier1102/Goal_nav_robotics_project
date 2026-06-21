# Collision Avoidance and Goal Navigation Based on Deep Reinforcement Learning

ROS 2 / Gazebo workspace for a simplified STORM skid-steer robot, replicating
the DDQN-based collision-avoidance method of Feng et al. (2021) and extending
it with goal-directed navigation via curriculum learning across three maps.
A classical TEB local planner is also benchmarked for comparison.

Full methodology, results, and discussion are presented in the accompanying
report.

## Repository structure

```
ros_ws/
└── src/
    └── storm_description/
        ├── urdf/
        │   └── storm_robot.urdf.xacro     # simplified skid-steer robot model
        ├── worlds/
        │   ├── test_map_1.world           # original test environment
        │   ├── test_map_2.world           # complex circuit (deployment target)
        │   └── test_map_3.world           # open environment (curriculum start)
        ├── launch/
        │   └── sim.launch.py              # Gazebo + bridge + RViz launch
        └── goal_dqn/
            ├── envs_maps/
            │   ├── storm_env_map1.py      # Gymnasium env, Map 1
            │   ├── storm_env_map2.py      # Gymnasium env, Map 2
            │   ├── storm_env_map3.py      # Gymnasium env, Map 3
            │   ├── train_map3.py          # curriculum Phase 1 (Map 3)
            │   ├── train_map2.py          # curriculum Phase 2 (Map 2, warm-start)
            │   └── genera_semi.py         # spawn-point seed generator
            ├── test_codes/
            │   ├── test_ddqn_map1.py      # zero-shot evaluation on Map 1
            │   ├── test_dqn_map2.py       # native evaluation on Map 2
            │   └── test_dqn_map3.py       # evaluation on Map 3
            └── storm_agents/              # saved network weights (.h5)
```

`build/`, `install/`, and `log/` are ROS 2 colcon artifacts and are not
tracked in version control (see `.gitignore`).

## Requirements

- Ubuntu 22.04
- ROS 2 Humble
- Ignition/Gazebo (`ros_gz_sim`, `ros_gz_bridge`)
- Python 3.10
- Python packages: `tensorflow`, `gymnasium`, `numpy`

```bash
pip install tensorflow gymnasium numpy
```

ROS 2 dependencies (`ros_gz_sim`, `ros_gz_bridge`, `robot_state_publisher`,
`xacro`) are declared in `package.xml` and are expected to already be
available in a standard ROS 2 Humble + Gazebo installation.

## Build

```bash
cd ros_ws
colcon build --packages-select storm_description
source install/setup.bash
```

## Running the simulation

`launch/sim.launch.py` starts Gazebo, spawns the robot, and bridges the
relevant topics to ROS 2. The active map is selected by editing the
`world_path` variable at the top of the file (only one map is active at a
time):

```python
world_path = os.path.join(pkg_path, 'worlds', 'test_map_2.world')
```

Then launch:

```bash
ros2 launch storm_description sim.launch.py
```

## Training

Training scripts are run from `goal_dqn/envs_maps/`, with the corresponding
map already running in Gazebo (see above).

**Phase 1 — Map 3 (open environment, curriculum start):**

```bash
cd goal_dqn/envs_maps
python3 train_map3.py
```

Produces `storm_agents/map3_FINAL.weights.h5`, plus periodic checkpoints
(`map3_ckpt_ep*.weights.h5`).

**Phase 2 — Map 2 (complex circuit, warm-started from Phase 1):**

Set `LOAD_WEIGHTS` at the top of `train_map2.py` to the Phase 1 weights
(`map3_FINAL.weights.h5`), then:

```bash
python3 train_map2.py
```

Produces `storm_agents/map2_FINAL.weights.h5`.

Training progress (reward, loss, goal rate, epsilon) is logged to
TensorBoard:

```bash
tensorboard --logdir ~/storm_dqn/logs
```

Both scripts save weights automatically on `Ctrl+C` interruption
(`*_interrupt.weights.h5`).

## Evaluation

Evaluation scripts load a saved policy and run greedy (no-exploration)
episodes from fixed spawn points, with the corresponding map running in
Gazebo. Run as modules from `goal_dqn/`:

```bash
cd goal_dqn
python3 -m test_codes.test_ddqn_map1   # zero-shot evaluation on Map 1
python3 -m test_codes.test_dqn_map2    # native evaluation on Map 2
python3 -m test_codes.test_dqn_map3    # evaluation on Map 3
```

By default, the scripts load `storm_agents/map2_FINAL.weights.h5` (the
final Phase 2 policy); edit the `WEIGHTS_PATH` variable at the top of each
script to evaluate a different checkpoint.

## Pre-trained weights

`goal_dqn/storm_agents/` contains the weights used to produce the results
in the report:

- `map3_FINAL.weights.h5` — end of curriculum Phase 1
- `map2_FINAL.weights.h5` — end of curriculum Phase 2 (final policy)
- intermediate checkpoints (`map3_ckpt_ep*`, `map2_ckpt_ep*`)

## Notes

- The robot model is a simplified skid-steer platform (cylindrical base +
  four continuous-joint wheels + simulated LiDAR), not a full replication
  of the original STORM mechanical design — see the report for details and
  justification.
- Training was conducted on a single consumer laptop with Gazebo running
  inside a virtual machine; runs took approximately 24–72 hours each. See
  the Limitations section of the report for details.
