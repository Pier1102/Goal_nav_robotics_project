import time
import math
import random
import threading
import subprocess
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

# ---------------------------------------------------------------------------
# Hardware constants
# ---------------------------------------------------------------------------
N_LIDAR        = 50
RANGE_MAX      = 5.0
N_ACTIONS      = 11
LINEAR_VEL     = 0.2
ANGULAR_VELS   = [-1.5 + 0.3 * i for i in range(11)]
COLLISION_DIST = 0.20
SPAWN_Z        = 0.05
WORLD_NAME     = 'test_map_1'

# ---------------------------------------------------------------------------
# Goal — Map 1
# ---------------------------------------------------------------------------
GOAL_X      = -0.7
GOAL_Y      =  0.59
GOAL_RADIUS =  0.7

# ---------------------------------------------------------------------------
# Spawn points
# ---------------------------------------------------------------------------
SPAWN_ALL = [
    ( 1.1190, -1.8742,  3.14),
    ( 2.2000,  1.3000, -1.57),
    (-0.8500, -1.8000,  1.57),
    (-2.3000, -2.7000,  1.57),
]

# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------
R_GOAL      = +200.0
R_COLLISION = -200.0
R_STEP      =   -0.1
W_PROGRESS  =   15.0
W_HEADING   =    0.3

# ---------------------------------------------------------------------------
# Odometry clamp (filters wheel-slip spikes, see _get_goal_data)
# ---------------------------------------------------------------------------
ODOM_CLAMP       = 15.0   # global safety clamp on total displacement
DELTA_CLAMP      =  0.10  # clamp on the progress reward term (Δdist)
STEP_DELTA_CLAMP =  0.05  # per-step clamp on raw odometry displacement


# ---------------------------------------------------------------------------
# ROS 2 node
# ---------------------------------------------------------------------------
class _RosNode(Node):
    def __init__(self):
        super().__init__('storm_dqn_map1')
        self.pub_vel = self.create_publisher(Twist, '/model/storm/cmd_vel', 10)

        self._scan      = None
        self._scan_lock = threading.Lock()
        self._scan_seq  = 0
        self.create_subscription(LaserScan, '/model/storm/scan', self._cb_scan, 10)

        self._odom      = None
        self._odom_lock = threading.Lock()
        self.create_subscription(Odometry, '/model/storm/odometry', self._cb_odom, 10)

    def _cb_scan(self, msg):
        with self._scan_lock:
            self._scan = msg
            self._scan_seq += 1

    def _cb_odom(self, msg):
        with self._odom_lock:
            self._odom = msg

    def get_scan(self):
        with self._scan_lock:
            return self._scan

    def get_scan_seq(self):
        with self._scan_lock:
            return self._scan_seq

    def get_odom(self):
        with self._odom_lock:
            return self._odom

    def pub_cmd(self, lin, ang):
        msg = Twist()
        msg.linear.x  = float(lin)
        msg.angular.z = float(ang)
        self.pub_vel.publish(msg)

    def _pause(self, paused: bool):
        val = 'true' if paused else 'false'
        cmd = ['ign', 'service',
               '-s', f'/world/{WORLD_NAME}/control',
               '--reqtype', 'ignition.msgs.WorldControl',
               '--reptype', 'ignition.msgs.Boolean',
               '--timeout', '2000',
               '--req', f'pause: {val}']
        try:
            subprocess.run(cmd, timeout=3.0, capture_output=True, text=True)
        except Exception:
            pass

    def set_pose(self, x, y, z, qx, qy, qz, qw):
        req = (f'name: "storm" '
               f'position: {{x: {x} y: {y} z: {z}}} '
               f'orientation: {{x: {qx} y: {qy} z: {qz} w: {qw}}}')
        cmd = ['ign', 'service',
               '-s', f'/world/{WORLD_NAME}/set_pose',
               '--reqtype', 'ignition.msgs.Pose',
               '--reptype', 'ignition.msgs.Boolean',
               '--timeout', '3000',
               '--req', req]
        try:
            self._pause(True)
            time.sleep(0.3)
            subprocess.run(cmd, timeout=4.0, capture_output=True, text=True)
            time.sleep(0.3)
            subprocess.run(cmd, timeout=4.0, capture_output=True, text=True)
            time.sleep(0.3)
            self._pause(False)
            time.sleep(0.5)
        except subprocess.TimeoutExpired:
            print('  [set_pose] timeout!')
            self._pause(False)
        except Exception as e:
            print(f'  [set_pose] error: {e}')
            self._pause(False)


# ---------------------------------------------------------------------------
# StormEnv — Map 1
# ---------------------------------------------------------------------------
class StormEnv(gym.Env):

    def __init__(self, max_steps=1500):
        super().__init__()

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(N_LIDAR + 2,), dtype=np.float32)
        self.action_space = spaces.Discrete(N_ACTIONS)

        self.max_steps  = max_steps
        self.step_count = 0
        self.dist_prev  = 0.0
        self.spawn_x    = 0.0
        self.spawn_y    = 0.0
        self.odom_x0    = 0.0
        self.odom_y0    = 0.0
        self.ep_count   = 0

        # Incremental clamp state, updated in _get_goal_data()
        self._last_odom_raw_x = None
        self._last_odom_raw_y = None
        self._est_dx = 0.0
        self._est_dy = 0.0

        self._last_x_robot = 0.0
        self._last_y_robot = 0.0
        self._last_yaw     = 0.0

        if not rclpy.ok():
            rclpy.init()
        self._node = _RosNode()
        threading.Thread(target=rclpy.spin, args=(self._node,), daemon=True).start()

        print(f'[StormEnv map1] Waiting for sensors...')
        t0 = time.time()
        while self._node.get_scan() is None or self._node.get_odom() is None:
            time.sleep(0.1)
            if time.time() - t0 > 15.0:
                raise RuntimeError(f'Sensor timeout. Is Gazebo running with {WORLD_NAME}?')
        print(f'[StormEnv map1] Connected. Goal=({GOAL_X},{GOAL_Y})')

    # -----------------------------------------------------------------------
    def step(self, action):
        self._send_action(action)
        time.sleep(0.05)

        obs        = self._get_obs()
        lidar      = obs[:N_LIDAR]
        dist_now   = obs[N_LIDAR]
        theta_goal = obs[N_LIDAR + 1]

        if dist_now < GOAL_RADIUS:
            print(f'  [GOAL!] dist={dist_now:.3f}')
            reward     = R_GOAL
            terminated = True
            self._stop()

        elif self._check_collision(lidar):
            print(f'  [COLLISION] min={np.min(lidar):.3f} dist={dist_now:.3f}')
            reward     = R_COLLISION
            terminated = True
            self._stop()

        else:
            delta      = self.dist_prev - dist_now
            delta      = max(-DELTA_CLAMP, min(DELTA_CLAMP, delta))
            r_progress = W_PROGRESS * delta
            r_heading  = W_HEADING  * math.cos(theta_goal)
            reward     = r_progress + r_heading + R_STEP
            terminated = False

        self.dist_prev   = dist_now
        self.step_count += 1
        truncated = (self.step_count >= self.max_steps)
        if truncated:
            self._stop()

        return obs, reward, terminated, truncated, {}

    # -----------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.ep_count += 1

        for _ in range(3):
            self._stop()
            time.sleep(0.1)

        exact = options['spawn'] if options and 'spawn' in options else None
        self._reset_pose(exact_spawn=exact)
        self.step_count = 0
        self._stop()

        # Wait for odometry to stabilize
        time.sleep(0.3)
        prev_x, prev_y = None, None
        t0 = time.time()
        while time.time() - t0 < 3.0:
            odom = self._node.get_odom()
            if odom is not None:
                x = odom.pose.pose.position.x
                y = odom.pose.pose.position.y
                if prev_x is not None:
                    if abs(x - prev_x) < 0.005 and abs(y - prev_y) < 0.005:
                        break
                prev_x, prev_y = x, y
            time.sleep(0.05)

        # Wait for a fresh scan
        seq_before = self._node.get_scan_seq()
        t0 = time.time()
        while self._node.get_scan_seq() < seq_before + 3:
            time.sleep(0.05)
            if time.time() - t0 > 4.0:
                break

        self._stop()
        time.sleep(0.1)

        # Save stabilized odometry reference
        odom = self._node.get_odom()
        self.odom_x0 = odom.pose.pose.position.x if odom else self.spawn_x
        self.odom_y0 = odom.pose.pose.position.y if odom else self.spawn_y

        # Reset incremental clamp state
        self._last_odom_raw_x = self.odom_x0
        self._last_odom_raw_y = self.odom_y0
        self._est_dx = 0.0
        self._est_dy = 0.0
        self._last_x_robot = self.spawn_x
        self._last_y_robot = self.spawn_y
        self._last_yaw     = 0.0

        obs            = self._get_obs()
        self.dist_prev = obs[N_LIDAR]

        if exact is not None:
            print(f'  [RESET benchmark] spawn=({self.spawn_x:.2f},{self.spawn_y:.2f}) '
                  f'dist_goal={obs[N_LIDAR]:.2f}m '
                  f'theta_goal0={math.degrees(obs[N_LIDAR+1]):.1f}deg')
        else:
            print(f'  [RESET ep={self.ep_count}] spawn=({self.spawn_x:.2f},{self.spawn_y:.2f}) '
                  f'dist_goal={obs[N_LIDAR]:.2f}m')

        return obs, {}

    def close(self):
        self._stop()
        self._node.destroy_node()

    # -----------------------------------------------------------------------
    def _get_goal_data(self):
        """Relative odometry with per-step incremental clamping.

        Filters wheel-slip spikes that would otherwise accumulate as
        drift over long episodes. ODOM_CLAMP is a global safety net.
        """
        odom = self._node.get_odom()
        if odom is None:
            return self.dist_prev, 0.0

        ox = odom.pose.pose.position.x
        oy = odom.pose.pose.position.y

        if self._last_odom_raw_x is None:
            self._last_odom_raw_x = ox
            self._last_odom_raw_y = oy

        step_dx  = ox - self._last_odom_raw_x
        step_dy  = oy - self._last_odom_raw_y
        step_mag = math.hypot(step_dx, step_dy)

        if step_mag > STEP_DELTA_CLAMP:
            scale    = STEP_DELTA_CLAMP / step_mag
            step_dx *= scale
            step_dy *= scale

        self._est_dx += step_dx
        self._est_dy += step_dy
        self._last_odom_raw_x = ox
        self._last_odom_raw_y = oy

        if math.hypot(self._est_dx, self._est_dy) > ODOM_CLAMP:
            return self.dist_prev, 0.0

        x_robot = self.spawn_x + self._est_dx
        y_robot = self.spawn_y + self._est_dy

        q = odom.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw  = math.atan2(siny, cosy)

        self._last_x_robot = x_robot
        self._last_y_robot = y_robot
        self._last_yaw     = yaw

        dist          = math.hypot(GOAL_X - x_robot, GOAL_Y - y_robot)
        angle_to_goal = math.atan2(GOAL_Y - y_robot, GOAL_X - x_robot)
        theta_goal    = math.atan2(
            math.sin(angle_to_goal - yaw),
            math.cos(angle_to_goal - yaw))

        return dist, theta_goal

    def _get_obs(self):
        msg = self._node.get_scan()
        if msg is None:
            lidar = np.full(N_LIDAR, RANGE_MAX, dtype=np.float32)
        else:
            raw = np.array(msg.ranges, dtype=np.float32)
            raw[np.isinf(raw) | np.isnan(raw)] = RANGE_MAX
            raw = np.clip(raw, 0.0, RANGE_MAX)
            idx   = np.round(np.linspace(0, len(raw) - 1, N_LIDAR)).astype(int)
            lidar = raw[idx]
        dist, theta = self._get_goal_data()
        return np.concatenate((lidar, [dist, theta]), dtype=np.float32)

    def _check_collision(self, lidar):
        return bool(np.any(lidar < COLLISION_DIST))

    def _send_action(self, action_idx):
        ang_vel = ANGULAR_VELS[int(action_idx)]
        lin_vel = LINEAR_VEL if abs(ang_vel) < 0.7 else (LINEAR_VEL / 2.0)
        self._node.pub_cmd(lin_vel, ang_vel)

    def _stop(self):
        self._node.pub_cmd(0.0, 0.0)

    def _reset_pose(self, exact_spawn=None):
        for _ in range(5):
            self._stop()
            time.sleep(0.05)

        if exact_spawn is not None:
            x, y, yaw = exact_spawn
        else:
            base_x, base_y, base_yaw = random.choice(SPAWN_ALL)
            x   = base_x   + random.uniform(-0.08, 0.08)
            y   = base_y   + random.uniform(-0.08, 0.08)
            yaw = base_yaw + random.uniform(-0.20, 0.20)

        self.spawn_x = x
        self.spawn_y = y
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)

        self._node.set_pose(x, y, SPAWN_Z, 0.0, 0.0, qz, qw)
        time.sleep(0.3)
