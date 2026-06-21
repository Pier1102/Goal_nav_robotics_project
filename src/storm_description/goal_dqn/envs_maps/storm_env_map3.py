import time
import math
import random
import threading
import numpy as np
import gymnasium as gym
import subprocess
from gymnasium import spaces

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ros_gz_interfaces.srv import SetEntityPose

N_LIDAR        = 50
RANGE_MAX      = 5.0
N_ACTIONS      = 11
LINEAR_VEL     = 0.2
ANGULAR_VELS   = [-1.5 + 0.3 * i for i in range(11)]
COLLISION_DIST = 0.20
SPAWN_Z        = 0.15
WORLD_NAME     = 'test_map_3'
GOAL_X      =  4.98
GOAL_Y      = -0.45
GOAL_RADIUS =  1.0

# ---------------------------------------------------------------------------
# Spawn points
# ---------------------------------------------------------------------------
SPAWN_NEAR = [
    ( 4.79,  2.96, -1.5708),
    ( 0.53, -2.81,  0.0000),
]

SPAWN_ALL = [
    ( 4.79,  2.96, -1.5708),
    ( 0.53, -2.81,  0.0000),
    (-4.00,  3.2,  0.0000),
    (-3.46, -3.14,  0.0000),
]

EP_BIAS = 300   # episodes <= EP_BIAS spawn only from SPAWN_NEAR

# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------
R_GOAL      = +200.0
R_COLLISION = -200.0
R_STEP      =   -0.2
W_PROGRESS  =   20.0
W_HEADING   =    1.0

# ---------------------------------------------------------------------------
# ROS 2 node
# ---------------------------------------------------------------------------
class _RosNode(Node):
    def __init__(self):
        super().__init__('storm_dqn')
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
        if msg.header.frame_id != 'odom':
            return
        with self._odom_lock:
            self._odom = msg

    def invalidate_odom(self):
        with self._odom_lock:
            self._odom = None

    def get_scan(self):
        with self._scan_lock: return self._scan

    def get_scan_seq(self):
        with self._scan_lock: return self._scan_seq

    def get_odom(self):
        with self._odom_lock: return self._odom

    def pub_cmd(self, lin, ang):
        msg = Twist()
        msg.linear.x  = float(lin)
        msg.angular.z = float(ang)
        self.pub_vel.publish(msg)

    def _pause(self, paused: bool):
        val = 'true' if paused else 'false'
        cmd = ['ign', 'service', '-s', f'/world/{WORLD_NAME}/control',
               '--reqtype', 'ignition.msgs.WorldControl', '--reptype', 'ignition.msgs.Boolean',
               '--timeout', '2000', '--req', f'pause: {val}']
        for _ in range(3):
            try:
                if subprocess.run(cmd, timeout=3.0, capture_output=True).returncode == 0: return
            except: pass
            time.sleep(0.5)

    def set_pose(self, x, y, z, qx, qy, qz, qw):
        req = f'name: "storm" position: {{x: {x} y: {y} z: {z}}} orientation: {{x: {qx} y: {qy} z: {qz} w: {qw}}}'
        cmd = ['ign', 'service', '-s', f'/world/{WORLD_NAME}/set_pose',
               '--reqtype', 'ignition.msgs.Pose', '--reptype', 'ignition.msgs.Boolean',
               '--timeout', '2000', '--req', req]
        try:
            self._pause(True)
            time.sleep(0.3)
            subprocess.run(cmd, timeout=5.0, capture_output=True)
            time.sleep(0.3)
            self._pause(False)
            time.sleep(0.5)
        except Exception as e:
            print(f'  [set_pose] error: {e}')
            self._pause(False)


# ---------------------------------------------------------------------------
# StormEnv — Map 3
# ---------------------------------------------------------------------------
class StormEnv(gym.Env):
    def __init__(self, max_steps=1500):
        super().__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(N_LIDAR + 2,), dtype=np.float32)
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.max_steps  = max_steps
        self.step_count = 0
        self.dist_prev  = 0.0
        self.spawn_x    = 0.0
        self.spawn_y    = 0.0
        self.ep_count   = 0

        if not rclpy.ok(): rclpy.init()
        self._node = _RosNode()
        threading.Thread(target=rclpy.spin, args=(self._node,), daemon=True).start()

        print('[StormEnv map3] Waiting for sensors...')
        t0 = time.time()
        while self._node.get_scan() is None or self._node.get_odom() is None:
            time.sleep(0.1)
            if time.time() - t0 > 15.0: raise RuntimeError('Sensor timeout.')
        print(f'[StormEnv map3] Connected. Goal=({GOAL_X},{GOAL_Y})')

    def step(self, action):
        self._send_action(action)
        time.sleep(0.05)

        obs        = self._get_obs()
        lidar      = obs[:N_LIDAR]
        dist_now   = obs[N_LIDAR]
        theta_goal = obs[N_LIDAR + 1]

        if dist_now < GOAL_RADIUS:
            print(f'  [GOAL!] dist={dist_now:.3f}')
            reward, terminated = R_GOAL, True
            self._stop()
        elif self._check_collision(lidar):
            print(f'  [COLLISION] min={np.min(lidar):.3f} dist={dist_now:.3f}')
            reward, terminated = R_COLLISION, True
            self._stop()
        else:
            reward = W_PROGRESS * (self.dist_prev - dist_now) + W_HEADING * math.cos(theta_goal) + R_STEP
            terminated = False

        self.dist_prev   = dist_now
        self.step_count += 1
        truncated = (self.step_count >= self.max_steps)
        if truncated: self._stop()

        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.ep_count += 1
        for _ in range(3):
            self._stop()
            time.sleep(0.1)

        self._reset_pose(options)
        self.step_count = 0
        self._stop()

        # Odometry is invalidated and re-acquired close to the new spawn,
        # to avoid using a stale reading from before the teleport.
        self._node.invalidate_odom()
        time.sleep(0.8)

        t0 = time.time()
        while time.time() - t0 < 10.0:
            odom = self._node.get_odom()
            if odom is not None and math.hypot(odom.pose.pose.position.x - self.spawn_x, odom.pose.pose.position.y - self.spawn_y) < 0.5:
                break
            time.sleep(0.1)

        seq_before = self._node.get_scan_seq()
        t0 = time.time()
        while self._node.get_scan_seq() < seq_before + 3:
            time.sleep(0.05)
            if time.time() - t0 > 3.0: break

        self._stop()
        time.sleep(0.1)
        obs = self._get_obs()
        self.dist_prev = obs[N_LIDAR]
        return obs, {}

    def close(self):
        self._stop()
        self._node.destroy_node()

    def _get_goal_data(self):
        odom = self._node.get_odom()
        if odom is None: return self.dist_prev, 0.0
        x_robot, y_robot = odom.pose.pose.position.x, odom.pose.pose.position.y
        q = odom.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        dist = math.hypot(GOAL_X - x_robot, GOAL_Y - y_robot)
        angle_to_goal = math.atan2(GOAL_Y - y_robot, GOAL_X - x_robot)
        theta_goal = math.atan2(math.sin(angle_to_goal - yaw), math.cos(angle_to_goal - yaw))
        return dist, theta_goal

    def _get_obs(self):
        msg = self._node.get_scan()
        if msg is None: lidar = np.full(N_LIDAR, RANGE_MAX, dtype=np.float32)
        else:
            raw = np.clip(np.nan_to_num(np.array(msg.ranges, dtype=np.float32), posinf=RANGE_MAX), 0.0, RANGE_MAX)
            lidar = raw[np.round(np.linspace(0, len(raw) - 1, N_LIDAR)).astype(int)]
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

    def _reset_pose(self, options=None):
        # 'spawn' in options forces an exact spawn point (used by benchmarks)
        if options is not None and 'spawn' in options:
            base_x, base_y, base_yaw = options['spawn']
        else:
            pool = SPAWN_NEAR if self.ep_count <= EP_BIAS else SPAWN_ALL
            base_x, base_y, base_yaw = random.choice(pool)

        x   = base_x   + random.uniform(-0.15, 0.15)
        y   = base_y   + random.uniform(-0.15, 0.15)
        yaw = base_yaw + random.uniform(-0.25, 0.25)

        self.spawn_x, self.spawn_y = x, y
        qz, qw = math.sin(yaw / 2.0), math.cos(yaw / 2.0)
        self._node.set_pose(x, y, SPAWN_Z, 0.0, 0.0, qz, qw)
        time.sleep(0.3)
