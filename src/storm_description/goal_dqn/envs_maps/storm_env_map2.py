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
# Costanti hardware
# ---------------------------------------------------------------------------
N_LIDAR        = 50
RANGE_MAX      = 5.0
N_ACTIONS      = 11
LINEAR_VEL     = 0.2
ANGULAR_VELS   = [-1.5 + 0.3 * i for i in range(11)]
COLLISION_DIST = 0.20
SPAWN_Z        = 0.05
WORLD_NAME     = 'test_map_2'

# ---------------------------------------------------------------------------
# Goal — coordinate assolute mondo test_map_2
# ---------------------------------------------------------------------------
GOAL_X      = 5.00
GOAL_Y      = 4.50
GOAL_RADIUS = 0.8

# ---------------------------------------------------------------------------
# Spawn — I 9 punti strategici per il training e il benchmark
# ---------------------------------------------------------------------------
SPAWN_ALL = [
    ( 3.50,  4.50,  0.00),   # vicino goal   ~1.5m  0 curve
    ( 5.60,  3.00,  1.50),   # curva leggera ~2.5m  1 curva
    ( 3.40,  0.59,  1.50),   # intermedio
    ( 2.20, -0.09,  0.00),   # 3 curve     ~7.0m
    (-4.38,  3.00,  1.50),
    ( 2.10, -0.05,  0.00),
    (-0.25,  3.31, -0.60),
    (-0.82, -3.86,  0.00),
    (-2.69,  4.14,  0.00),
]

# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------
R_GOAL      = +200.0
R_COLLISION = -200.0
R_STEP      =   -0.1
W_PROGRESS  =   15.0
W_HEADING   =    0.3

ODOM_CLAMP  =   15.0
DELTA_CLAMP =    0.10


# ---------------------------------------------------------------------------
# Nodo ROS2
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
        subprocess.run(cmd, timeout=5.0, capture_output=True, text=True)

    def set_pose(self, x, y, z, qx, qy, qz, qw):
        req = (f'name: "storm" '
               f'position: {{x: {x} y: {y} z: {z}}} '
               f'orientation: {{x: {qx} y: {qy} z: {qz} w: {qw}}}')
        cmd = ['ign', 'service',
               '-s', f'/world/{WORLD_NAME}/set_pose',
               '--reqtype', 'ignition.msgs.Pose',
               '--reptype', 'ignition.msgs.Boolean',
               '--timeout', '2000',
               '--req', req]
        try:
            self._pause(True)
            time.sleep(0.3)
            subprocess.run(cmd, timeout=5.0, capture_output=True, text=True)
            time.sleep(0.3)
            subprocess.run(cmd, timeout=5.0, capture_output=True, text=True)
            time.sleep(0.3)
            self._pause(False)
            time.sleep(0.3)
        except subprocess.TimeoutExpired:
            print('  [set_pose] timeout!')
            self._pause(False)
        except Exception as e:
            print(f'  [set_pose] errore: {e}')
            self._pause(False)


# ---------------------------------------------------------------------------
# StormEnv map2
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

        if not rclpy.ok():
            rclpy.init()
        self._node = _RosNode()
        threading.Thread(target=rclpy.spin, args=(self._node,), daemon=True).start()

        print('[StormEnv map2] In attesa dei sensori...')
        t0 = time.time()
        while self._node.get_scan() is None or self._node.get_odom() is None:
            time.sleep(0.1)
            if time.time() - t0 > 15.0:
                raise RuntimeError('Timeout sensori. Gazebo avviato con test_map_2?')
        print(f'[StormEnv map2] Connesso. Goal=({GOAL_X},{GOAL_Y})')

    # -----------------------------------------------------------------------
    def step(self, action):
        self._send_action(action)
        time.sleep(0.05)

        obs        = self._get_obs()
        lidar      = obs[:N_LIDAR]
        dist_now   = obs[N_LIDAR]
        theta_goal = obs[N_LIDAR + 1]

        if dist_now < GOAL_RADIUS:
            reward     = R_GOAL
            terminated = True
            self._stop()

        elif self._check_collision(lidar):
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

        # Gestione opzioni (fondamentale per il Benchmark Test)
        exact = options['spawn'] if options and 'spawn' in options else None
        self._reset_pose(exact_spawn=exact)

        self.step_count = 0
        self._stop()

        # Aspetta stabilizzazione odom
        time.sleep(0.3)
        prev_x, prev_y = None, None
        t0 = time.time()
        while time.time() - t0 < 3.0:
            odom = self._node.get_odom()
            if odom is not None:
                x = odom.pose.pose.position.x
                y = odom.pose.pose.position.y
                if prev_x is not None:
                    if abs(x - prev_x) < 0.001 and abs(y - prev_y) < 0.001:
                        break
                prev_x, prev_y = x, y
            time.sleep(0.05)

        # Aspetta scan fresco
        seq_before = self._node.get_scan_seq()
        t0 = time.time()
        while self._node.get_scan_seq() < seq_before + 3:
            time.sleep(0.05)
            if time.time() - t0 > 3.0:
                break

        self._stop()
        time.sleep(0.1)

        # Salva riferimento odom
        odom = self._node.get_odom()
        self.odom_x0 = odom.pose.pose.position.x if odom else 0.0
        self.odom_y0 = odom.pose.pose.position.y if odom else 0.0

        obs            = self._get_obs()
        self.dist_prev = obs[N_LIDAR]

        # Log pulito al reset
        if exact is not None:
            print(f'  [RESET Benchmark] spawn=({self.spawn_x:.2f},{self.spawn_y:.2f}) dist_goal={obs[N_LIDAR]:.2f}m')
        else:
            print(f'  [RESET ep={self.ep_count}] spawn=({self.spawn_x:.2f},{self.spawn_y:.2f}) dist_goal={obs[N_LIDAR]:.2f}m')

        return obs, {}

    def close(self):
        self._stop()
        self._node.destroy_node()

    # -----------------------------------------------------------------------
    def _get_goal_data(self):
        odom = self._node.get_odom()
        if odom is None:
            return self.dist_prev, 0.0

        x_robot = odom.pose.pose.position.x
        y_robot = odom.pose.pose.position.y

        q = odom.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw  = math.atan2(siny, cosy)

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
        if exact_spawn is not None:
            # Modalità Benchmark: usa esattamente le coordinate richieste dal test
            x, y, yaw = exact_spawn
        else:
            # Modalità Training: pesca a caso dai punti strategici con rumore
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