"""
train_dqn_map2.py - DDQN on test_map_2 (Curriculum Phase 2)
Warm-start using map3_FINAL.weights.h5

Launch instructions:
    1. ros2 launch storm_description sim_launch_map2.py
    2. source ~/storm_env/bin/activate
       python3 train_dqn_map2.py

TensorBoard:
    tensorboard --logdir ~/storm_dqn/logs
"""

import os
import sys
import signal
import random
import time
import collections
from datetime import datetime
import numpy as np
import tensorflow as tf

from storm_env_map2 import StormEnv, SPAWN_ALL, GOAL_X, GOAL_Y

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOAD_WEIGHTS = os.path.join(base_dir, 'storm_agents', 'map3_FINAL.weights.h5')

MAX_EPISODES = 3000
MAX_STEPS    = 2000

# Reduced Epsilon: map3 policy is already decent, but exploration 
# is still required for the new geometric features of map2.
EPSILON_START = 0.50
EPSILON_MIN   = 0.05
BETA          = 0.997      # Reaches min epsilon in ~1100 episodes

LEARNING_RATE      = 0.00025
BUFFER_SIZE        = 100_000   # Larger buffer for a more complex map
BATCH_SIZE         = 64
GAMMA              = 0.99
TARGET_UPDATE_FREQ = 2000
WINDOW_SIZE        = 50

STOP_RATE = 0.80
STOP_N    = 30

SAVE_DIR  = os.path.expanduser('~/storm_dqn/storm_agents')
LOG_DIR   = os.path.expanduser('~/storm_dqn/logs')
N_OBS     = 52
N_ACTIONS = 11

# ===========================================================================

def build_qnet(name='QNet'):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(N_OBS,)),
        tf.keras.layers.Dense(300, activation='relu'),
        tf.keras.layers.Dense(300, activation='relu'),
        tf.keras.layers.Dense(N_ACTIONS, activation='linear'),
    ], name=name)


class ReplayBuffer:
    def __init__(self, maxlen):
        self.buf = collections.deque(maxlen=maxlen)

    def add(self, obs, action, reward, next_obs, done):
        self.buf.append((obs, action, reward, next_obs, float(done)))

    def sample(self, n):
        batch = random.sample(self.buf, n)
        obs, act, rew, nobs, done = zip(*batch)
        return (np.array(obs,  dtype=np.float32),
                np.array(act,  dtype=np.int32),
                np.array(rew,  dtype=np.float32),
                np.array(nobs, dtype=np.float32),
                np.array(done, dtype=np.float32))

    def __len__(self):
        return len(self.buf)


@tf.function
def train_step(q_net, target_net, optimizer,
               obs_b, act_b, rew_b, nobs_b, done_b):
    best_actions = tf.argmax(q_net(nobs_b, training=False), axis=1)
    nq = tf.reduce_sum(
        target_net(nobs_b, training=False) *
        tf.one_hot(best_actions, N_ACTIONS), axis=1)
    targets = rew_b + GAMMA * nq * (1.0 - done_b)
    
    with tf.GradientTape() as tape:
        q_pred = tf.reduce_sum(
            q_net(obs_b, training=True) *
            tf.one_hot(act_b, N_ACTIONS), axis=1)
        loss = tf.reduce_mean(tf.square(targets - q_pred))
        
    grads = tape.gradient(loss, q_net.trainable_variables)
    optimizer.apply_gradients(zip(grads, q_net.trainable_variables))
    return loss


_q_net_ref = None

def _handle_sigint(sig, frame):
    print('\n[!] Ctrl+C detected - saving current weights...')
    if _q_net_ref is not None:
        path = os.path.join(SAVE_DIR, 'map2_interrupt.weights.h5')
        _q_net_ref.save_weights(path)
        print(f'    Saved: {path}')
    sys.exit(0)

signal.signal(signal.SIGINT, _handle_sigint)


def main():
    global _q_net_ref
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,  exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir   = os.path.join(LOG_DIR, f'{timestamp}_map2_ddqn')
    os.makedirs(log_dir, exist_ok=True)
    writer = tf.summary.create_file_writer(log_dir)

    print('Creating Map 2 environment...')
    env = StormEnv(max_steps=MAX_STEPS)
    print('Environment ready.\n')

    q_net      = build_qnet('online')
    target_net = build_qnet('target')
    dummy      = np.zeros((1, N_OBS), dtype=np.float32)
    q_net(dummy); target_net(dummy)

    # Warm-start
    load_path = os.path.expanduser(LOAD_WEIGHTS) if LOAD_WEIGHTS else ''
    if load_path and os.path.exists(load_path):
        q_net.load_weights(load_path)
        print(f'[WARM-START] Weights loaded: {load_path}')
    elif load_path:
        print(f'[WARNING] {load_path} not found - starting from scratch.')
    else:
        print('[INFO] No LOAD_WEIGHTS specified - starting from scratch.')

    target_net.set_weights(q_net.get_weights())
    _q_net_ref = q_net

    optimizer    = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    buffer       = ReplayBuffer(BUFFER_SIZE)
    epsilon      = EPSILON_START
    total_steps  = 0
    ep_rewards   = []
    goal_history = []
    goals_total  = 0
    collisions   = 0
    timeouts     = 0
    consec_above = 0
    t_start      = time.time()

    print('=' * 65)
    print('MAP2 - DDQN Complex Map (Curriculum Phase 2)')
    print(f'ε={EPSILON_START}→{EPSILON_MIN} (β={BETA}) | steps={MAX_STEPS}')
    print(f'Auto-stop: goal_rate_{WINDOW_SIZE} >= {STOP_RATE:.0%} for {STOP_N} eps')
    print(f'TensorBoard: tensorboard --logdir {LOG_DIR}')
    print('=' * 65)

    for ep in range(1, MAX_EPISODES + 1):

        obs, _    = env.reset()
        ep_reward = 0.0
        ep_steps  = 0
        losses    = []
        esito     = 'TIMEOUT'

        for _ in range(MAX_STEPS):
            if random.random() < epsilon:
                action = random.randint(0, N_ACTIONS - 1)
            else:
                qvals  = q_net(obs[np.newaxis], training=False).numpy()[0]
                action = int(np.argmax(qvals))

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if terminated:
                esito = 'GOAL' if reward > 0 else 'COLLISION'

            buffer.add(obs, action, reward, next_obs, done)
            obs        = next_obs
            ep_reward += reward
            ep_steps  += 1
            total_steps += 1

            if len(buffer) >= BATCH_SIZE:
                obs_b, act_b, rew_b, nobs_b, done_b = buffer.sample(BATCH_SIZE)
                loss = train_step(q_net, target_net, optimizer,
                                  obs_b, act_b, rew_b, nobs_b, done_b)
                losses.append(float(loss))
                if total_steps % TARGET_UPDATE_FREQ == 0:
                    target_net.set_weights(q_net.get_weights())

            if done:
                break

        if epsilon > EPSILON_MIN:
            epsilon = max(EPSILON_MIN, epsilon * BETA)

        goal_hit = (esito == 'GOAL')
        if goal_hit:            goals_total += 1
        elif esito == 'COLLISION': collisions  += 1
        else:                   timeouts    += 1

        goal_history.append(goal_hit)
        ep_rewards.append(ep_reward)

        avg_rew   = np.mean(ep_rewards[-WINDOW_SIZE:])
        goal_rate = np.mean(goal_history[-WINDOW_SIZE:])
        avg_loss  = float(np.mean(losses)) if losses else 0.0
        elapsed   = (time.time() - t_start) / 60.0
        bias_str  = 'BIAS' if ep <= 300 else 'FULL'

        with writer.as_default():
            tf.summary.scalar('reward/episode',   ep_reward,          step=ep)
            tf.summary.scalar('reward/avg',       avg_rew,            step=ep)
            tf.summary.scalar('train/loss',       avg_loss,           step=ep)
            tf.summary.scalar('train/epsilon',    epsilon,            step=ep)
            tf.summary.scalar('goal/rate',        goal_rate,          step=ep)
            tf.summary.scalar('goal/total',       float(goals_total), step=ep)

        print(f'Ep {ep:4d}/{MAX_EPISODES}  '
              f'steps {ep_steps:4d}  '
              f'rew {ep_reward:8.1f}  '
              f'avg {avg_rew:8.1f}  '
              f'rate {goal_rate:.2f}  '
              f'eps {epsilon:.4f}  '
              f'{elapsed:.1f}min  [{esito}] [{bias_str}]')

        if goal_hit:
            path = os.path.join(SAVE_DIR, f'map2_ep{ep}_r{int(ep_reward)}.weights.h5')
            q_net.save_weights(path)

        if ep % 200 == 0:
            path = os.path.join(SAVE_DIR, f'map2_ckpt_ep{ep}.weights.h5')
            q_net.save_weights(path)
            print(f'  [ckpt] {path} | G:{goals_total} C:{collisions} T:{timeouts}')

        if goal_rate >= STOP_RATE and ep >= WINDOW_SIZE:
            consec_above += 1
        else:
            consec_above = 0

        if consec_above >= STOP_N:
            print(f'\n{"="*65}')
            print(f'MAP 2 COMPLETED! goal_rate={goal_rate:.2%}')
            print(f'Goals:{goals_total} | Collisions:{collisions} | Timeouts:{timeouts}')
            break

    final = os.path.join(SAVE_DIR, 'map2_FINAL.weights.h5')
    q_net.save_weights(final)
    print(f'\nFinal Map 2 weights saved: {final}')
    env.close()

if __name__ == '__main__':
    main()
