"""
test_dqn_map3_zoned.py — Test diviso per Zone
"""
import os
import numpy as np
import tensorflow as tf
from envs_maps.storm_env_map3 import StormEnv, SPAWN_ALL, GOAL_X, GOAL_Y
import time
import math

# ---------------------------------------------------------------------------
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_PATH = os.path.join(base_dir, 'storm_agents', 'map2_FINAL.weights.h5')

# DIVIDIAMO GLI SPAWN IN DUE ZONE LOGICHE
ZONA_1_FACILE = [
    ( 4.79,  2.96, -1.5708), # dist ~3.4m
    ( 0.53, -2.81,  0.0000), # dist ~5.0m
]

ZONA_2_DIFFICILE = [
    # ( 0.0,  0.0,  -1.50), # dist ~5
    (-4.00,  3.2,  0.0000), # dist ~9.3m
    (-3.46, -3.14,  0.0000), # dist ~8.9m
]

REPS_PER_SPAWN = 2  
MAX_STEPS      = 1500
N_OBS     = 52
N_ACTIONS = 11
# ---------------------------------------------------------------------------

def build_qnet():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(N_OBS,)),
        tf.keras.layers.Dense(300, activation='relu'),
        tf.keras.layers.Dense(300, activation='relu'),
        tf.keras.layers.Dense(N_ACTIONS, activation='linear'),
    ])

def run_test_on_zone(env, model, spawn_list, zone_name):
    print('\n' + '=' * 65)
    print(f' INIZIO TEST: {zone_name} (Punti: {len(spawn_list)})')
    print('=' * 65)
    
    goals = 0
    collisions = 0
    timeouts = 0
    rewards = []
    ep_total = len(spawn_list) * REPS_PER_SPAWN
    ep = 0

    for rep in range(REPS_PER_SPAWN):
        for sx, sy, syaw in spawn_list:
            ep += 1
            obs, _ = env.reset(options={'spawn': (sx, sy, syaw)})
            ep_reward = 0.0
            esito = 'TIMEOUT'

            for _ in range(MAX_STEPS):
                q_values = model(obs[np.newaxis], training=False).numpy()[0]
                action = int(np.argmax(q_values))

                obs, reward, terminated, truncated, _ = env.step(action)
                ep_reward += reward

                if terminated:
                    esito = 'GOAL' if reward > 0 else 'COLLISIONE'
                    break
                if truncated:
                    esito = 'TIMEOUT'
                    break

            rewards.append(ep_reward)

            if esito == 'GOAL':            goals += 1
            elif esito == 'COLLISIONE': collisions += 1
            else:                       timeouts += 1

            dist_spawn = math.hypot(sx - 4.98, sy - (-0.45))
            print(f'  Ep {ep:2d}/{ep_total} | dist={dist_spawn:4.1f}m | rew {ep_reward:6.1f} | [{esito}]')

    print('-' * 65)
    print(f' RISULTATI {zone_name}:')
    print(f'  Success Rate : {goals}/{ep_total} ({goals/ep_total*100:.0f}%)')
    print(f'  Collisioni   : {collisions}/{ep_total} ({collisions/ep_total*100:.0f}%)')
    print(f'  Reward Media : {np.mean(rewards):.1f} ± {np.std(rewards):.1f}')
    print('=' * 65)


def main():
    env = StormEnv(max_steps=MAX_STEPS)
    model = build_qnet()
    model(np.zeros((1, N_OBS), dtype=np.float32))
    model.load_weights(WEIGHTS_PATH)
    
    # Eseguiamo i test separatamente
    run_test_on_zone(env, model, ZONA_1_FACILE, "ZONA 1 (Corto Raggio < 5m)")
    run_test_on_zone(env, model, ZONA_2_DIFFICILE, "ZONA 2 (Lungo Raggio > 5m)")

    env.close()

if __name__ == '__main__':
    main()