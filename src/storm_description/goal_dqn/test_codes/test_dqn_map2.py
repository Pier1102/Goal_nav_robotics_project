import os
import numpy as np
import tensorflow as tf
from envs_maps.storm_env_map2 import StormEnv, SPAWN_ALL, GOAL_X, GOAL_Y
import time
import math

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_PATH = os.path.join(base_dir, 'storm_agents', 'map2_FINAL.weights.h5')

BENCHMARK_SPAWNS = [
    ( 3.50,  4.50,  0.00),   
    ( 5.60,  3.00,  1.50),   
    ( 3.40,  0.59,  1.50),   
    ( 2.20, -0.09,  0.00),   
    (-4.38,  3.00,  1.50),
    (-0.25,  3.31, -0.60),
    (-0.82, -3.86,  0.00),
    (-2.69,  4.14,  0.00),
]

MAX_STEPS = 1500
N_OBS     = 52
N_ACTIONS = 11

def build_qnet():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(N_OBS,)),
        tf.keras.layers.Dense(300, activation='relu'),
        tf.keras.layers.Dense(300, activation='relu'),
        tf.keras.layers.Dense(N_ACTIONS, activation='linear'),
    ])

def main():
    env = StormEnv(max_steps=MAX_STEPS)
    model = build_qnet()
    model(np.zeros((1, N_OBS), dtype=np.float32))
    
    try:
        model.load_weights(WEIGHTS_PATH)
        print(f'Pesi caricati: {WEIGHTS_PATH}')
    except Exception as e:
        print(f'[ERRORE] Pesi non trovati: {e}')
        env.close()
        return

    print('\n' + '=' * 65)
    print(f' INIZIO BENCHMARK TEST: {len(BENCHMARK_SPAWNS)} Punti Strategici')
    print('=' * 65)

    goals = collisions = timeouts = 0

    for i, spawn_pt in enumerate(BENCHMARK_SPAWNS):
        obs, _ = env.reset(options={'spawn': spawn_pt})
        ep_reward = 0.0
        esito = 'TIMEOUT'

        for _ in range(MAX_STEPS):
            q_values = model(obs[np.newaxis], training=False).numpy()[0]
            action   = int(np.argmax(q_values))

            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward

            if terminated:
                esito = 'GOAL' if reward > 0 else 'COLLISIONE'
                break
            if truncated:
                break

        if esito == 'GOAL':            goals += 1
        elif esito == 'COLLISIONE': collisions += 1
        else:                       timeouts += 1

        print(f'  Test {i+1:2d}/{len(BENCHMARK_SPAWNS)} | Spawn=({spawn_pt[0]:5.2f}, {spawn_pt[1]:5.2f}) | Rew {ep_reward:6.1f} | [{esito}]')

    print('-' * 65)
    print(f' SUCCESS RATE : {goals}/{len(BENCHMARK_SPAWNS)} ({(goals/len(BENCHMARK_SPAWNS))*100:.0f}%)')
    print(f' COLLISIONI   : {collisions}/{len(BENCHMARK_SPAWNS)}')
    print('=' * 65)
    env.close()

if __name__ == '__main__':
    main()