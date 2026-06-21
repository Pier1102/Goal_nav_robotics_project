import os
import math
import numpy as np
import tensorflow as tf
from envs_maps.storm_env_map1 import StormEnv, SPAWN_ALL, GOAL_X, GOAL_Y
# ---------------------------------------------------------------------------
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_PATH = os.path.join(base_dir, 'storm_agents', 'map2_FINAL.weights.h5')
TESTS_PER_SPAWN = 3
MAX_STEPS       = 1500

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


def main():
    print('Creazione environment map1...')
    env = StormEnv(max_steps=MAX_STEPS)
    print('Environment pronto.\n')

    model = build_qnet()
    model(np.zeros((1, N_OBS), dtype=np.float32))

    try:
        model.load_weights(WEIGHTS_PATH)
        weights_name = os.path.basename(WEIGHTS_PATH)
        print(f'Pesi caricati: {weights_name}')
    except Exception as e:
        print(f'[ERRORE] Pesi non trovati: {e}')
        env.close()
        return

    tot_episodes      = len(SPAWN_ALL) * TESTS_PER_SPAWN
    results_per_spawn = {i: [] for i in range(len(SPAWN_ALL))}
    goals = collisions = timeouts = 0
    rewards    = []
    steps_list = []

    print('\n' + '=' * 65)
    print(f'TEST ZERO-SHOT MAP1 — {weights_name}')
    print(f'{len(SPAWN_ALL)} spawn × {TESTS_PER_SPAWN} reps = {tot_episodes} episodi')
    print('=' * 65)

    ep = 0
    for spawn_idx, spawn_pt in enumerate(SPAWN_ALL):
        sx, sy, syaw = spawn_pt
        dist = math.hypot(sx - GOAL_X, sy - GOAL_Y)
        print(f'\n--- Spawn {spawn_idx+1}: ({sx:.2f},{sy:.2f}) dist={dist:.1f}m ---')

        for t in range(TESTS_PER_SPAWN):
            ep += 1
            obs, _ = env.reset(options={'spawn': spawn_pt})

            ep_reward = 0.0
            ep_steps  = 0
            esito     = 'TIMEOUT'

            for _ in range(MAX_STEPS):
                q_values = model(obs[np.newaxis], training=False).numpy()[0]
                action   = int(np.argmax(q_values))
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_reward += reward
                ep_steps  += 1
                if terminated:
                    esito = 'GOAL' if reward > 0 else 'COLLISIONE'
                    break
                if truncated:
                    break

            rewards.append(ep_reward)
            steps_list.append(ep_steps)
            results_per_spawn[spawn_idx].append(esito)

            if esito == 'GOAL':         goals      += 1
            elif esito == 'COLLISIONE': collisions += 1
            else:                       timeouts   += 1

            print(f'  Ep {ep:2d}/{tot_episodes} '
                  f'(rep {t+1}/{TESTS_PER_SPAWN})  '
                  f'steps {ep_steps:4d}  '
                  f'rew {ep_reward:7.1f}  [{esito}]')

    print('\n' + '=' * 65)
    print(f'RISULTATI ZERO-SHOT MAP1 — {weights_name}')
    print(f'  Goal:       {goals}/{tot_episodes} ({goals/tot_episodes*100:.0f}%)')
    print(f'  Collisioni: {collisions}/{tot_episodes} ({collisions/tot_episodes*100:.0f}%)')
    print(f'  Timeout:    {timeouts}/{tot_episodes} ({timeouts/tot_episodes*100:.0f}%)')
    print(f'  Reward avg: {np.mean(rewards):.1f} ± {np.std(rewards):.1f}')
    print(f'  Steps avg:  {np.mean(steps_list):.0f}')

    print('\nPer spawn:')
    for i, (sx, sy, _) in enumerate(SPAWN_ALL):
        dist = math.hypot(sx - GOAL_X, sy - GOAL_Y)
        res  = results_per_spawn[i]
        g    = res.count('GOAL')
        print(f'  ({sx:6.2f},{sy:6.2f}) dist={dist:.1f}m  '
              f'goal {g}/{len(res)} — {res}')
    print('=' * 65)

    env.close()


if __name__ == '__main__':
    main()