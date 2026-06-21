"""
genera_semi.py — genera semi_corridoio.npy
Identico alla logica del collega + goal aggiunto come ancora.
Esegui UNA VOLTA prima del training di Fase 3:
    python3 genera_semi.py
"""
import math
import numpy as np

ANCORE = [
    (-4.38,  3.00),
    ( 2.10, -0.05),
    (-0.25,  3.31),
    (-0.82, -3.86),
    (-2.69,  4.14),
    (-1.90, -0.15),
    ( 2.95,  4.46),
    ( 5.68,  4.17),
    ( 5.02,  1.64),
    ( 0.95, -3.02),
    ( 3.56,  0.66),
    (-2.54, -2.55),
    ( 1.02, -1.36),
    (-2.84,  1.34),
    # Ancore aggiuntive vicino al goal (5.00, 4.50)
    # per garantire spawn densi nella zona finale
    ( 4.00,  4.50),
    ( 4.50,  4.00),
    ( 4.50,  5.00),
    ( 3.50,  4.50),
]

_OFF  = (0.654951, 0.036824)
_CYL  = [
    (-4.05 + _OFF[0], -0.02 + _OFF[1], 1.0),
    (-1.50 + _OFF[0], -2.22 + _OFF[1], 1.1),
]

_WALLS_LOCAL = [
    (-3.65705,-1.32130,-2.09440,1.25,0.15),(-3.97128,-2.45953,-1.62999,1.47617,0.15),
    (-4.86060,1.29248,2.87979,2.0,0.15),(-5.74101,3.06881,1.55662,3.22416,0.15),
    (-3.92788,4.60574,0.0,3.73266,0.15),(5.75719,3.04244,-1.5708,4.25,0.15),
    (4.61512,1.00004,3.13502,2.46071,0.15),(0.45284,-3.85548,1.02847,1.79371,0.15),
    (0.93572,-0.62107,0.0,0.15,0.15),(0.90637,-1.91274,-1.59399,2.68116,0.15),
    (2.22362,-0.81026,2.99574,2.75343,0.15),(3.49437,0.00409,1.59656,2.15775,0.15),
    (-2.26066,-1.31701,1.0472,1.25,0.15),(-1.97687,-0.14373,1.55819,1.54405,0.15),
    (-2.46311,1.2405,-0.96103,1.84812,0.15),(-3.67002,2.1299,-0.26212,1.64234,0.15),
    (-4.3907,2.87325,1.5708,1.25,0.15),(-3.35389,3.39689,-0.02543,2.2243,0.15),
    (-1.16927,2.71173,-0.53352,2.77532,0.15),(-0.28634,-0.83901,-1.5708,3.75,0.15),
    (-3.66429,-3.75833,-1.06706,1.58449,0.15),(0.91333,0.74127,2.96045,2.58925,0.15),
    (1.06335,2.95272,-2.44916,2.9688,0.15),(2.113,1.44655,1.5708,2.0,0.15),
    (3.23087,2.36716,-0.00393,2.38576,0.15),(3.26603,3.84811,-0.00393,2.38576,0.15),
    (4.3839,3.10324,-1.59453,1.63137,0.15),(-2.34465,-4.76466,-0.36755,2.25485,0.15),
    (0.78988,4.30405,-2.35619,2.5,0.15),(-1.08876,4.03947,2.64612,2.53204,0.15),
    (3.68896,5.11367,3.13133,4.28668,0.15),(-0.66693,-4.85547,0.39623,1.63911,0.15),
]
WALLS = [(cx+_OFF[0], cy+_OFF[1], yaw, L, T) for (cx,cy,yaw,L,T) in _WALLS_LOCAL]

CLEAR_WALL = 0.30
CLEAR_CYL  = 0.45

def dist_wall(px, py, wx, wy, yaw, L, T):
    dx, dy = px - wx, py - wy
    c, s   = math.cos(-yaw), math.sin(-yaw)
    lx, ly = c*dx - s*dy, s*dx + c*dy
    ex, ey = max(abs(lx) - L/2, 0.0), max(abs(ly) - T/2, 0.0)
    return math.hypot(ex, ey)

def is_free(x, y):
    for (cx, cy, r) in _CYL:
        if math.hypot(x - cx, y - cy) < r + CLEAR_CYL:
            return False
    for w in WALLS:
        if dist_wall(x, y, *w) < CLEAR_WALL:
            return False
    return True

semi = []
for (ax, ay) in ANCORE:
    for r in np.arange(0.0, 0.9, 0.2):
        for th in np.arange(0, 2*math.pi, math.pi/6):
            x = ax + r * math.cos(th)
            y = ay + r * math.sin(th)
            if is_free(x, y):
                semi.append((round(x, 2), round(y, 2)))

semi = list(set(semi))
arr  = np.array(semi)
np.save('semi_corridoio.npy', arr)
print(f'Generati {len(arr)} spawn validi (da {len(ANCORE)} ancore)')

# Distribuzione distanze dal goal
GOAL_X, GOAL_Y = 5.00, 4.50
dists = sorted([math.hypot(GOAL_X-s[0], GOAL_Y-s[1]) for s in semi])
bins  = [0, 2, 4, 6, 8, 10, 15]
print('\nDistribuzione distanze dal goal:')
for i in range(len(bins)-1):
    n = sum(1 for d in dists if bins[i] <= d < bins[i+1])
    print(f'  {bins[i]}-{bins[i+1]}m: {n} ({n/len(arr)*100:.0f}%)')

try:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 9))
    for (wx, wy, yaw, L, T) in WALLS:
        c, s = math.cos(yaw), math.sin(yaw)
        ax.plot([wx - c*L/2, wx + c*L/2], [wy - s*L/2, wy + s*L/2], 'b-', lw=2)
    for (cx, cy, r) in _CYL:
        th = np.linspace(0, 2*math.pi, 40)
        ax.plot(cx + r*np.cos(th), cy + r*np.sin(th), 'gray', lw=2)
    ax.scatter(arr[:, 0], arr[:, 1], s=12, c='red',   label='spawn')
    ax.scatter([a[0] for a in ANCORE], [a[1] for a in ANCORE],
               s=80, c='green', marker='*', label='ancore')
    ax.scatter([GOAL_X], [GOAL_Y], s=200, c='lime',
               marker='o', zorder=5, label='GOAL')
    ax.set_aspect('equal')
    ax.legend()
    ax.set_title(f'{len(arr)} spawn validi')
    plt.savefig('semi_plot.png', dpi=100)
    print('\nPlot salvato: semi_plot.png')
except ImportError:
    print("\nLibreria matplotlib non trovata, skippo il plot visivo.")