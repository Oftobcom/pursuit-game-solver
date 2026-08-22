#test_h_neutral.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Исследование T0(h) для нейтральной системы.
Сравнение базового, явного и рекуррентного методов.
"""

import numpy as np
import matplotlib.pyplot as plt
from basic_T0_solver_1D_neutral import find_T0 as find_basic
from explicit_T0_solver_1D_neutral import find_T0_explicit
from recurrent_T0_solver_1D_neutral import find_T0_recurrent

FIXED_PARAMS = {
    'a1': 0.02, 'b1': 0.005,
    'a2': 0.1, 'b2': 0.002,
    'alpha': 0.22, 'beta': 0.08,
    'z01': 5.0, 'z02': 100.0
}

# FIXED_PARAMS = {
#     'a1': 0.2, 'b1': 2,
#     'a2': 0.5, 'b2': 2,
#     'alpha': 30, 'beta': 20,
#     'z01': 0, 'z02': 10
# }

h_min = 1
h_max = 10
step = 1
h_values = np.arange(h_min, h_max + step/2, step)

EPS = 1e-6
NMAX = 1000
T_MAX = 800.0

results = {'h': [], 'basic': [], 'explicit': [], 'recurrent': []}

for h in h_values:
    print(f"h = {h:.2f}")
    params = FIXED_PARAMS.copy()
    params['h'] = h

    # # Базовый (численное интегрирование)
    # try:
    #     T_basic = find_basic(
    #         x0=params['z01'], y0=params['z02'],
    #         alpha=params['alpha'], beta=params['beta'],
    #         a1=params['a1'], b1=params['b1'],
    #         a2=params['a2'], b2=params['b2'],
    #         h=h, T_max=T_MAX, eps=EPS, verbose=False
    #     )
    # except:
    #     T_basic = None

    # # Явный
    # try:
    #     T_exp = find_T0_explicit(params, eps=EPS, nmax=NMAX)
    # except:
    #     T_exp = None

    # Рекуррентный
    try:
        T_rec = find_T0_recurrent(params, eps=EPS, nmax=NMAX)
    except:
        T_rec = None

    results['h'].append(h)
    # results['basic'].append(T_basic)
    # results['explicit'].append(T_exp)
    results['recurrent'].append(T_rec)

# Построение графиков
plt.figure(figsize=(10,6))
h_arr = np.array(results['h'])

# for method, label, color in [('basic', 'Basic (quad)', 'blue'),
#                              ('explicit', 'Explicit', 'green'),
#                              ('recurrent', 'Recurrent', 'red')]:
for method, label, color in [('recurrent', 'Recurrent', 'red')]:
    vals = results[method]
    mask = [v is not None for v in vals]
    if any(mask):
        plt.plot(h_arr[mask], np.array(vals)[mask], 'o-', label=label, color=color)

plt.xlabel('Запаздывание h')
plt.ylabel('Минимальное время T0')
plt.title('T0(h) для нейтральной системы')
plt.grid(True)
plt.legend()
plt.show()
