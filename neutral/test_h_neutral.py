#test_h_neutral.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Исследование T0(h) для нейтральной системы.
Сравнение базового, явного и рекуррентного методов.
"""

import numpy as np
import matplotlib.pyplot as plt
from basic_T0_solver_1D_neutral import find_T0_basic
from explicit_T0_solver_1D_neutral import find_T0_explicit
from recurrent_T0_solver_1D_neutral import find_T0_recurrent

# FIXED_PARAMS = {
#  'a1':0.005, 'b1':1.7,
#  'a2':0.004, 'b2':1.5,
#  'alpha':20, 'beta':8,
#  'z01':0.0, 'z02':10.0
# }
# h_values = np.linspace(1, 10, 18)

# FIXED_PARAMS = {
#     'a1': 0.8, 'b1': 0.2,
#     'a2': 0.2, 'b2': 0.05,
#     'alpha': 20.0, 'beta': 7,
#     'z01': 0.0, 'z02': 20.0
# }
# h_values = np.linspace(0.2, 4.0, 30)

# FIXED_PARAMS = {
#  'a1':0.97,  'b1':0.02,  # было 0.8/0.2
#  'a2':0.7,  'b2':0.01,  # было 0.2/0.05
#  'alpha':8.0, 'beta':7.5, # diff=0.5 для стабильности
#  'z01':0.0, 'z02':5.0
# }
# h_values = np.linspace(0.08, 20, 80)

# FIXED_PARAMS = {
#  'a1':0.9,  'b1':0.05,  # было 0.8/0.2
#  'a2':0.5,  'b2':0.01,  # было 0.2/0.05
#  'alpha':8.0, 'beta':7.5, # diff=0.5 для стабильности
#  'z01':0.0, 'z02':20.0
# }
# h_values = np.linspace(0.2, 2.5, 40)

# FIXED_PARAMS = {
#  'a1':0.97,  'b1':0.02,  # было 0.8/0.2
#  'a2':0.7,  'b2':0.01,  # было 0.2/0.05
#  'alpha':8.0, 'beta':7.5, # diff=0.5 для стабильности
#  'z01':0.0, 'z02':20.0
# }
# h_values = np.linspace(0.08, 20, 80)

# FIXED_PARAMS = {
#  'a1':0.2,  'b1':0.3,  # было 0.8/0.2
#  'a2':0.1,  'b2':0.2,  # было 0.2/0.05
#  'alpha':1.0, 'beta':0.5, # diff=0.5 для стабильности
#  'z01':0.0, 'z02':5.0
# }
# h_values = np.linspace(0.2, 2, 40)

FIXED_PARAMS = {
 'a1':0.2,  'b1':1.1,  # было 0.8/0.2
 'a2':0.1,  'b2':0.9,  # было 0.2/0.05
 'alpha':1.0, 'beta':0.7, # diff=0.5 для стабильности
 'z01':0.0, 'z02':5.0
}
h_values = np.linspace(0.5, 4, 40)

EPS = 1e-6
NMAX = 1000
T_MAX = 800.0

results = {'h': [], 'basic': [], 'explicit': [], 'recurrent': []}

for h in h_values:
    print(f"h = {h:.2f}")
    params = FIXED_PARAMS.copy()
    params['h'] = h

    # Базовый (численное интегрирование)
    try:
        T_basic = find_T0_basic(
            x0=params['z01'], y0=params['z02'],
            alpha=params['alpha'], beta=params['beta'],
            a1=params['a1'], b1=params['b1'],
            a2=params['a2'], b2=params['b2'],
            h=h, T_max=T_MAX, eps=EPS, verbose=False
        )
    except:
        T_basic = None

    # Явный
    try:
        T_exp = find_T0_explicit(params, eps=EPS, nmax=NMAX)
    except:
        T_exp = None

    # Рекуррентный
    try:
        T_rec = find_T0_recurrent(params, eps=EPS, nmax=NMAX)
    except:
        T_rec = None

    results['h'].append(h)
    results['basic'].append(T_basic)
    results['explicit'].append(T_exp)
    results['recurrent'].append(T_rec)

# Построение графиков
plt.figure(figsize=(10,6))
h_arr = np.array(results['h'])

# for method, label, color in [('basic', 'Basic (quad)', 'blue'),
#                              ('explicit', 'Explicit', 'green'),
#                              ('recurrent', 'Recurrent', 'red')]:
for method, label, color in [('recurrent', 'Recurrent', 'red'), 
                             ('explicit', 'Explicit', 'green'),
                             ('basic', 'Basic (quad)', 'blue')]:
    vals = results[method]
    mask = [v is not None for v in vals]
    if any(mask):
        plt.plot(h_arr[mask], np.array(vals)[mask], 'o-', label=label, color=color)

plt.xlabel('Запаздывание h')
plt.ylabel('Минимальное время T0')
plt.title('T0(h) для нейтральной системы')
plt.grid(True)
plt.legend()

# --- Добавляем параметры внизу ---
params_str = (
    f"a1={FIXED_PARAMS['a1']:.2f}, b1={FIXED_PARAMS['b1']:.2f}, "
    f"a2={FIXED_PARAMS['a2']:.2f}, b2={FIXED_PARAMS['b2']:.2f}\n"
    f"α={FIXED_PARAMS['alpha']:.2f}, β={FIXED_PARAMS['beta']:.2f}, "
    f"z01={FIXED_PARAMS['z01']:.1f}, z02={FIXED_PARAMS['z02']:.1f}"
)
plt.figtext(0.5, 0.01, params_str, ha='center', fontsize=11,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
plt.tight_layout(rect=[0, 0.05, 1, 1])  # освобождаем место для текста внизу

plt.show()
