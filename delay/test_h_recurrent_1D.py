#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Исследование влияния величины запаздывания h на минимальное время преследования T0
с использованием рекуррентного алгоритма (recurrent_T0_solver_1D.py).

Параметры фиксированы (взяты из примера в recurrent_T0_solver_1D.py):
    a1=0.0015, b1=0.005, a2=0.0008, b2=0.002,
    alpha=0.22, beta=0.08, z01=5.0, z02=100.0
h меняется от 1 до 24 с шагом 0.5.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

# Импортируем функцию из рекуррентного модуля
try:
    from recurrent_T0_solver_1D_v01 import find_T0_recurrent
except ImportError as e:
    print("Ошибка импорта: убедитесь, что файл recurrent_T0_solver_1D.py находится в текущей директории.")
    sys.exit(1)

# ----------------------------------------------------------------------
# Фиксированные параметры (кроме h)
# ----------------------------------------------------------------------
FIXED_PARAMS = {
    'a1': 0.10,
    'b1': 0.40,
    'a2': 0.05,
    'b2': 0.20,
    'alpha': 15.0,
    'beta': 8.0,
    'z01': 5.0,
    'z02': 100.0
}

# Диапазон изменения h
h_min = 1.0
h_max = 6.0
step = 1
h_values = np.arange(h_min, h_max + step/2, step)

# Параметры алгоритма
EPS = 1e-6
NMAX = 1000          # максимальное число шагов при поиске T0

# ----------------------------------------------------------------------
# Основной цикл по h
# ----------------------------------------------------------------------
results = {
    'h': [],
    'T0': [],
    'success': []
}

print("Выполняется расчёт...")
for h in h_values:
    print(f"h = {h:.2f}", end=' ', flush=True)
    params = FIXED_PARAMS.copy()
    params['h'] = h
    try:
        T0 = find_T0_recurrent(params, eps=EPS, nmax=NMAX)
    except Exception as e:
        print(f"ошибка: {e}")
        T0 = None

    results['h'].append(h)
    results['T0'].append(T0)
    results['success'].append(T0 is not None)
    print(f"T0 = {T0 if T0 is not None else 'None'}")

# ----------------------------------------------------------------------
# Построение графика
# ----------------------------------------------------------------------
plt.figure(figsize=(12, 7))

h_arr = np.array(results['h'])
mask_success = np.array(results['success'])

# Точки, где решение найдено
if np.any(mask_success):
    h_success = h_arr[mask_success]
    T_success = np.array([results['T0'][i] for i in range(len(h_arr)) if mask_success[i]])
    plt.plot(h_success, T_success, 'o-', label='Рекуррентный метод', color='red', linewidth=2, markersize=6)

# Точки, где решение не найдено (отметим крестиками)
mask_fail = ~mask_success
if np.any(mask_fail):
    h_fail = h_arr[mask_fail]
    # Чтобы отметить на графике, используем scatter с маркером 'x'
    # Но значения T0 нет, поэтому поставим их на уровне минимального T0, чтобы они были видны
    # Лучше просто закрасить области вертикально
    for hf in h_fail:
        plt.axvspan(hf - step/2, hf + step/2, alpha=0.2, color='gray')
    # Добавим подпись в первой такой области
    plt.text(h_fail[0], plt.ylim()[1]*0.9, 'Неразрешимость\n(условие 1)', ha='center', color='black', fontsize=9)

plt.xlabel('Запаздывание h', fontsize=14)
plt.ylabel('Минимальное время преследования T₀', fontsize=14)
plt.title('Зависимость T₀ от величины запаздывания h (рекуррентный метод)', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.tight_layout()

# Добавим подпись о параметрах
params_text = (f"Параметры: a1={FIXED_PARAMS['a1']}, b1={FIXED_PARAMS['b1']}, "
               f"a2={FIXED_PARAMS['a2']}, b2={FIXED_PARAMS['b2']}, "
               f"α={FIXED_PARAMS['alpha']}, β={FIXED_PARAMS['beta']}, "
               f"z01={FIXED_PARAMS['z01']}, z02={FIXED_PARAMS['z02']}")
plt.figtext(0.5, 0.01, params_text, ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Сохраним график
plt.savefig('T0_vs_h_recurrent.png', dpi=150)
plt.show()

# ----------------------------------------------------------------------
# Вывод таблицы результатов
# ----------------------------------------------------------------------
print("\nТаблица результатов (рекуррентный метод):")
print(" h       T0")
print("--------------")
for i, h in enumerate(results['h']):
    T0 = results['T0'][i]
    status = f"{T0:.6f}" if T0 is not None else "None"
    print(f"{h:5.2f}   {status}")