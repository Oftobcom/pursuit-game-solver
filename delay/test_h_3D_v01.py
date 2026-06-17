#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Исследование влияния величины запаздывания h на минимальное время преследования T0.
Сравниваются три метода: базовый (численное интегрирование), явный (аналитические формулы),
рекуррентный (устойчивые рекуррентные соотношения).

Параметры фиксированы (взяты из basic_T0_solver_1D.py):
    a1=0.0015, b1=0.005, a2=0.0008, b2=0.002,
    alpha=0.22, beta=0.08, z01=5.0, z02=100.0
h меняется от 1 до 24 с шагом 0.5.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

# Импортируем функции из трёх модулей
try:
    from basic_T0_solver_1D_v01 import find_T0 as find_T0_basic #, find_Tcrit
    from explicit_T0_solver_1D_v01 import find_T0 as find_T0_explicit
    from recurrent_T0_solver_1D_v01 import find_T0_recurrent
except ImportError as e:
    print("Ошибка импорта: убедитесь, что все файлы лежат в одной директории.")
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
h_max = 5.0
step = 1
h_values = np.arange(h_min, h_max + step/2, step)

# Параметры алгоритмов
EPS = 1e-6
NMAX = 1000          # максимальное число шагов при поиске T0
T_MAX = 800.0        # для basic (чтобы не зацикливалось)

# ----------------------------------------------------------------------
# Функция для вызова трёх методов при заданном h
# ----------------------------------------------------------------------
def compute_T0_for_h(h):
    """
    Возвращает кортеж (T0_basic, T0_explicit, T0_rec, T_crit)
    """
    # 1. Базовый метод (basic)
    try:
        T0_basic = find_T0_basic(
            x0=FIXED_PARAMS['z01'],
            y0=FIXED_PARAMS['z02'],
            alpha=FIXED_PARAMS['alpha'],
            beta=FIXED_PARAMS['beta'],
            a1=FIXED_PARAMS['a1'],
            b1=FIXED_PARAMS['b1'],
            a2=FIXED_PARAMS['a2'],
            b2=FIXED_PARAMS['b2'],
            h=h,
            T_max=T_MAX,
            eps=EPS,
            verbose=False
        )
    except Exception as e:
        print(f"basic: ошибка при h={h}: {e}")
        T0_basic = None

    # # Поиск T_crit (момента нарушения условия 1) – только для информации
    # try:
    #     T_crit = find_Tcrit(
    #         alpha=FIXED_PARAMS['alpha'],
    #         beta=FIXED_PARAMS['beta'],
    #         a1=FIXED_PARAMS['a1'],
    #         b1=FIXED_PARAMS['b1'],
    #         a2=FIXED_PARAMS['a2'],
    #         b2=FIXED_PARAMS['b2'],
    #         h=h,
    #         T_max=T_MAX
    #     )
    #     if np.isinf(T_crit):
    #         T_crit = None
    # except Exception:
    #     T_crit = None

    # 2. Явный метод (explicit)
    params = FIXED_PARAMS.copy()
    params['h'] = h
    try:
        T0_explicit = find_T0_explicit(params, eps=EPS, Nmax=NMAX)
    except Exception as e:
        print(f"explicit: ошибка при h={h}: {e}")
        T0_explicit = None

    # 3. Рекуррентный метод (recurrent)
    try:
        T0_rec = find_T0_recurrent(params, eps=EPS, nmax=NMAX)
    except Exception as e:
        print(f"recurrent: ошибка при h={h}: {e}")
        T0_rec = None

    return T0_basic, T0_explicit, T0_rec #, T_crit


# ----------------------------------------------------------------------
# Основной цикл по h
# ----------------------------------------------------------------------
results = {
    'h': [],
    'T0_basic': [],
    'T0_explicit': [],
    'T0_rec': [],
    # 'T_crit': [],
    'basic_ok': [],
    'explicit_ok': [],
    'rec_ok': []
}

print("Выполняется расчёт...")
for h in h_values:
    print(f"h = {h:.2f}", end=' ', flush=True)
    # Tb, Te, Tr, Tc = compute_T0_for_h(h)
    Tb, Te, Tr = compute_T0_for_h(h)
    results['h'].append(h)
    results['T0_basic'].append(Tb)
    results['T0_explicit'].append(Te)
    results['T0_rec'].append(Tr)
    # results['T_crit'].append(Tc)
    results['basic_ok'].append(Tb is not None)
    results['explicit_ok'].append(Te is not None)
    results['rec_ok'].append(Tr is not None)
    print(f"basic={Tb if Tb is not None else 'None'}, explicit={Te if Te is not None else 'None'}, rec={Tr if Tr is not None else 'None'}")

# ----------------------------------------------------------------------
# Построение графиков
# ----------------------------------------------------------------------
plt.figure(figsize=(12, 7))

# Преобразуем списки в numpy-массивы для поэлементных операций
h_arr = np.array(results['h'])
mask_basic = np.array(results['basic_ok'])
mask_exp = np.array(results['explicit_ok'])
mask_rec = np.array(results['rec_ok'])

# Кривые для каждого метода (только где есть значения)

# Базовый метод
mask_basic = results['basic_ok']
if np.any(mask_basic):
    h_basic = h_arr[mask_basic]
    T_basic = np.array([results['T0_basic'][i] for i in range(len(h_arr)) if mask_basic[i]])
    plt.plot(h_basic, T_basic, 'o-', label='Basic (quad)', color='blue', linewidth=2, markersize=6)

# Явный метод
mask_exp = results['explicit_ok']
if np.any(mask_exp):
    h_exp = h_arr[mask_exp]
    T_exp = np.array([results['T0_explicit'][i] for i in range(len(h_arr)) if mask_exp[i]])
    plt.plot(h_exp, T_exp, 's-', label='Explicit', color='green', linewidth=2, markersize=6)

# Рекуррентный метод
mask_rec = results['rec_ok']
if np.any(mask_rec):
    h_rec = h_arr[mask_rec]
    T_rec = np.array([results['T0_rec'][i] for i in range(len(h_arr)) if mask_rec[i]])
    plt.plot(h_rec, T_rec, 'd-', label='Recurrent', color='red', linewidth=2, markersize=6)

# Отметим области, где хотя бы один метод не нашёл решение (неразрешимость)
# Для этого найдём h, где любой из методов вернул None
mask_any_fail = ~(mask_basic & mask_exp & mask_rec)
if np.any(mask_any_fail):
    h_fail = h_arr[mask_any_fail]
    # закрасим вертикальные области
    for hf in h_fail:
        plt.axvspan(hf - step/2, hf + step/2, alpha=0.2, color='gray')
    # добавим пометку (на первой точке сбоя)
    plt.text(h_fail[0], plt.ylim()[1]*0.9, 'Неразрешимость\n(условие 1)', ha='center', color='black', fontsize=9)

# Дополнительно покажем T_crit (если есть) – момент нарушения условия 1
# В виде вертикальных линий для тех h, где T_crit конечен
# for i, h in enumerate(results['h']):
#     Tc = results['T_crit'][i]
#     if Tc is not None and Tc < T_MAX:
#         # не будем рисовать слишком много линий, только для справки
#         pass

plt.xlabel('Запаздывание h', fontsize=14)
plt.ylabel('Минимальное время преследования T₀', fontsize=14)
plt.title('Зависимость T₀ от величины запаздывания h', fontsize=16)
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
plt.savefig('T0_vs_h_comparison.png', dpi=150)
plt.show()

# ----------------------------------------------------------------------
# Вывод таблицы результатов
# ----------------------------------------------------------------------
print("\nТаблица результатов:")
print(" h     basic     explicit   recurrent")
print("--------------------------------------")
for i, h in enumerate(results['h']):
    Tb = results['T0_basic'][i]
    Te = results['T0_explicit'][i]
    Tr = results['T0_rec'][i]
    # Форматируем вывод, заменяя None на строку "None"
    print(f"{h:5.2f}  {str(Tb) if Tb is not None else 'None':>8}  {str(Te) if Te is not None else 'None':>10}  {str(Tr) if Tr is not None else 'None':>10}")