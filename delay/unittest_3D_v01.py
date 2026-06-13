#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
unittest_3D.py – сравнение рекуррентного и явного методов вычисления T0
для задачи преследования с запаздыванием в трёхмерном пространстве E = R^3.
"""

import sys
import math

# Импортируем функции из обоих 3D-модулей
try:
    from recurrent_T0_solver_3D import find_T0_recurrent_3d, compute_Delta_norm as rec_Delta_norm, compute_R_recurrent as rec_R
    from explicit_T0_solver_3D import find_T0 as find_T0_explicit, compute_Delta_norm as exp_Delta_norm, compute_R as exp_R
except ImportError as e:
    print("Ошибка импорта: убедитесь, что файлы recurrent_T0_solver_3D.py и explicit_T0_solver_3D.py находятся в текущей директории.")
    sys.exit(1)

# Параметры трёхмерного численного эксперимента (из таблицы)
PARAMS_3D = {
    'a1': 0.10,
    'b1': 0.40,
    'a2': 0.05,
    'b2': 0.20,
    'h': 0.3,
    'alpha': 15.0,
    'beta': 8.0,
    'z01': [0.0, 0.0, 0.0],
    'z02': [100.0, 80.0, 60.0]
}

# PARAMS_3D = {
#     'a1': 0.0015,
#     'b1': 0.005,
#     'a2': 0.0008,
#     'b2': 0.002,
#     'h': 24.0,
#     'alpha': 0.22,
#     'beta': 0.08,
#     'z01': [5.0, 0.0, 0.0],
#     'z02': [100.0, 0.0, 0.0]
# }

# PARAMS_3D = {
#     'a1': 0.10,
#     'b1': 0.40,
#     'a2': 0.05,
#     'b2': 0.20,
#     'h': 0.3,
#     'alpha': 15.0,
#     'beta': 8.0,
#     'z01': [5.0, 0.0, 0.0],
#     'z02': [100.0, 0.0, 0.0]
# }

EPS = 1e-6          # точность для бисекции
NMAX = 500          # максимальное число шагов (T_max = NMAX * h)

def compare_T0():
    """Сравнение результатов find_T0_recurrent_3d и find_T0_explicit."""
    print("=== Сравнение T0 ===")
    T0_rec = find_T0_recurrent_3d(PARAMS_3D, eps=EPS, nmax=NMAX)
    T0_exp = find_T0_explicit(PARAMS_3D, eps=EPS, Nmax=NMAX)

    if T0_rec is None:
        print("Рекуррентный метод: решение не найдено (None)")
    else:
        print(f"Рекуррентный T0 = {T0_rec:.10f} с")

    if T0_exp is None:
        print("Явный метод: решение не найдено (None)")
    else:
        print(f"Явный T0        = {T0_exp:.10f} с")

    if T0_rec is not None and T0_exp is not None:
        diff = abs(T0_rec - T0_exp)
        rel_diff = diff / max(abs(T0_rec), abs(T0_exp), 1.0)
        print(f"Абсолютная разница: {diff:.2e} с")
        print(f"Относительная разница: {rel_diff:.2e}")
        if diff < 1e-6:
            print("✓ Результаты совпадают в пределах 1e-6.")
        else:
            print("✗ Результаты различаются более чем на 1e-6.")
    elif T0_rec is None and T0_exp is None:
        print("Оба метода не нашли решения – проверьте параметры.")
    else:
        print("Один из методов не нашёл решения – возможно, параметры требуют большего NMAX.")

def compare_Delta_R(T_values):
    """Сравнение ||Δ(T)|| и R(T) в заданных точках T."""
    print("\n=== Сравнение ||Δ(T)|| и R(T) ===")
    print(f"{'T (с)':>8} | {'||Δ|| diff':>12} | {'R diff':>12}")
    print("-" * 40)
    for T in T_values:
        try:
            Delta_norm_rec = rec_Delta_norm(T, PARAMS_3D)
            R_rec = rec_R(T, PARAMS_3D)
        except Exception as e:
            Delta_norm_rec = R_rec = None
            print(f"T={T:6.2f}: рекуррентная ошибка – {e}")
            continue

        try:
            Delta_norm_exp = exp_Delta_norm(T, PARAMS_3D)
            R_exp = exp_R(T, PARAMS_3D)
        except Exception as e:
            Delta_norm_exp = R_exp = None
            print(f"T={T:6.2f}: явная ошибка – {e}")
            continue

        if Delta_norm_rec is not None and Delta_norm_exp is not None:
            delta_diff = abs(Delta_norm_rec - Delta_norm_exp)
            r_diff = abs(R_rec - R_exp)
            print(f"{T:8.2f} | {delta_diff:12.3e} | {r_diff:12.3e}")
        else:
            print(f"{T:8.2f} | {'не вычислено':12} | {'не вычислено':12}")

def main():
    print("Сравнение рекуррентного и явного методов для трёхмерного случая")
    print("Параметры эксперимента:")
    print(f"  h = {PARAMS_3D['h']} с")
    print(f"  a1 = {PARAMS_3D['a1']}, b1 = {PARAMS_3D['b1']}")
    print(f"  a2 = {PARAMS_3D['a2']}, b2 = {PARAMS_3D['b2']}")
    print(f"  α = {PARAMS_3D['alpha']} м/с, β = {PARAMS_3D['beta']} м/с")
    print(f"  z01 = {PARAMS_3D['z01']} м")
    print(f"  z02 = {PARAMS_3D['z02']} м")
    print(f"Точность EPS = {EPS}, NMAX = {NMAX} (T_max = {NMAX * PARAMS_3D['h']} с)\n")

    compare_T0()

    # Тестовые точки T – кратные h (первые 10 шагов)
    test_T = [PARAMS_3D['h'] * i for i in range(1, min(NMAX, 10) + 1)]
    compare_Delta_R(test_T)

if __name__ == "__main__":
    main()