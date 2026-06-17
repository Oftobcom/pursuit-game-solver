#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
unittest.py – сравнение рекуррентного и явного методов вычисления T0
для задачи преследования с запаздыванием.
"""

import sys
import math

# Импортируем функции из обоих модулей
# Предполагается, что файлы находятся в той же директории
try:
    from recurrent_T0_solver_1D_v01 import find_T0_recurrent, compute_Delta_recurrent, compute_R_recurrent
    from explicit_T0_solver_1D_v01 import find_T0, compute_Delta, compute_R
except ImportError as e:
    print("Ошибка импорта: убедитесь, что файлы recurrent_T0_solver.py и explicit_T0_solver.py находятся в текущей директории.")
    sys.exit(1)

# Параметры из условия
# PARAMS = {
#     'a1': 0.1,
#     'b1': 0.6,
#     'a2': 0.05,
#     'b2': 0.4,
#     'h': 0.8,
#     'alpha': 14,
#     'beta': 7,
#     'z01': 0.0,
#     'z02': 100.0
# }

# PARAMS = {
#     'a1': 0.0015,
#     'b1': 0.005,
#     'a2': 0.0008,
#     'b2': 0.002,
#     'h': 24.0,
#     'alpha': 0.22,
#     'beta': 0.08,
#     'z01': 5.0,
#     'z02': 100.0
# }

PARAMS = {
    'a1': 0.10,
    'b1': 0.40,
    'a2': 0.05,
    'b2': 0.20,
    'h': 5,
    'alpha': 15.0,
    'beta': 8.0,
    'z01': 5.0,
    'z02': 100.0
}

EPS = 1e-6          # точность для бисекции
NMAX = 500           # максимальное число шагов (чтобы избежать переполнения при k>170)

def compare_T0():
    """Сравнение результатов find_T0_recurrent и find_T0."""
    print("=== Сравнение T0 ===")
    T0_rec = find_T0_recurrent(PARAMS, eps=EPS, nmax=NMAX)
    T0_exp = find_T0(PARAMS, eps=EPS, Nmax=NMAX)

    if T0_rec is None:
        print("Рекуррентный метод: решение не найдено (None)")
    else:
        print(f"Рекуррентный T0 = {T0_rec:.10f}")

    if T0_exp is None:
        print("Явный метод: решение не найдено (None)")
    else:
        print(f"Явный T0        = {T0_exp:.10f}")

    if T0_rec is not None and T0_exp is not None:
        diff = abs(T0_rec - T0_exp)
        rel_diff = diff / max(abs(T0_rec), abs(T0_exp), 1.0)
        print(f"Абсолютная разница: {diff:.2e}")
        print(f"Относительная разница: {rel_diff:.2e}")
        if diff < 1e-6:
            print("✓ Результаты совпадают в пределах 1e-6.")
        else:
            print("✗ Результаты различаются более чем на 1e-6.")
    elif T0_rec is None and T0_exp is None:
        print("Оба метода не нашли решения – это может быть нормально.")
    else:
        print("Один из методов не нашёл решения – проверьте параметры.")

def compare_Delta_R(T_values):
    """Сравнение Δ(T) и R(T) в заданных точках T."""
    print("\n=== Сравнение Δ(T) и R(T) ===")
    for T in T_values:
        try:
            Delta_rec = compute_Delta_recurrent(T, PARAMS)
            R_rec = compute_R_recurrent(T, PARAMS)
        except Exception as e:
            Delta_rec = R_rec = None
            print(f"T={T}: рекуррентная ошибка – {e}")

        try:
            Delta_exp = compute_Delta(T, PARAMS)
            R_exp = compute_R(T, PARAMS)
        except Exception as e:
            Delta_exp = R_exp = None
            print(f"T={T}: явная ошибка – {e}")

        if Delta_rec is not None and Delta_exp is not None:
            delta_diff = abs(Delta_rec - Delta_exp)
            r_diff = abs(R_rec - R_exp)
            print(f"T={T:8.2f} | Δ_diff = {delta_diff:.3e} | R_diff = {r_diff:.3e}")
        else:
            print(f"T={T}: одно из значений не вычислено")

def main():
    print("Сравнение рекуррентного и явного методов для параметров:")
    for k, v in PARAMS.items():
        print(f"  {k} = {v}")
    print(f"Точность EPS = {EPS}, NMAX = {NMAX}\n")

    compare_T0()

    # Проверяем на нескольких значениях T (включая возможный интервал T0)
    # Используем кратные h, чтобы не выходить за пределы устойчивости
    test_T = [PARAMS['h'] * i for i in range(1, min(NMAX, 10) + 1)]
    compare_Delta_R(test_T)

if __name__ == "__main__":
    main()
