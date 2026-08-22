#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
unittest_3D_neutral.py – сравнение рекуррентного и явного методов вычисления T0
для задачи преследования нейтрального типа в трёхмерном пространстве E = R^3.
Основан на unittest_3D_v01.py, адаптирован для нейтральной системы (one_7.tex).
"""

import sys
import math

# Импортируем функции из 3D-модулей для нейтрального случая
try:
    from recurrent_T0_solver_3D_neutral import (
        find_T0_recurrent,
        Delta_norm_recurrent,
        R_recurrent
    )
    from explicit_T0_solver_3D_neutral import (
        find_T0_explicit,
        Delta_norm_explicit,
        R_explicit
    )
except ImportError as e:
    print("Ошибка импорта: убедитесь, что файлы recurrent_T0_solver_3D_neutral.py и explicit_T0_solver_3D_neutral.py находятся в текущей директории.")
    sys.exit(1)

# Параметры трёхмерного нейтрального эксперимента (из раздела численных примеров one_7)
parameters_3d = {
    'a1': 0.15,
    'b1': 0.25,
    'a2': 0.08,
    'b2': 0.12,
    'h': 0.8,
    'alpha': 2.0,
    'beta': 0.8,
    'z01': [0.0, 0.0, 0.0],
    'z02': [20.0, 15.0, 10.0]
}

# Альтернативные параметры (можно раскомментировать для проверки другого примера)
# parameters_3d = {
#     'a1': 0.15, 'b1': 0.25,
#     'a2': 0.08, 'b2': 0.12,
#     'h': 0.8,
#     'alpha': 2.0, 'beta': 0.8,
#     'z01': [0.0, 0.0, 0.0],
#     'z02': [20.0, 15.0, 10.0]
# }

EPS = 1e-6          # точность для бисекции
NMAX = 500          # максимальное число шагов (T_max = NMAX * h)

def compare_T0():
    """Сравнение результатов find_T0_recurrent и find_T0_explicit."""
    print("=== Сравнение T0 (нейтральный случай, 3D) ===")
    T0_rec = find_T0_recurrent(parameters_3d, eps=EPS, nmax=NMAX)
    T0_exp = find_T0_explicit(parameters_3d, eps=EPS, nmax=NMAX)

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
    print("\n=== Сравнение ||Δ(T)|| и R(T) (нейтральный случай) ===")
    print(f"{'T (с)':>8} | {'||Δ|| diff':>12} | {'R diff':>12}")
    print("-" * 40)
    for T in T_values:
        try:
            Delta_norm_rec = Delta_norm_recurrent(T, parameters_3d['z01'], parameters_3d['z02'],
                                                  parameters_3d['a1'], parameters_3d['b1'],
                                                  parameters_3d['a2'], parameters_3d['b2'],
                                                  parameters_3d['h'])
            R_rec = R_recurrent(T, parameters_3d['alpha'], parameters_3d['beta'],
                                parameters_3d['a1'], parameters_3d['b1'],
                                parameters_3d['a2'], parameters_3d['b2'],
                                parameters_3d['h'])
        except Exception as e:
            Delta_norm_rec = R_rec = None
            print(f"T={T:6.2f}: рекуррентная ошибка – {e}")
            continue

        try:
            Delta_norm_exp = Delta_norm_explicit(T, parameters_3d['z01'], parameters_3d['z02'],
                                                 parameters_3d['a1'], parameters_3d['b1'],
                                                 parameters_3d['a2'], parameters_3d['b2'],
                                                 parameters_3d['h'])
            R_exp = R_explicit(T, parameters_3d['alpha'], parameters_3d['beta'],
                               parameters_3d['a1'], parameters_3d['b1'],
                               parameters_3d['a2'], parameters_3d['b2'],
                               parameters_3d['h'])
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
    print("Сравнение рекуррентного и явного методов для трёхмерного нейтрального случая")
    print("Параметры эксперимента:")
    print(f"  h = {parameters_3d['h']} с")
    print(f"  a1 = {parameters_3d['a1']}, b1 = {parameters_3d['b1']}")
    print(f"  a2 = {parameters_3d['a2']}, b2 = {parameters_3d['b2']}")
    print(f"  α = {parameters_3d['alpha']}, β = {parameters_3d['beta']}")
    print(f"  z01 = {parameters_3d['z01']}")
    print(f"  z02 = {parameters_3d['z02']}")
    print(f"Точность EPS = {EPS}, NMAX = {NMAX} (T_max = {NMAX * parameters_3d['h']} с)\n")

    compare_T0()

    # Тестовые точки T – кратные h (первые 10 шагов)
    test_T = [parameters_3d['h'] * i for i in range(1, min(NMAX, 10) + 1)]
    compare_Delta_R(test_T)

if __name__ == "__main__":
    main()