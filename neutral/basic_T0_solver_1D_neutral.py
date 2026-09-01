#basic_T0_solver_1D_neutral.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Базовый (эталонный) алгоритм для нейтральной системы (1D).
Вычисляет T0 с помощью численного интегрирования фундаментального решения.
"""

import math
import numpy as np
from scipy.integrate import quad
from functools import lru_cache
import sys
sys.stdout.reconfigure(encoding='utf-8')

# ----------------------------------------------------------------------
# Фундаментальное решение φ(t) для нейтрального уравнения
# (формула (4) из one_7)
# ----------------------------------------------------------------------
def phi_neutral(t, a, b, h):
    """
    φ(t) = 1 + Σ_{k=1}^{⌊t/h⌋} Σ_{j=1}^{k} C(k-1, j-1) a^{k-j} b^j (t - kh)^j / j!
    """
    if t < 0:
        return 0.0
    n = int(math.floor(t / h))
    total = 1.0
    for k in range(1, n + 1):
        tau = t - k * h
        if tau == 0.0:
            continue
        # Внутренняя сумма по j
        for j in range(1, k + 1):
            coeff = math.comb(k - 1, j - 1)
            term = coeff * (a ** (k - j)) * (b ** j) * (tau ** j) / math.factorial(j)
            total += term
    return total

# ----------------------------------------------------------------------
# Интегралы I(T) = ∫_0^T φ(s) ds, J(T) = ∫_{T-h}^T φ(s) ds
# ----------------------------------------------------------------------
def integral_phi(T, a, b, h, eps=1e-12):
    if T <= 0:
        return 0.0
    n = int(math.floor(T / h))
    total = 0.0
    for i in range(n):
        left = i * h
        right = (i + 1) * h
        val, _ = quad(lambda s: phi_neutral(s, a, b, h), left, right, epsabs=eps, epsrel=eps)
        total += val
    if T > n * h:
        val, _ = quad(lambda s: phi_neutral(s, a, b, h), n * h, T, epsabs=eps, epsrel=eps)
        total += val
    return total

def J_integral(T, a, b, h, eps=1e-12):
    lower = max(0.0, T - h)
    upper = T
    if lower >= upper:
        return 0.0
    total = 0.0
    i_start = int(math.floor(lower / h)) + 1
    i_end = int(math.floor(upper / h))
    if lower < i_start * h:
        val, _ = quad(lambda s: phi_neutral(s, a, b, h), lower, i_start * h, epsabs=eps, epsrel=eps)
        total += val
    for i in range(i_start, i_end):
        val, _ = quad(lambda s: phi_neutral(s, a, b, h), i * h, (i + 1) * h, epsabs=eps, epsrel=eps)
        total += val
    if upper > i_end * h:
        val, _ = quad(lambda s: phi_neutral(s, a, b, h), i_end * h, upper, epsabs=eps, epsrel=eps)
        total += val
    return total

# ----------------------------------------------------------------------
# Δ(T) и R(T) для нейтрального случая (формулы (Delta:main), (R:main))
# ----------------------------------------------------------------------
def Delta(T, x0, y0, a1, b1, a2, b2, h):
    phi1_T = phi_neutral(T, a1, b1, h)
    phi1_Th = phi_neutral(T - h, a1, b1, h) if T >= h else 0.0
    phi2_T = phi_neutral(T, a2, b2, h)
    phi2_Th = phi_neutral(T - h, a2, b2, h) if T >= h else 0.0
    J1 = J_integral(T, a1, b1, h)
    J2 = J_integral(T, a2, b2, h)
    term1 = (phi1_T - a1 * phi1_Th) * x0
    term2 = (phi2_T - a2 * phi2_Th) * y0
    term3 = b1 * x0 * J1
    term4 = b2 * y0 * J2
    return term1 - term2 + term3 - term4

def R(T, alpha, beta, a1, b1, a2, b2, h):
    I1 = integral_phi(T, a1, b1, h)
    I2 = integral_phi(T, a2, b2, h)
    return alpha * I1 - beta * I2

# ----------------------------------------------------------------------
# Основной алгоритм поиска T0 (перебор + бисекция)
# ----------------------------------------------------------------------
def find_T0_basic(x0, y0, alpha, beta, a1, b1, a2, b2, h,
            T_max=500.0, eps=1e-6, verbose=True):
    if abs(x0 - y0) < eps:
        if verbose:
            print("Начальные положения совпадают: T0 = 0")
        return 0.0

    T_low = 0.0
    T_high = None
    m = 1
    while True:
        T = m * h
        if T > T_max:
            if verbose:
                print(f"Достигнут T_max = {T_max}, преследование не найдено")
            return None
        delta_val = Delta(T, x0, y0, a1, b1, a2, b2, h)
        R_val = R(T, alpha, beta, a1, b1, a2, b2, h)
        f_val = abs(delta_val) - R_val
        if verbose:
            print(f"T = {T:.2f}, |Δ| = {abs(delta_val):.6e}, R = {R_val:.6e}, f = {f_val:.6e}")
        if f_val <= 0:
            T_high = T
            break
        T_low = T
        m += 1

    if T_high is None:
        return None

    if verbose:
        print(f"Найден интервал: [{T_low:.6f}, {T_high:.6f}]")

    while T_high - T_low > eps:
        T_mid = (T_low + T_high) / 2.0
        delta_mid = Delta(T_mid, x0, y0, a1, b1, a2, b2, h)
        R_mid = R(T_mid, alpha, beta, a1, b1, a2, b2, h)
        f_mid = abs(delta_mid) - R_mid
        if f_mid > 0:
            T_low = T_mid
        else:
            T_high = T_mid

    T0 = (T_low + T_high) / 2.0
    if verbose:
        print(f"T0 = {T0:.8f}")
    return T0

# ----------------------------------------------------------------------
# Пример использования
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Параметры из одномерного примера (one_7)
    params = {
        'a1': 0.15, 'b1': 0.05,
        'a2': 0.08, 'b2': 0.02,
        'h': 2.4,
        'alpha': 0.22, 'beta': 0.08,
        'z01': 5.0, 'z02': 100.0
    }
    T0 = find_T0_basic(
        x0=params['z01'], y0=params['z02'],
        alpha=params['alpha'], beta=params['beta'],
        a1=params['a1'], b1=params['b1'],
        a2=params['a2'], b2=params['b2'],
        h=params['h'],
        T_max=500.0, eps=1e-6, verbose=True
    )
    if T0 is not None:
        print(f"\nРезультат: T0 = {T0:.6f}")
    else:
        print("\nПреследование не гарантировано в пределах T_max")
