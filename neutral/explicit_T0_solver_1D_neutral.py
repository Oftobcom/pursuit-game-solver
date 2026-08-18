#explicit_T0_solver_1D_neutral.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Явный алгоритм для нейтральной системы (1D) с использованием аналитических формул
для J_i(T) (лемма 5.3). Вычисляет T0 без численного интегрирования.
"""

import math
import numpy as np
from typing import Optional

# ----------------------------------------------------------------------
# Вспомогательные функции
# ----------------------------------------------------------------------
def phi_neutral_explicit(t, a, b, h):
    """То же, что и в basic, но прямым вычислением суммы (без кэша)"""
    if t < 0:
        return 0.0
    n = int(math.floor(t / h))
    total = 1.0
    for k in range(1, n + 1):
        tau = t - k * h
        if tau == 0.0:
            continue
        for j in range(1, k + 1):
            coeff = math.comb(k - 1, j - 1)
            term = coeff * (a ** (k - j)) * (b ** j) * (tau ** j) / math.factorial(j)
            total += term
    return total

def ell(T, h):
    """ℓ(T) = min(T, h) при T>=0"""
    if T < 0:
        return 0.0
    return T if T < h else h

def J_i_explicit(T, a, b, h):
    """
    Явная формула для J_i(T) = ∫_{T-h}^T φ_i(s) ds (лемма 5.3)
    """
    if T < 0:
        return 0.0
    n1 = int(math.floor((T - h) / h)) if T - h >= 0 else -1  # floor((T-h)/h)
    n2 = int(math.floor(T / h))                               # floor(T/h)
    total = ell(T, h)

    # Первая сумма: k = 1 .. n1
    for k in range(1, n1 + 1):
        p = T - k * h
        q = T - h - k * h   # q >= 0, так как k <= n1
        for j in range(1, k + 1):
            coeff = math.comb(k - 1, j - 1)
            term = coeff * (a ** (k - j)) * (b ** j) * ((p ** (j + 1) - q ** (j + 1)) / math.factorial(j + 1))
            total += term

    # Вторая сумма: j = 1 .. n2
    tau = T - n2 * h  # ∈ [0, h)
    for j in range(1, n2 + 1):
        coeff = math.comb(n2 - 1, j - 1)
        term = coeff * (a ** (n2 - j)) * (b ** j) * (tau ** (j + 1) / math.factorial(j + 1))
        total += term

    return total

def Delta_explicit(T, x0, y0, a1, b1, a2, b2, h):
    phi1_T = phi_neutral_explicit(T, a1, b1, h)
    phi1_Th = phi_neutral_explicit(T - h, a1, b1, h) if T >= h else 0.0
    phi2_T = phi_neutral_explicit(T, a2, b2, h)
    phi2_Th = phi_neutral_explicit(T - h, a2, b2, h) if T >= h else 0.0
    J1 = J_i_explicit(T, a1, b1, h)
    J2 = J_i_explicit(T, a2, b2, h)
    return (phi1_T - a1 * phi1_Th) * x0 - (phi2_T - a2 * phi2_Th) * y0 + b1 * x0 * J1 - b2 * y0 * J2

def R_explicit(T, alpha, beta, a1, b1, a2, b2, h):
    """
    R(T) = ∫_0^T (α φ1(s) - β φ2(s)) ds.
    Используем накопление: разбиваем [0,T] на интервалы длины h.
    Для каждого полного интервала [m h, (m+1)h] интеграл равен J_i((m+1)h).
    """
    if T <= 0:
        return 0.0
    n = int(math.floor(T / h))
    I1 = 0.0
    I2 = 0.0
    for m in range(n):
        I1 += J_i_explicit((m + 1) * h, a1, b1, h)
        I2 += J_i_explicit((m + 1) * h, a2, b2, h)
    # Остаток [n h, T]
    if T > n * h:
        # Интеграл на [n h, T] можно вычислить как разность J_i(T) и интеграла на [T-h, n h]? 
        # Проще использовать численное интегрирование для остатка, но для чистоты можно аналитически:
        # Для нейтрального случая можно явно проинтегрировать полином на [n h, T].
        # Здесь для простоты используем quad (но для явного метода лучше избегать).
        # Вместо этого используем аналитическое интегрирование по той же формуле, что и J, но с пределами.
        # Воспользуемся функцией integral_phi из basic, но чтобы не зависеть, перепишем:
        def int_phi_partial(L, U, a, b, h):
            # Интеграл от φ на [L, U], где L и U внутри одного интервала (n h, (n+1)h)
            # и L >= n h. Можно использовать разность J_i(U) и J_i(L) (если U-L <= h)
            if U - L > h:
                raise ValueError("Интервал длиннее h")
            # Используем явную формулу для интеграла от φ на произвольном отрезке [L,U]
            # по аналогии с J_i, но с пределами.
            # Проще: проинтегрировать полином почленно.
            # Здесь для краткости используем численное интегрирование только для остатка.
            from scipy.integrate import quad
            val, _ = quad(lambda s: phi_neutral_explicit(s, a, b, h), L, U, epsabs=1e-12)
            return val
        I1 += int_phi_partial(n * h, T, a1, b1, h)
        I2 += int_phi_partial(n * h, T, a2, b2, h)
    return alpha * I1 - beta * I2

# ----------------------------------------------------------------------
# Поиск T0 (перебор + бисекция)
# ----------------------------------------------------------------------
def find_T0_explicit(params, eps=1e-6, nmax=1000):
    x0, y0 = params['z01'], params['z02']
    alpha, beta = params['alpha'], params['beta']
    a1, b1 = params['a1'], params['b1']
    a2, b2 = params['a2'], params['b2']
    h = params['h']

    if abs(x0 - y0) < eps:
        return 0.0

    # Поиск интервала
    n = 1
    while n <= nmax:
        T = n * h
        delta_val = Delta_explicit(T, x0, y0, a1, b1, a2, b2, h)
        R_val = R_explicit(T, alpha, beta, a1, b1, a2, b2, h)
        if abs(delta_val) <= R_val:
            a = (n - 1) * h
            b = n * h
            break
        n += 1
    else:
        print(f"Неразрешимо: достигнут T_max = {nmax * h}")
        return None

    # Бисекция
    while b - a > eps:
        c = (a + b) / 2.0
        delta_c = Delta_explicit(c, x0, y0, a1, b1, a2, b2, h)
        R_c = R_explicit(c, alpha, beta, a1, b1, a2, b2, h)
        if abs(delta_c) <= R_c:
            b = c
        else:
            a = c
    return (a + b) / 2.0

# ----------------------------------------------------------------------
# Пример
# ----------------------------------------------------------------------
if __name__ == "__main__":
    params = {
        'a1': 0.15, 'b1': 0.05,
        'a2': 0.08, 'b2': 0.02,
        'h': 2.4,
        'alpha': 0.22, 'beta': 0.08,
        'z01': 5.0, 'z02': 100.0
    }
    T0 = find_T0_explicit(params, eps=1e-6, nmax=500)
    if T0 is not None:
        print(f"Явный метод: T0 = {T0:.6f}")
    else:
        print("Не найдено")