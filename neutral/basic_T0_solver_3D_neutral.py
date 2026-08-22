#basic_T0_solver_3D_neutral.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Базовый алгоритм для нейтральной системы (3D) с численным интегрированием.
"""

import math
import numpy as np
from scipy.integrate import quad
from typing import List, Optional
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Вспомогательные векторные операции
def norm(vec: List[float]) -> float:
    return math.sqrt(sum(v*v for v in vec))

def scalar_mul(scalar: float, vec: List[float]) -> List[float]:
    return [scalar * v for v in vec]

def vec_sub(vec1: List[float], vec2: List[float]) -> List[float]:
    return [v1 - v2 for v1, v2 in zip(vec1, vec2)]

def vec_add(vec1: List[float], vec2: List[float]) -> List[float]:
    return [v1 + v2 for v1, v2 in zip(vec1, vec2)]

# Фундаментальное решение (скалярное) — такое же, как в 1D
def phi_neutral(t, a, b, h):
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

def Delta_vector(T, x0: List[float], y0: List[float],
                 a1, b1, a2, b2, h):
    phi1_T = phi_neutral(T, a1, b1, h)
    phi1_Th = phi_neutral(T - h, a1, b1, h) if T >= h else 0.0
    phi2_T = phi_neutral(T, a2, b2, h)
    phi2_Th = phi_neutral(T - h, a2, b2, h) if T >= h else 0.0
    J1 = J_integral(T, a1, b1, h)
    J2 = J_integral(T, a2, b2, h)
    term1 = scalar_mul(phi1_T - a1 * phi1_Th, x0)
    term2 = scalar_mul(phi2_T - a2 * phi2_Th, y0)
    term3 = scalar_mul(b1 * J1, x0)
    term4 = scalar_mul(b2 * J2, y0)
    return vec_add(vec_sub(term1, term2), vec_sub(term3, term4))

def Delta_norm(T, x0, y0, a1, b1, a2, b2, h):
    return norm(Delta_vector(T, x0, y0, a1, b1, a2, b2, h))

def R(T, alpha, beta, a1, b1, a2, b2, h):
    I1 = integral_phi(T, a1, b1, h)
    I2 = integral_phi(T, a2, b2, h)
    return alpha * I1 - beta * I2

def find_T0(x0: List[float], y0: List[float],
            alpha, beta, a1, b1, a2, b2, h,
            T_max=500.0, eps=1e-6, verbose=True):
    diff = vec_sub(x0, y0)
    if norm(diff) < eps:
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
                print(f"Достигнут T_max = {T_max}")
            return None
        delta_n = Delta_norm(T, x0, y0, a1, b1, a2, b2, h)
        R_val = R(T, alpha, beta, a1, b1, a2, b2, h)
        f_val = delta_n - R_val
        if verbose:
            print(f"T={T:.2f}, ||Δ||={delta_n:.6e}, R={R_val:.6e}, f={f_val:.6e}")
        if f_val <= 0:
            T_high = T
            break
        T_low = T
        m += 1

    if T_high is None:
        return None

    while T_high - T_low > eps:
        T_mid = (T_low + T_high) / 2.0
        delta_m = Delta_norm(T_mid, x0, y0, a1, b1, a2, b2, h)
        R_m = R(T_mid, alpha, beta, a1, b1, a2, b2, h)
        if delta_m - R_m > 0:
            T_low = T_mid
        else:
            T_high = T_mid
    return (T_low + T_high) / 2.0

if __name__ == "__main__":
    params = {
        'a1': 0.10, 'b1': 0.40,
        'a2': 0.05, 'b2': 0.20,
        'h': 0.4,
        'alpha': 15.0, 'beta': 8.0,
        'z01': [0.0, 0.0, 0.0],
        'z02': [100.0, 80.0, 60.0]
    }
    T0 = find_T0(
        x0=params['z01'], y0=params['z02'],
        alpha=params['alpha'], beta=params['beta'],
        a1=params['a1'], b1=params['b1'],
        a2=params['a2'], b2=params['b2'],
        h=params['h'],
        T_max=500.0, eps=1e-6, verbose=True
    )
    if T0 is not None:
        print(f"T0 = {T0:.6f}")