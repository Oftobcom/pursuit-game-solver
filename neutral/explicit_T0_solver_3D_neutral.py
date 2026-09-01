#explicit_T0_solver_3D_neutral.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Явный алгоритм для нейтральной системы (3D) с аналитическими формулами J_i.
"""

import math
from typing import List, Optional

# Векторные операции (те же)
def norm(vec):
    return math.sqrt(sum(v*v for v in vec))
def scalar_mul(scalar, vec):
    return [scalar * v for v in vec]
def vec_sub(v1, v2):
    return [v1[i]-v2[i] for i in range(len(v1))]
def vec_add(v1, v2):
    return [v1[i]+v2[i] for i in range(len(v1))]

# Скалярные функции (такие же, как в explicit_1D)
def phi_neutral_explicit(t, a, b, h):
    if t < 0:
        return 0.0
    n = int(math.floor(t / h))
    total = 1.0
    for k in range(1, n + 1):
        tau = t - k * h
        if abs(tau) < 1e-07:
            continue
        for j in range(1, k + 1):
            coeff = math.comb(k - 1, j - 1)
            term = coeff * (a ** (k - j)) * (b ** j) * (tau ** j) / math.factorial(j)
            total += term
    return total

def ell(T, h):
    return T if 0 <= T < h else h if T >= h else 0.0

def J_i_explicit(T, a, b, h):
    if T < 0:
        return 0.0
    n1 = int(math.floor((T - h) / h)) if T - h >= 0 else -1
    n2 = int(math.floor(T / h))
    total = ell(T, h)
    for k in range(1, n1 + 1):
        p = T - k * h
        q = T - h - k * h
        for j in range(1, k + 1):
            coeff = math.comb(k - 1, j - 1)
            term = coeff * (a ** (k - j)) * (b ** j) * ((p ** (j+1) - q ** (j+1)) / math.factorial(j+1))
            total += term
    tau = T - n2 * h
    for j in range(1, n2 + 1):
        coeff = math.comb(n2 - 1, j - 1)
        term = coeff * (a ** (n2 - j)) * (b ** j) * (tau ** (j+1) / math.factorial(j+1))
        total += term
    return total

def Delta_vector_explicit(T, x0, y0, a1, b1, a2, b2, h):
    phi1_T = phi_neutral_explicit(T, a1, b1, h)
    phi1_Th = phi_neutral_explicit(T - h, a1, b1, h) if T >= h else 0.0
    phi2_T = phi_neutral_explicit(T, a2, b2, h)
    phi2_Th = phi_neutral_explicit(T - h, a2, b2, h) if T >= h else 0.0
    J1 = J_i_explicit(T, a1, b1, h)
    J2 = J_i_explicit(T, a2, b2, h)
    term1 = scalar_mul(phi1_T - a1 * phi1_Th, x0)
    term2 = scalar_mul(phi2_T - a2 * phi2_Th, y0)
    term3 = scalar_mul(b1 * J1, x0)
    term4 = scalar_mul(b2 * J2, y0)
    return vec_add(vec_sub(term1, term2), vec_sub(term3, term4))

def Delta_norm_explicit(T, x0, y0, a1, b1, a2, b2, h):
    return norm(Delta_vector_explicit(T, x0, y0, a1, b1, a2, b2, h))

def R_explicit(T, alpha, beta, a1, b1, a2, b2, h):
    # Используем накопление аналогично 1D
    if T <= 0:
        return 0.0
    n = int(math.floor(T / h))
    I1 = 0.0
    I2 = 0.0
    for m in range(n):
        I1 += J_i_explicit((m+1)*h, a1, b1, h)
        I2 += J_i_explicit((m+1)*h, a2, b2, h)
    if T > n * h:
        # Интегрируем остаток аналитически (полином)
        # Для простоты используем численное интегрирование (но можно и аналитически)
        from scipy.integrate import quad
        def phi1(s):
            return phi_neutral_explicit(s, a1, b1, h)
        def phi2(s):
            return phi_neutral_explicit(s, a2, b2, h)
        val1, _ = quad(phi1, n*h, T, epsabs=1e-12)
        val2, _ = quad(phi2, n*h, T, epsabs=1e-12)
        I1 += val1
        I2 += val2
    return alpha * I1 - beta * I2

def find_T0_explicit(params, eps=1e-6, nmax=1000):
    x0, y0 = params['z01'], params['z02']
    alpha, beta = params['alpha'], params['beta']
    a1, b1 = params['a1'], params['b1']
    a2, b2 = params['a2'], params['b2']
    h = params['h']

    if norm(vec_sub(x0, y0)) < eps:
        return 0.0

    n = 1
    while n <= nmax:
        T = n * h
        delta_n = Delta_norm_explicit(T, x0, y0, a1, b1, a2, b2, h)
        R_val = R_explicit(T, alpha, beta, a1, b1, a2, b2, h)
        if delta_n <= R_val:
            a = (n - 1) * h
            b = n * h
            break
        n += 1
    else:
        print(f"Неразрешимо до T_max = {nmax * h}")
        return None

    while b - a > eps:
        c = (a + b) / 2.0
        delta_c = Delta_norm_explicit(c, x0, y0, a1, b1, a2, b2, h)
        R_c = R_explicit(c, alpha, beta, a1, b1, a2, b2, h)
        if delta_c <= R_c:
            b = c
        else:
            a = c
    return (a + b) / 2.0

if __name__ == "__main__":
    params = {
        'a1': 0.10, 'b1': 0.40,
        'a2': 0.05, 'b2': 0.20,
        'h': 0.4,
        'alpha': 15.0, 'beta': 8.0,
        'z01': [0.0, 0.0, 0.0],
        'z02': [100.0, 80.0, 60.0]
    }
    T0 = find_T0_explicit(params, eps=1e-6, nmax=500)
    if T0 is not None:
        print(f"Явный 3D: T0 = {T0:.6f}")