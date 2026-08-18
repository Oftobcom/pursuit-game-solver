#recurrent_T0_solver_3D_neutral.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Рекуррентный алгоритм для нейтральной системы (3D).
"""

import math
from typing import List, Optional

# Векторные операции
def norm(vec):
    return math.sqrt(sum(v*v for v in vec))
def scalar_mul(scalar, vec):
    return [scalar * v for v in vec]
def vec_sub(v1, v2):
    return [v1[i]-v2[i] for i in range(len(v1))]
def vec_add(v1, v2):
    return [v1[i]+v2[i] for i in range(len(v1))]

# Скалярные рекуррентные функции (из 1D рекуррентного)
def phi_recurrent_neutral(t, a, b, h):
    if t < 0:
        return 0.0
    n = int(math.floor(t / h))
    total = 1.0
    for k in range(1, n+1):
        tau = t - k * h
        if tau == 0.0:
            continue
        if a > 0:
            c = (a ** (k-1)) * b * tau
            total += c
            for j in range(1, k):
                c = c * (k - j) / j * (b / a) * tau / (j + 1)
                total += c
        else:
            total += (b ** k) * (tau ** k) / math.factorial(k)
    return total

def J_i_recurrent_neutral(T, a, b, h):
    if T < 0:
        return 0.0
    n1 = int(math.floor((T - h) / h)) if T - h >= 0 else -1
    n2 = int(math.floor(T / h))
    total = (T if T < h else h) if T >= 0 else 0.0  # ell(T)
    if n1 >= 1:
        for k in range(1, n1+1):
            p = T - k * h
            q = T - h - k * h
            if a > 0:
                d = (a ** (k-1)) * b * (p*p - q*q) / 2.0
            else:
                d = b * (p*p - q*q) / 2.0 if k == 1 else 0.0
            total += d
            for j in range(1, k):
                if a > 0:
                    d = d * (k - j) / j * (b / a) * (p**(j+2) - q**(j+2)) / ((j+2) * (p**(j+1) - q**(j+1)))
                else:
                    break
                total += d
    if n2 >= 1:
        tau = T - n2 * h
        if tau > 0:
            if a > 0:
                e = (a ** (n2-1)) * b * (tau*tau) / 2.0
            else:
                e = b * (tau*tau) / 2.0 if n2 == 1 else 0.0
            total += e
            for j in range(1, n2):
                if a > 0:
                    e = e * (n2 - j) / j * (b / a) * tau / (j + 2)
                else:
                    break
                total += e
    return total

def Delta_vector_recurrent(T, x0, y0, a1, b1, a2, b2, h):
    phi1_T = phi_recurrent_neutral(T, a1, b1, h)
    phi1_Th = phi_recurrent_neutral(T - h, a1, b1, h) if T >= h else 0.0
    phi2_T = phi_recurrent_neutral(T, a2, b2, h)
    phi2_Th = phi_recurrent_neutral(T - h, a2, b2, h) if T >= h else 0.0
    J1 = J_i_recurrent_neutral(T, a1, b1, h)
    J2 = J_i_recurrent_neutral(T, a2, b2, h)
    term1 = scalar_mul(phi1_T - a1 * phi1_Th, x0)
    term2 = scalar_mul(phi2_T - a2 * phi2_Th, y0)
    term3 = scalar_mul(b1 * J1, x0)
    term4 = scalar_mul(b2 * J2, y0)
    return vec_add(vec_sub(term1, term2), vec_sub(term3, term4))

def Delta_norm_recurrent(T, x0, y0, a1, b1, a2, b2, h):
    return norm(Delta_vector_recurrent(T, x0, y0, a1, b1, a2, b2, h))

def R_recurrent(T, alpha, beta, a1, b1, a2, b2, h):
    if T <= 0:
        return 0.0
    n = int(math.floor(T / h))
    I1 = 0.0
    I2 = 0.0
    for m in range(n):
        I1 += J_i_recurrent_neutral((m+1)*h, a1, b1, h)
        I2 += J_i_recurrent_neutral((m+1)*h, a2, b2, h)
    if T > n * h:
        # Интегрируем остаток аналитически (аналогично 1D)
        delta_T = T - n * h
        I1 += delta_T
        I2 += delta_T
        for k in range(1, n+1):
            for j in range(1, k+1):
                coeff1 = math.comb(k-1, j-1) * (a1 ** (k-j)) * (b1 ** j) / math.factorial(j)
                coeff2 = math.comb(k-1, j-1) * (a2 ** (k-j)) * (b2 ** j) / math.factorial(j)
                term1 = coeff1 * ((T - k*h) ** (j+1) - (n*h - k*h) ** (j+1)) / (j+1)
                term2 = coeff2 * ((T - k*h) ** (j+1) - (n*h - k*h) ** (j+1)) / (j+1)
                I1 += term1
                I2 += term2
    return alpha * I1 - beta * I2

def find_T0_recurrent(params, eps=1e-6, nmax=1000):
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
        delta_n = Delta_norm_recurrent(T, x0, y0, a1, b1, a2, b2, h)
        R_val = R_recurrent(T, alpha, beta, a1, b1, a2, b2, h)
        if delta_n <= R_val:
            a = (n - 1) * h
            b = n * h
            break
        n += 1
    else:
        print(f"Неразрешимо до T_max = {nmax*h}")
        return None

    while b - a > eps:
        c = (a + b) / 2.0
        delta_c = Delta_norm_recurrent(c, x0, y0, a1, b1, a2, b2, h)
        R_c = R_recurrent(c, alpha, beta, a1, b1, a2, b2, h)
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
    T0 = find_T0_recurrent(params, eps=1e-6, nmax=500)
    if T0 is not None:
        print(f"Рекуррентный 3D: T0 = {T0:.6f}")