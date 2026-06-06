#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль реализует рекуррентный алгоритм вычисления оптимального времени T0
в задаче преследования с запаздыванием для трёхмерного пространства E = R^3.

Основан на псевдокоде из раздела "Рекуррентные формулы для вычисления J_i(T) и других величин"
и адаптирован для векторных начальных позиций.
"""

import math
from typing import Tuple, Optional, List, Union

# ------------------------------------------------------------
# Вспомогательные векторные операции
# ------------------------------------------------------------
def norm(vec: List[float]) -> float:
    """Евклидова норма вектора."""
    return math.sqrt(sum(v * v for v in vec))

def scalar_mul(scalar: float, vec: List[float]) -> List[float]:
    """Умножение вектора на скаляр."""
    return [scalar * v for v in vec]

def vec_sub(vec1: List[float], vec2: List[float]) -> List[float]:
    """Разность двух векторов."""
    return [v1 - v2 for v1, v2 in zip(vec1, vec2)]

def vec_add(vec1: List[float], vec2: List[float]) -> List[float]:
    """Сумма двух векторов."""
    return [v1 + v2 for v1, v2 in zip(vec1, vec2)]

# ------------------------------------------------------------
# Рекуррентное вычисление φ_i(t) (лемма 5.1)
# ------------------------------------------------------------
def phi_recurrent(a: float, b: float, h: float, t: float) -> float:
    """
    Вычисляет φ(t) = Σ_{k=0}^{⌊t/h⌋} b^k * (t - k*h)^k / k! * exp(a*(t - k*h))
    с использованием рекуррентного соотношения для слагаемых.
    """
    if h <= 0:
        raise ValueError("h должно быть положительным")
    if t < 0:
        return 0.0
    n = int(t // h)
    if n < 0:
        return 0.0
    c = math.exp(a * t)
    total = c
    for k in range(n):
        delta_k = t - k * h
        delta_k1 = t - (k + 1) * h
        if delta_k == 0.0 or delta_k1 == 0.0:
            break
        log_c = math.log(c) + math.log(b) - a * h - math.log(k + 1) + \
                (k + 1) * math.log(abs(delta_k1)) - k * math.log(abs(delta_k))
        c = math.exp(log_c)
        total += c
        if abs(c) < 1e-15 * abs(total):
            break
    return total

# ------------------------------------------------------------
# Рекуррентное вычисление J(0, U, k, a) (лемма 5.3)
# ------------------------------------------------------------
def J0_recurrent(U: float, k: int, a: float) -> float:
    """
    Вычисляет J(0, U, k, a) = ∫_0^U (w^k / k!) * exp(a*w) dw
    рекуррентно по k.
    """
    if a < 0:
        raise ValueError("a должна быть неотрицательной")
    if U < 0:
        return 0.0
    if a == 0.0:
        return (U ** (k + 1)) / math.factorial(k + 1)
    if k == 0:
        return (math.exp(a * U) - 1.0) / a
    J_prev = (math.exp(a * U) - 1.0) / a
    fact = 1.0
    for i in range(1, k + 1):
        fact *= i
        term = (U ** i) * math.exp(a * U) / (a * fact)
        J_curr = term - J_prev / a
        J_prev = J_curr
    return J_prev

# ------------------------------------------------------------
# Вычисление J_i(T) через J(0,*,*,a) (лемма 5.4)
# ------------------------------------------------------------
def J_i_recurrent(T: float, a: float, b: float, h: float) -> float:
    """
    Вычисляет J_i(T) = ∫_{T-h}^{T} φ_i(t) dt
    по формуле (J:recur) с использованием J0_recurrent.
    """
    if h <= 0:
        raise ValueError("h > 0")
    if T < 0:
        return 0.0
    n_full = int((T - h) // h) if T - h >= 0 else -1
    n_total = int(T // h)
    total = 0.0
    for k in range(n_full + 1):
        U1 = T - k * h
        U2 = T - h - k * h
        if U2 < 0:
            U2 = 0.0
        total += (b ** k) * (J0_recurrent(U1, k, a) - J0_recurrent(U2, k, a))
    for k in range(n_full + 1, n_total + 1):
        U = T - k * h
        if U < 0:
            U = 0.0
        total += (b ** k) * J0_recurrent(U, k, a)
    return total

# ------------------------------------------------------------
# Рекуррентное вычисление скалярной суммы S_i(T) (лемма 5.5)
# ------------------------------------------------------------
def S_i_scalar_recurrent(T: float, a: float, b: float, h: float) -> float:
    """
    Вычисляет S_i(T) = Σ_{k=0}^{⌊T/h⌋} b^k * (T - k*h)^k / k! * exp(a*(T - k*h))
    (без умножения на z0). Используется для дальнейшего умножения на вектор z0.
    """
    if h <= 0:
        raise ValueError("h > 0")
    if T < 0:
        return 0.0
    n = int(T // h)
    if n < 0:
        return 0.0
    c = math.exp(a * T)
    total = c
    for k in range(n):
        delta_k = T - k * h
        delta_k1 = T - (k + 1) * h
        if delta_k == 0.0 or delta_k1 == 0.0:
            break
        log_c = math.log(abs(c)) + math.log(b) - a * h - math.log(k + 1) + \
                (k + 1) * math.log(abs(delta_k1)) - k * math.log(abs(delta_k))
        c = math.exp(log_c) * (1.0 if c >= 0 else -1.0)
        total += c
        if abs(c) < 1e-15 * abs(total):
            break
    return total

# ------------------------------------------------------------
# Вычисление Δ(T) (вектор) и его нормы
# ------------------------------------------------------------
def compute_Delta_vector(T: float, params: dict) -> List[float]:
    """
    Возвращает вектор Δ(T) ∈ R^3.
    Параметры: a1,b1,a2,b2,h, z01 (list), z02 (list)
    """
    a1, b1 = params['a1'], params['b1']
    a2, b2 = params['a2'], params['b2']
    h = params['h']
    z01 = params['z01']   # list of 3 floats
    z02 = params['z02']   # list of 3 floats

    S1_scalar = S_i_scalar_recurrent(T, a1, b1, h)
    S2_scalar = S_i_scalar_recurrent(T, a2, b2, h)
    J1 = J_i_recurrent(T, a1, b1, h)
    J2 = J_i_recurrent(T, a2, b2, h)

    S1_vec = scalar_mul(S1_scalar, z01)
    S2_vec = scalar_mul(S2_scalar, z02)
    term1_vec = scalar_mul(b1 * J1, z01)
    term2_vec = scalar_mul(b2 * J2, z02)

    Delta_vec = vec_sub(vec_sub(S1_vec, S2_vec), vec_sub(term1_vec, term2_vec))
    return Delta_vec

def compute_Delta_norm(T: float, params: dict) -> float:
    """Вычисляет норму вектора Δ(T)."""
    return norm(compute_Delta_vector(T, params))

# ------------------------------------------------------------
# Вычисление R(T) (скаляр)
# ------------------------------------------------------------
def compute_R_recurrent(T: float, params: dict) -> float:
    """
    Вычисляет R(T) = ∫_0^T (α φ1(s) - β φ2(s)) ds
    через сумму по k интегралов J(0, T-kh, k, a_i).
    """
    alpha = params['alpha']
    beta = params['beta']
    a1, b1 = params['a1'], params['b1']
    a2, b2 = params['a2'], params['b2']
    h = params['h']
    n = int(T // h)

    def integral_phi(a, b):
        total = 0.0
        for k in range(n + 1):
            U = T - k * h
            if U < 0:
                U = 0.0
            total += (b ** k) * J0_recurrent(U, k, a)
        return total

    I1 = integral_phi(a1, b1)
    I2 = integral_phi(a2, b2)
    return alpha * I1 - beta * I2

# ------------------------------------------------------------
# Проверка поточечного условия 1 (рекуррентное вычисление φ)
# ------------------------------------------------------------
def check_condition1_recurrent(params: dict, T_max: float) -> bool:
    """
    Проверяет выполнение условия φ1(t)*α ≥ φ2(t)*β для всех t ∈ [0, T_max].
    Использует рекуррентное вычисление φ и дискретизацию с шагом h/2.
    """
    h = params['h']
    a1, b1 = params['a1'], params['b1']
    a2, b2 = params['a2'], params['b2']
    alpha, beta = params['alpha'], params['beta']
    n = int(T_max // h)

    for k in range(n + 1):
        t = k * h
        phi1 = phi_recurrent(a1, b1, h, t)
        phi2 = phi_recurrent(a2, b2, h, t)
        if phi1 * alpha < phi2 * beta:
            return False

    for k in range(n):
        t = (k + 0.5) * h
        phi1 = phi_recurrent(a1, b1, h, t)
        phi2 = phi_recurrent(a2, b2, h, t)
        if phi1 * alpha < phi2 * beta:
            return False

    if abs(T_max - n * h) > 1e-12:
        phi1 = phi_recurrent(a1, b1, h, T_max)
        phi2 = phi_recurrent(a2, b2, h, T_max)
        if phi1 * alpha < phi2 * beta:
            return False
    return True

# ------------------------------------------------------------
# Основной алгоритм поиска T0 (рекуррентный, 3D)
# ------------------------------------------------------------
def find_T0_recurrent_3d(params: dict, eps: float = 1e-6, nmax: int = 1000) -> Optional[float]:
    """
    Реализует алгоритм \ref{alg:recurrent_T0} для R^3.
    Возвращает T0 или None (если неразрешимо).
    """
    # Тривиальный случай
    delta0_vec = vec_sub(params['z01'], params['z02'])
    if norm(delta0_vec) <= eps:
        return 0.0

    # Проверка условия 1 на [0, nmax*h]
    if not check_condition1_recurrent(params, nmax * params['h']):
        print("Решение не найдено: поточечное условие не выполнено")
        return None

    n = 1
    while n <= nmax:
        T = n * params['h']
        Delta_norm = compute_Delta_norm(T, params)
        R_val = compute_R_recurrent(T, params)
        if Delta_norm <= R_val:
            a = (n - 1) * params['h']
            b = n * params['h']
            break
        n += 1
    else:
        print(f"Неразрешимо: достигнут T_max = {nmax * params['h']}")
        return None

    # Уточнение бисекцией
    while b - a > eps:
        c = (a + b) / 2.0
        Delta_norm = compute_Delta_norm(c, params)
        R_c = compute_R_recurrent(c, params)
        if Delta_norm <= R_c:
            b = c
        else:
            a = c
    return (a + b) / 2.0

# ------------------------------------------------------------
# Пример использования (трёхмерный случай из статьи)
# ------------------------------------------------------------
if __name__ == "__main__":
    # Параметры для трёхмерного эксперимента (раздел 10.3)
    parameters_3d = {
        'a1': 0.10, 'b1': 0.40,
        'a2': 0.05, 'b2': 0.20,
        'h': 0.3,
        'alpha': 15.0, 'beta': 8.0,
        'z01': [0.0, 0.0, 0.0],
        'z02': [100.0, 80.0, 60.0]
    }
    T0 = find_T0_recurrent_3d(parameters_3d, eps=1e-6, nmax=500)
    if T0 is not None:
        print(f"Оптимальное время преследования (3D, рекуррентный метод): T0 = {T0:.6f} с")
        # Сравнение с классическим временем
        delta_norm = norm(parameters_3d['z02'])
        alpha_beta = parameters_3d['alpha'] - parameters_3d['beta']
        T_classic = delta_norm / alpha_beta
        print(f"Классическое время без запаздывания: T_classic = {T_classic:.6f} с")
    else:
        print("Задача неразрешима за разумное время.")