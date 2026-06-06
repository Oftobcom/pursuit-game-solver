#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль реализует алгоритм вычисления оптимального времени T0 по явным формулам
из раздела "Явные формулы для вычисления интегралов J_i(T)".
Адаптирован для трёхмерного пространства E = R^3.
"""

import math
from typing import List, Optional, Union

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
# Вспомогательные функции для явных формул
# ------------------------------------------------------------
def compute_phi(a: float, b: float, h: float, t: float) -> float:
    """
    Функция ϕ(t) = Σ_{k=0}^{⌊t/h⌋} b^k * (t - k*h)^k / k! * exp(a*(t - k*h)).
    """
    if h <= 0:
        raise ValueError("h должно быть положительным")
    n = int(t // h)
    total = 0.0
    for k in range(n + 1):
        delta = t - k * h
        if delta == 0.0 and k > 0:
            term = 0.0
        else:
            try:
                term = (b ** k) * (delta ** k) / math.factorial(k) * math.exp(a * delta)
            except OverflowError:
                log_term = k * math.log(b) + k * math.log(delta) - math.log(math.factorial(k)) + a * delta
                term = math.exp(log_term)
        total += term
    return total

def Qk(xi: float, a: float, k: int) -> float:
    """
    Полином Q_k(ξ) = Σ_{j=0}^{k} (-1)^j * ξ^{k-j} / ((k-j)! * a^j)
    для a > 0.
    """
    s = 0.0
    for j in range(k + 1):
        if k - j == 0:
            term = (-1) ** j / (math.factorial(0) * (a ** j))
        else:
            term = (-1) ** j * (xi ** (k - j)) / (math.factorial(k - j) * (a ** j))
        s += term
    return s

def J_integral_aux(L: float, U: float, k: int, a: float) -> float:
    """
    Вычисляет J(L,U,k,a) = ∫_{L}^{U} (w^k / k!) * exp(a*w) dw.
    """
    if a > 0:
        if L == 0.0:
            return (math.exp(a * U) / a) * Qk(U, a, k) - ((-1) ** k) / (a ** (k + 1))
        else:
            return (math.exp(a * U) / a) * Qk(U, a, k) - (math.exp(a * L) / a) * Qk(L, a, k)
    elif a == 0:
        return (U ** (k + 1) / math.factorial(k + 1)) - (L ** (k + 1) / math.factorial(k + 1))
    else:
        raise ValueError("a должна быть неотрицательной")

def J_i(T: float, a: float, b: float, h: float) -> float:
    """
    Вычисляет J_i(T) = ∫_{T-h}^{T} ϕ_i(t) dt по явным формулам (6) или (7).
    """
    if h <= 0:
        raise ValueError("h > 0")
    n_full = int((T - h) // h) if T - h >= 0 else -1
    n_total = int(T // h)
    total = 0.0
    if a > 0:
        for k in range(0, n_full + 1):
            arg1 = T - k * h
            arg2 = T - h - k * h
            term1 = math.exp(a * arg1) / a * Qk(arg1, a, k)
            term2 = math.exp(a * arg2) / a * Qk(arg2, a, k)
            total += (b ** k) * (term1 - term2)
        for k in range(n_full + 1, n_total + 1):
            arg = T - k * h
            term = (math.exp(a * arg) / a) * Qk(arg, a, k) - ((-1) ** k) / (a ** (k + 1))
            total += (b ** k) * term
    else:  # a == 0
        for k in range(0, n_full + 1):
            delta1 = T - k * h
            delta2 = T - h - k * h
            total += (b ** k) * ((delta1 ** (k + 1) - delta2 ** (k + 1)) / math.factorial(k + 1))
        for k in range(n_full + 1, n_total + 1):
            delta = T - k * h
            total += (b ** k) * (delta ** (k + 1) / math.factorial(k + 1))
    return total

# ------------------------------------------------------------
# Скалярная сумма S_i_scalar (без умножения на z0)
# ------------------------------------------------------------
def S_i_scalar(T: float, a: float, b: float, h: float) -> float:
    """
    Вычисляет скалярную сумму
    Σ_{k=0}^{⌊T/h⌋} b^k * (T - k*h)^k / k! * exp(a*(T - k*h))
    """
    n = int(T // h)
    total = 0.0
    for k in range(n + 1):
        delta = T - k * h
        if delta == 0.0 and k > 0:
            term = 0.0
        else:
            term = (b ** k) * (delta ** k) / math.factorial(k) * math.exp(a * delta)
        total += term
    return total

# ------------------------------------------------------------
# Векторные функции для Δ(T)
# ------------------------------------------------------------
def compute_Delta_vector(T: float, params: dict) -> List[float]:
    """
    Возвращает вектор Δ(T) ∈ R^3 по формуле (Delta:main).
    Параметры: a1,b1,a2,b2,h, z01 (list), z02 (list)
    """
    a1, b1 = params['a1'], params['b1']
    a2, b2 = params['a2'], params['b2']
    h = params['h']
    z01 = params['z01']   # list of 3
    z02 = params['z02']   # list of 3

    S1_scalar = S_i_scalar(T, a1, b1, h)
    S2_scalar = S_i_scalar(T, a2, b2, h)
    J1 = J_i(T, a1, b1, h)
    J2 = J_i(T, a2, b2, h)

    term1_vec = scalar_mul(S1_scalar, z01)
    term2_vec = scalar_mul(S2_scalar, z02)
    term3_vec = scalar_mul(b1 * J1, z01)
    term4_vec = scalar_mul(b2 * J2, z02)

    Delta_vec = vec_sub(vec_sub(term1_vec, term2_vec), vec_sub(term3_vec, term4_vec))
    return Delta_vec

def compute_Delta_norm(T: float, params: dict) -> float:
    """Вычисляет норму вектора Δ(T)."""
    return norm(compute_Delta_vector(T, params))

# ------------------------------------------------------------
# Функция R(T)
# ------------------------------------------------------------
def compute_R(T: float, params: dict) -> float:
    """
    Вычисляет R(T) = ∫_0^T (α*ϕ1(s) - β*ϕ2(s)) ds
    (скалярный радиус).
    """
    alpha = params['alpha']
    beta = params['beta']
    a1, b1 = params['a1'], params['b1']
    a2, b2 = params['a2'], params['b2']
    h = params['h']

    def int_phi(a, b):
        n = int(T // h)
        total = 0.0
        for k in range(n + 1):
            U = T - k * h
            if U < 0:
                U = 0.0
            total += (b ** k) * J_integral_aux(0.0, U, k, a)
        return total

    I1 = int_phi(a1, b1)
    I2 = int_phi(a2, b2)
    return alpha * I1 - beta * I2

# ------------------------------------------------------------
# Проверка поточечного условия 1
# ------------------------------------------------------------
def check_condition1(params: dict, T: float) -> bool:
    """
    Проверка поточечного условия на [0, T] с шагом h/2.
    """
    h = params['h']
    a1, b1 = params['a1'], params['b1']
    a2, b2 = params['a2'], params['b2']
    alpha, beta = params['alpha'], params['beta']
    n = int(T // h)

    for k in range(n + 1):
        t = k * h
        phi1 = compute_phi(a1, b1, h, t)
        phi2 = compute_phi(a2, b2, h, t)
        if phi1 * alpha < phi2 * beta:
            return False
    for k in range(n):
        t = (k + 0.5) * h
        phi1 = compute_phi(a1, b1, h, t)
        phi2 = compute_phi(a2, b2, h, t)
        if phi1 * alpha < phi2 * beta:
            return False
    if abs(T - n * h) > 1e-12:
        phi1 = compute_phi(a1, b1, h, T)
        phi2 = compute_phi(a2, b2, h, T)
        if phi1 * alpha < phi2 * beta:
            return False
    return True

# ------------------------------------------------------------
# Основной алгоритм поиска T0 (явный, 3D)
# ------------------------------------------------------------
def find_T0(params: dict, eps: float = 1e-6, Nmax: int = 1000) -> Optional[float]:
    """
    Поиск минимального времени T0 по явным формулам.
    Параметры: a1,b1,a2,b2,h,alpha,beta, z01 (list), z02 (list)
    Возвращает T0 либо None.
    """
    # Тривиальный случай
    delta0_vec = vec_sub(params['z01'], params['z02'])
    if norm(delta0_vec) <= eps:
        return 0.0

    # Проверка условия 1
    if not check_condition1(params, Nmax * params['h']):
        print("Решение не найдено: поточечное условие не выполнено")
        return None

    # Поиск интервала [a,b] перебором T = n*h
    n = 1
    while n <= Nmax:
        T = n * params['h']
        Delta_norm = compute_Delta_norm(T, params)
        R_val = compute_R(T, params)
        if Delta_norm <= R_val:
            a = (n - 1) * params['h']
            b = n * params['h']
            break
        n += 1
    else:
        print(f"Неразрешимо: достигнут T_max = {Nmax * params['h']}")
        return None

    # Уточнение бисекцией
    while b - a > eps:
        c = (a + b) / 2.0
        Delta_norm = compute_Delta_norm(c, params)
        R_c = compute_R(c, params)
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
    T0 = find_T0(parameters_3d, eps=1e-6, Nmax=500)
    if T0 is not None:
        print(f"Оптимальное время преследования (явный метод, 3D): T0 = {T0:.6f} с")
        # Сравнение с классическим временем
        delta_norm = norm(parameters_3d['z02'])
        alpha_beta = parameters_3d['alpha'] - parameters_3d['beta']
        T_classic = delta_norm / alpha_beta
        print(f"Классическое время без запаздывания: T_classic = {T_classic:.6f} с")
    else:
        print("Задача неразрешима за разумное время.")