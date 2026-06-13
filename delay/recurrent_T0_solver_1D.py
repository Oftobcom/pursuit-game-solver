#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль реализует рекуррентный алгоритм вычисления оптимального времени T0
в задаче преследования с запаздыванием (одномерный случай E = R).

Основан на псевдокоде из раздела "Рекуррентные формулы для вычисления J_i(T) и других величин"
(алгоритм \ref{alg:recurrent_T0}).

Имя файла: recurrent_pursuit_T0.py
"""

import math
from typing import Tuple, Optional

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
    # c0
    c = math.exp(a * t)
    total = c
    for k in range(n):
        delta_k = t - k * h
        delta_k1 = t - (k + 1) * h
        # Если текущий или следующий член обращается в ноль – дальнейшие равны нулю
        if delta_k == 0.0 or delta_k1 == 0.0:
            # при delta_k = 0 и k > 0, c_k = 0, тогда и все последующие нули
            break
        # c_{k+1} = c_k * (b * e^{-a*h} / (k+1)) * (delta_{k+1}^{k+1} / delta_k^k)
        # Вычисляем через логарифмы для устойчивости
        log_c = math.log(c) + math.log(b) - a * h - math.log(k + 1) + \
                (k + 1) * math.log(abs(delta_k1)) - k * math.log(abs(delta_k))
        c = math.exp(log_c)
        total += c
        # Досрочное завершение, если слагаемое стало пренебрежимо малым
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
        # J(0,U,k,0) = U^{k+1} / (k+1)!
        return (U ** (k + 1)) / math.factorial(k + 1)
    # a > 0
    if k == 0:
        return (math.exp(a * U) - 1.0) / a
    # Рекуррентно вычисляем J для k-1, затем по формуле
    # J(0,U,k,a) = (U^k * e^{aU}) / (a * k!) - (1/a) * J(0,U,k-1,a)
    # Для устойчивости используем накопление
    J_prev = (math.exp(a * U) - 1.0) / a
    fact = 1.0
    for i in range(1, k + 1):
        fact *= i  # i!
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
    b_pow = 1.0
    # Первая сумма: k = 0 .. floor((T-h)/h)
    for k in range(n_full + 1):
        U1 = T - k * h
        U2 = T - h - k * h
        if U2 < 0:
            U2 = 0.0
        total += b_pow * (J0_recurrent(U1, k, a) - J0_recurrent(U2, k, a))
        b_pow *= b
    # Второе слагаемое
    U = T - n_total * h
    if U < 0:
        U = 0.0
    total += b ** n_total * J0_recurrent(U, n_total, a)
    return total


# ------------------------------------------------------------
# Рекуррентное вычисление S_i(T) (лемма 5.5)
# ------------------------------------------------------------
def S_i_recurrent(T: float, a: float, b: float, h: float, z0: float) -> float:
    """
    Вычисляет S_i(T) = Σ_{k=0}^{⌊T/h⌋} b^k * (T - k*h)^k / k! * exp(a*(T - k*h)) * z0
    используя рекуррентное соотношение (5.8).
    """
    if z0 == 0.0:
        return 0.0
    if h <= 0:
        raise ValueError("h > 0")
    if T < 0:
        return 0.0
    n = int(T // h)
    if n < 0:
        return 0.0
    c = math.exp(a * T) * z0
    total = c
    for k in range(n):
        delta_k = T - k * h
        delta_k1 = T - (k + 1) * h
        if delta_k == 0.0 or delta_k1 == 0.0:
            break
        # c_{k+1} = c_k * (b * e^{-a*h} / (k+1)) * (delta_{k+1}^{k+1} / delta_k^k)
        log_c = math.log(abs(c)) + math.log(b) - a * h - math.log(k + 1) + \
                (k + 1) * math.log(abs(delta_k1)) - k * math.log(abs(delta_k))
        c = math.exp(log_c) * (1.0 if c >= 0 else -1.0)
        total += c
        if abs(c) < 1e-15 * abs(total):
            break
    return total


# ------------------------------------------------------------
# Вычисление Δ(T) и R(T) (скалярный случай)
# ------------------------------------------------------------
def compute_Delta_recurrent(T: float, params: dict) -> float:
    """
    Вычисляет Δ(T) (скаляр) по формуле (Delta:main).
    Параметры: a1,b1,a2,b2,h,z01,z02
    """
    a1, b1 = params['a1'], params['b1']
    a2, b2 = params['a2'], params['b2']
    h = params['h']
    z01, z02 = params['z01'], params['z02']

    S1 = S_i_recurrent(T, a1, b1, h, z01)
    S2 = S_i_recurrent(T, a2, b2, h, z02)
    J1 = J_i_recurrent(T, a1, b1, h)
    J2 = J_i_recurrent(T, a2, b2, h)
    return (S1 - S2) + (b1 * z01 * J1 - b2 * z02 * J2)


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
    # Функция для одного i
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
    # Проверка в узлах t = k*h
    for k in range(n + 1):
        t = k * h
        phi1 = phi_recurrent(a1, b1, h, t)
        phi2 = phi_recurrent(a2, b2, h, t)
        if phi1 * alpha < phi2 * beta:
            return False
    # Проверка в серединах интервалов t = (k+0.5)*h
    for k in range(n):
        t = (k + 0.5) * h
        phi1 = phi_recurrent(a1, b1, h, t)
        phi2 = phi_recurrent(a2, b2, h, t)
        if phi1 * alpha < phi2 * beta:
            return False
    # Конечная точка, если T_max не кратна h
    if abs(T_max - n * h) > 1e-12:
        phi1 = phi_recurrent(a1, b1, h, T_max)
        phi2 = phi_recurrent(a2, b2, h, T_max)
        if phi1 * alpha < phi2 * beta:
            return False
    return True


# ------------------------------------------------------------
# Основной алгоритм поиска T0 (рекуррентный)
# ------------------------------------------------------------
def find_T0_recurrent(params: dict, eps: float = 1e-6, nmax: int = 1000) -> Optional[float]:
    """
    Реализует алгоритм \ref{alg:recurrent_T0} поиска минимального времени T0.
    Возвращает T0 или None (если неразрешимо).
    """
    # Тривиальный случай
    delta0 = params['z01'] - params['z02']
    if abs(delta0) <= eps:
        return 0.0

    # Проверка условия 1 на [0, nmax*h]
    if not check_condition1_recurrent(params, nmax * params['h']):
        print("Решение не найдено: поточечное условие не выполнено")
        return None

    n = 1
    while n <= nmax:
        T = n * params['h']
        Delta_val = compute_Delta_recurrent(T, params)
        R_val = compute_R_recurrent(T, params)
        if abs(Delta_val) <= R_val:
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
        Delta_c = compute_Delta_recurrent(c, params)
        R_c = compute_R_recurrent(c, params)
        if abs(Delta_c) <= R_c:
            b = c
        else:
            a = c
    return (a + b) / 2.0


# ------------------------------------------------------------
# Пример использования (параметры из статьи)
# ------------------------------------------------------------
if __name__ == "__main__":
    # Параметры из раздела "Примеры вычисления"
    # parameters = {
    #     'a1': 0.0015, 'b1': 0.005,
    #     'a2': 0.0008, 'b2': 0.002,
    #     'h': 24.0,
    #     'alpha': 0.22, 'beta': 0.08,
    #     'z01': 5.0, 'z02': 100.0
    # }

    parameters = {
        'a1': 0.1, 'b1': 0.6,
        'a2': 0.05, 'b2': 0.4,
        'h': 0.2,
        'alpha': 10, 'beta': 7,
        'z01': 0, 'z02': 100.0
    }


    T0 = find_T0_recurrent(parameters, eps=1e-3, nmax=500)
    if T0 is not None:
        print(f"Оптимальное время преследования (рекуррентный метод): T0 = {T0:.6f}")
    else:
        print("Задача неразрешима за разумное время.")