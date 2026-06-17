#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль реализует базовый (эталонный) алгоритм вычисления оптимального времени T0
для трёхмерного пространства E = R^3 с использованием численного интегрирования
(scipy.integrate.quad). Служит для верификации явных и рекуррентных методов.

Основан на basic_T0_solver_1D.py, адаптирован для векторных начальных состояний.
"""

import math
import numpy as np
from scipy.integrate import quad
from scipy.special import gammaln
from functools import lru_cache
from typing import List, Optional, Tuple

# ----------------------------------------------------------------------
# Вспомогательные векторные операции
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# 1. Фундаментальные функции φ_i(t) с логарифмическим вычислением
#    и кэшированием для ускорения многократных вызовов (quad)
# ----------------------------------------------------------------------
@lru_cache(maxsize=100000)
def phi(t: float, a: float, b: float, h: float) -> float:
    """
    Вычисляет φ(t) по формуле:
    φ(t) = Σ_{k=0}^{⌊t/h⌋} b^k * (t - k*h)^k / k! * exp(a*(t - k*h))
    Использует логарифмический метод для избежания переполнения.
    """
    if t < 0:
        return 0.0
    n = int(math.floor(t / h))
    total = 0.0
    for k in range(n + 1):
        tau = t - k * h
        if k == 0:
            term = math.exp(a * tau)
        elif tau <= 0.0:
            term = 0.0
        else:
            log_term = k * math.log(b) + k * math.log(tau) - gammaln(k + 1) + a * tau
            term = math.exp(log_term)
        total += term
    return total

# ----------------------------------------------------------------------
# 2. Интегралы I(T) = ∫_0^T φ(s) ds
#    Разбиваем на отрезки [0, h], [h, 2h], ...
# ----------------------------------------------------------------------
def integral_phi(T: float, a: float, b: float, h: float, eps: float = 1e-12) -> float:
    if T <= 0:
        return 0.0
    n = int(math.floor(T / h))
    total = 0.0
    # Полные интервалы
    for i in range(n):
        left = i * h
        right = (i + 1) * h
        val, _ = quad(lambda s: phi(s, a, b, h), left, right, epsabs=eps, epsrel=eps)
        total += val
    # Последний неполный интервал
    if T > n * h:
        val, _ = quad(lambda s: phi(s, a, b, h), n * h, T, epsabs=eps, epsrel=eps)
        total += val
    return total

# ----------------------------------------------------------------------
# 3. J(T) = ∫_{T-h}^{T} φ(s) ds
# ----------------------------------------------------------------------
def J_integral(T: float, a: float, b: float, h: float, eps: float = 1e-12) -> float:
    lower = max(0.0, T - h)
    upper = T
    if lower >= upper:
        return 0.0
    total = 0.0
    i_start = int(math.floor(lower / h)) + 1
    i_end = int(math.floor(upper / h))
    if lower < i_start * h:
        val, _ = quad(lambda s: phi(s, a, b, h), lower, i_start * h, epsabs=eps, epsrel=eps)
        total += val
    for i in range(i_start, i_end):
        val, _ = quad(lambda s: phi(s, a, b, h), i * h, (i + 1) * h, epsabs=eps, epsrel=eps)
        total += val
    if upper > i_end * h:
        val, _ = quad(lambda s: phi(s, a, b, h), i_end * h, upper, epsabs=eps, epsrel=eps)
        total += val
    return total

# ----------------------------------------------------------------------
# 4. Вспомогательная скалярная сумма S_i(T) (без умножения на z0)
# ----------------------------------------------------------------------
def S_scalar(T: float, a: float, b: float, h: float) -> float:
    """
    Вычисляет S_i(T) = Σ_{k=0}^{⌊T/h⌋} b^k * (T - kh)^k / k! * exp(a*(T - kh))
    """
    n = int(math.floor(T / h))
    total = 0.0
    for k in range(n + 1):
        tau = T - k * h
        if k == 0:
            coeff = 1.0
        else:
            coeff = (tau ** k) / math.factorial(k)
        total += coeff * (b ** k) * math.exp(a * tau)
    return total

# ----------------------------------------------------------------------
# 5. Вектор Δ(T) и его норма
# ----------------------------------------------------------------------
def Delta_vector(T: float, x0: List[float], y0: List[float],
                 a1: float, b1: float, a2: float, b2: float, h: float) -> List[float]:
    """
    Вычисляет вектор Δ(T) ∈ R^3 по формуле (Delta:main).
    """
    J1 = J_integral(T, a1, b1, h)
    J2 = J_integral(T, a2, b2, h)
    # Интегральная часть: b1*J1*x0 - b2*J2*y0
    term_integral = vec_sub(scalar_mul(b1 * J1, x0), scalar_mul(b2 * J2, y0))

    # Суммарная часть: Σ (S1_coeff*x0 - S2_coeff*y0)
    n = int(math.floor(T / h))
    term_sum = [0.0, 0.0, 0.0]
    for k in range(n + 1):
        tau = T - k * h
        if k == 0:
            coeff = 1.0
        else:
            coeff = (tau ** k) / math.factorial(k)
        c1 = coeff * (b1 ** k) * math.exp(a1 * tau)
        c2 = coeff * (b2 ** k) * math.exp(a2 * tau)
        term_sum = vec_add(term_sum, vec_sub(scalar_mul(c1, x0), scalar_mul(c2, y0)))
    return vec_add(term_integral, term_sum)

def Delta_norm(T: float, x0: List[float], y0: List[float],
               a1: float, b1: float, a2: float, b2: float, h: float) -> float:
    """Норма вектора Δ(T)."""
    return norm(Delta_vector(T, x0, y0, a1, b1, a2, b2, h))

# ----------------------------------------------------------------------
# 6. R(T) – скалярный радиус
# ----------------------------------------------------------------------
def R(T: float, alpha: float, beta: float,
      a1: float, b1: float, a2: float, b2: float, h: float) -> float:
    I1 = integral_phi(T, a1, b1, h)
    I2 = integral_phi(T, a2, b2, h)
    return alpha * I1 - beta * I2

# ----------------------------------------------------------------------
# 7. Поиск T_crit – момента нарушения поточечного условия
# ----------------------------------------------------------------------
def find_Tcrit(alpha: float, beta: float,
               a1: float, b1: float, a2: float, b2: float, h: float,
               T_max: float = 500.0, dt: Optional[float] = None) -> float:
    if dt is None:
        dt = h / 100.0
    t = 0.0
    while t <= T_max:
        phi1 = phi(t, a1, b1, h)
        phi2 = phi(t, a2, b2, h)
        if alpha * phi1 < beta * phi2 - 1e-15:
            return t
        t += dt
    return float('inf')

# ----------------------------------------------------------------------
# 8. Основной алгоритм поиска T0 (3D)
# ----------------------------------------------------------------------
def find_T0(x0: List[float], y0: List[float],
            alpha: float, beta: float,
            a1: float, b1: float, a2: float, b2: float, h: float,
            T_max: float = 500.0, eps: float = 1e-6, verbose: bool = True) -> Optional[float]:
    """
    Поиск минимального времени T0 для трёхмерного случая.
    Возвращает T0 или None, если не найден.
    """
    # Тривиальный случай: начальные положения совпадают
    diff = vec_sub(x0, y0)
    if norm(diff) < eps:
        if verbose:
            print("Начальные положения совпадают: T0 = 0")
        return 0.0

    # Поиск T_crit
    T_crit = find_Tcrit(alpha, beta, a1, b1, a2, b2, h, T_max)
    if verbose:
        if np.isfinite(T_crit):
            print(f"T_crit = {T_crit:.6f} (нарушение поточечного условия)")
        else:
            print("Поточечное условие выполнено на всём [0, T_max]")

    T_low = 0.0
    T_high = None
    m = 1
    while True:
        T = m * h
        if T > T_max:
            if verbose:
                print(f"Достигнут T_max = {T_max}, преследование не найдено")
            return None
        if np.isfinite(T_crit) and T > T_crit:
            if verbose:
                print(f"Превышен T_crit = {T_crit}, дальнейший поиск невозможен")
            return None

        delta_norm_val = Delta_norm(T, x0, y0, a1, b1, a2, b2, h)
        R_val = R(T, alpha, beta, a1, b1, a2, b2, h)
        f_val = delta_norm_val - R_val

        if verbose:
            print(f"T = {T:.2f}, ||Δ|| = {delta_norm_val:.6e}, R = {R_val:.6e}, f = {f_val:.6e}")

        if f_val <= 0:
            T_high = T
            break
        T_low = T
        m += 1

    if T_high is None:
        return None

    if verbose:
        print(f"Найден интервал: [{T_low:.6f}, {T_high:.6f}]")

    # Бисекция
    while T_high - T_low > eps:
        T_mid = (T_low + T_high) / 2.0
        delta_mid = Delta_norm(T_mid, x0, y0, a1, b1, a2, b2, h)
        R_mid = R(T_mid, alpha, beta, a1, b1, a2, b2, h)
        f_mid = delta_mid - R_mid
        if f_mid > 0:
            T_low = T_mid
        else:
            T_high = T_mid

    T0 = (T_low + T_high) / 2.0
    if verbose:
        print(f"T0 = {T0:.8f}")
    return T0

# ----------------------------------------------------------------------
# Запуск расчёта (пример из статьи для 3D)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Параметры трёхмерного эксперимента (раздел 10.3)
    # parameters_3d = {
    #     'a1': 0.10, 'b1': 0.40,
    #     'a2': 0.05, 'b2': 0.20,
    #     'h': 0.3,
    #     'alpha': 15.0, 'beta': 8.0,
    #     'z01': [0.0, 0.0, 0.0],
    #     'z02': [100.0, 80.0, 60.0]
    # }

    parameters_3d = {
        'a1': 0.0015,
        'b1': 0.005,
        'a2': 0.0008,
        'b2': 0.002,
        'h': 24.0,
        'alpha': 0.22,
        'beta': 0.08,
        'z01': [5.0, 0.0, 0.0],
        'z02': [100.0, 0.0, 0.0]
    }    

    T0 = find_T0(
        x0=parameters_3d['z01'],
        y0=parameters_3d['z02'],
        alpha=parameters_3d['alpha'],
        beta=parameters_3d['beta'],
        a1=parameters_3d['a1'],
        b1=parameters_3d['b1'],
        a2=parameters_3d['a2'],
        b2=parameters_3d['b2'],
        h=parameters_3d['h'],
        T_max=500.0,          # достаточно для поиска T0 (можно увеличить)
        eps=1e-6,
        verbose=True
    )
    if T0 is not None:
        print(f"\nРезультат: минимальное время преследования T0 = {T0:.6f}")
        # Сравнение с классическим временем
        delta_norm = norm(parameters_3d['z02'])
        alpha_beta = parameters_3d['alpha'] - parameters_3d['beta']
        T_classic = delta_norm / alpha_beta
        print(f"Классическое время без запаздывания: T_classic = {T_classic:.6f}")
    else:
        print("\nПреследование не гарантировано в пределах T_max")