#recurrent_T0_solver_1D_neutral.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Рекуррентный алгоритм для нейтральной системы (1D).
Основан на леммах 5.1, 5.4, 5.5 из one_7.
"""

import math
from typing import Optional

# ----------------------------------------------------------------------
# Рекуррентное вычисление φ(t) (лемма 5.1)
# ----------------------------------------------------------------------
def phi_recurrent_neutral(t, a, b, h):
    """
    Вычисляет φ(t) рекуррентно по k и j.
    Использует соотношения:
    c_{k,1} = a^{k-1} b (t - k h)
    c_{k,j+1} = c_{k,j} * (k-j)/j * (b/a) * (t - k h)/(j+1)   (a>0)
    При a=0: только j=k.
    """
    if t < 0:
        return 0.0
    n = int(math.floor(t / h))
    total = 1.0
    if n == 0:
        return total
    # Для каждого k
    for k in range(1, n + 1):
        tau = t - k * h
        if tau == 0.0:
            continue
        if a > 0:
            # Начальное значение для j=1
            c = (a ** (k - 1)) * b * tau
            total += c
            # Переход по j
            for j in range(1, k):
                c = c * (k - j) / j * (b / a) * tau / (j + 1)
                total += c
        else:  # a == 0
            # только j = k
            term = (b ** k) * (tau ** k) / math.factorial(k)
            total += term
    return total

# ----------------------------------------------------------------------
# Рекуррентное вычисление J(0, U, k, a) (вспомогательное)
# Для a=0: J = U^{k+1}/(k+1)!
# Для a>0: J(0,U,0,a) = (e^{aU}-1)/a, но в нейтральном случае a=0 всегда?
# В нейтральном случае a может быть >0, но φ не содержит экспонент. Однако интегралы полиномов
# не требуют экспонент. В нашем случае a входит только в степени, но не в экспоненту.
# Поэтому интеграл J(0,U,k,a) для полинома (без экспоненты) просто U^{k+1}/(k+1)!.
# Это не зависит от a. Однако в общем случае a>0, но формула для φ не содержит e^{a t}.
# Значит a используется только в коэффициентах a^{k-j}. Интеграл от (t - kh)^j/j! не зависит от a.
# Поэтому вспомогательная функция не нужна, J_i(T) можно вычислять прямо по рекуррентным формулам для d_{k,j} и e_{n2,j}.
# Реализуем J_i рекуррентно по леммам 5.4 и 5.5.
# ----------------------------------------------------------------------

def J_i_recurrent_neutral(T, a, b, h):
    """
    Вычисляет J_i(T) рекуррентно, используя d_{k,j} и e_{n2,j}.
    """
    if T < 0:
        return 0.0
    n1 = int(math.floor((T - h) / h)) if T - h >= 0 else -1
    n2 = int(math.floor(T / h))
    total = ell(T, h)   # ℓ(T)

    # Первая сумма: k = 1..n1
    if n1 >= 1:
        for k in range(1, n1 + 1):
            p = T - k * h
            q = T - h - k * h
            # Начальное значение для j=1
            if a > 0:
                d = (a ** (k - 1)) * b * (p**2 - q**2) / 2.0
            else:
                # a=0: только j=k
                if k == 1:
                    d = b * (p**2 - q**2) / 2.0
                else:
                    d = 0.0
            total += d
            # Переход по j
            for j in range(1, k):
                # Вычисляем d_{k,j+1}
                if a > 0:
                    d = d * (k - j) / j * (b / a) * (p**(j+2) - q**(j+2)) / ((j+2) * (p**(j+1) - q**(j+1)))
                else:
                    # a=0: только j=k, остальные нули
                    break
                total += d

    # Вторая сумма: j = 1..n2
    if n2 >= 1:
        tau = T - n2 * h
        if tau > 0:
            if a > 0:
                e = (a ** (n2 - 1)) * b * (tau**2) / 2.0
            else:
                # a=0: только j=n2
                if n2 == 1:
                    e = b * (tau**2) / 2.0
                else:
                    e = 0.0
            total += e
            for j in range(1, n2):
                if a > 0:
                    e = e * (n2 - j) / j * (b / a) * tau / (j + 2)
                else:
                    break
                total += e
    return total

def ell(T, h):
    return T if T < h else h if T >= 0 else 0.0

# ----------------------------------------------------------------------
# Вычисление Δ(T) и R(T) рекуррентно
# ----------------------------------------------------------------------
def Delta_recurrent(T, x0, y0, a1, b1, a2, b2, h):
    phi1_T = phi_recurrent_neutral(T, a1, b1, h)
    phi1_Th = phi_recurrent_neutral(T - h, a1, b1, h) if T >= h else 0.0
    phi2_T = phi_recurrent_neutral(T, a2, b2, h)
    phi2_Th = phi_recurrent_neutral(T - h, a2, b2, h) if T >= h else 0.0
    J1 = J_i_recurrent_neutral(T, a1, b1, h)
    J2 = J_i_recurrent_neutral(T, a2, b2, h)
    return (phi1_T - a1 * phi1_Th) * x0 - (phi2_T - a2 * phi2_Th) * y0 + b1 * x0 * J1 - b2 * y0 * J2

def R_recurrent(T, alpha, beta, a1, b1, a2, b2, h):
    # Используем накопление I_i(nh) = Σ_{m=0}^{n-1} J_i((m+1)h)
    if T <= 0:
        return 0.0
    n = int(math.floor(T / h))
    I1 = 0.0
    I2 = 0.0
    for m in range(n):
        I1 += J_i_recurrent_neutral((m + 1) * h, a1, b1, h)
        I2 += J_i_recurrent_neutral((m + 1) * h, a2, b2, h)
    # Остаток [n h, T] — интегрируем аналитически (полином)
    if T > n * h:
        # Интеграл от φ на [n h, T] можно вычислить по той же схеме, что и J_i, но с пределами n h и T.
        # Для простоты используем рекурсивно J_i, но сдвинув начало координат.
        # Можно применить формулу для интеграла от φ на произвольном отрезке [L, U] длиной ≤ h.
        # Здесь для краткости применим численное интегрирование (или аналитическое).
        # Поскольку мы в рекуррентном методе, лучше использовать аналитику.
        # Воспользуемся явным интегрированием полинома на [n h, T].
        # Для этого проинтегрируем φ(s) на [n h, T] (длина < h).
        # φ(s) = 1 + Σ_{k=1}^{n} Σ_{j=1}^{k} binom(k-1,j-1) a^{k-j} b^j (s - k h)^j / j!
        # Интеграл от 1 даёт (T - n h). Остальные члены интегрируются.
        delta_T = T - n * h
        I1 += delta_T
        I2 += delta_T
        # Добавляем полиномиальные члены для k=1..n
        for k in range(1, n + 1):
            # Интеграл на [n h, T] от (s - k h)^j
            # пределы: s от n h до T, но (s - k h) может быть отрицательным? Нет, т.к. s >= n h >= k h? 
            # На самом деле k ≤ n, поэтому s - k h >= 0.
            for j in range(1, k + 1):
                coeff = math.comb(k - 1, j - 1) * (a1 ** (k - j)) * (b1 ** j) / math.factorial(j)
                # ∫_{n h}^{T} (s - k h)^j ds = [ (T - k h)^{j+1} - (n h - k h)^{j+1} ] / (j+1)
                term1 = coeff * ((T - k * h) ** (j + 1) - (n * h - k * h) ** (j + 1)) / (j + 1)
                I1 += term1
                coeff2 = math.comb(k - 1, j - 1) * (a2 ** (k - j)) * (b2 ** j) / math.factorial(j)
                term2 = coeff2 * ((T - k * h) ** (j + 1) - (n * h - k * h) ** (j + 1)) / (j + 1)
                I2 += term2
    return alpha * I1 - beta * I2

# ----------------------------------------------------------------------
# Поиск T0 (рекуррентный)
# ----------------------------------------------------------------------
def find_T0_recurrent(params, eps=1e-6, nmax=1000):
    x0, y0 = params['z01'], params['z02']
    alpha, beta = params['alpha'], params['beta']
    a1, b1 = params['a1'], params['b1']
    a2, b2 = params['a2'], params['b2']
    h = params['h']

    if abs(x0 - y0) < eps:
        return 0.0

    n = 1
    while n <= nmax:
        T = n * h
        delta_val = Delta_recurrent(T, x0, y0, a1, b1, a2, b2, h)
        R_val = R_recurrent(T, alpha, beta, a1, b1, a2, b2, h)
        if abs(delta_val) <= R_val:
            a = (n - 1) * h
            b = n * h
            break
        n += 1
    else:
        print(f"Неразрешимо: достигнут T_max = {nmax * h}")
        return None

    while b - a > eps:
        c = (a + b) / 2.0
        delta_c = Delta_recurrent(c, x0, y0, a1, b1, a2, b2, h)
        R_c = R_recurrent(c, alpha, beta, a1, b1, a2, b2, h)
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
    T0 = find_T0_recurrent(params, eps=1e-6, nmax=500)
    if T0 is not None:
        print(f"Рекуррентный метод: T0 = {T0:.6f}")
    else:
        print("Не найдено")