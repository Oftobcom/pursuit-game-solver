import math
import numpy as np
from scipy.integrate import quad
from scipy.special import gammaln
from functools import lru_cache

# ----------------------------------------------------------------------
# Параметры задачи (можно менять)
# ----------------------------------------------------------------------
# parameters = {
#     'a1': 0.10,
#     'b1': 0.40,
#     'a2': 0.05,
#     'b2': 0.20,
#     'h': 0.3,
#     'alpha': 15.0,
#     'beta': 8.0,
#     'z01': 5.0,
#     'z02': 100.0
# }

# parameters = {
#     'a1': 0.0015,
#     'b1': 0.005,
#     'a2': 0.0008,
#     'b2': 0.002,
#     'h': 24.0,
#     'alpha': 0.22,
#     'beta': 0.08,
#     'z01': 5.0,
#     'z02': 100.0
# }

parameters = {
    'a1': 0.10,
    'b1': 0.40,
    'a2': 0.05,
    'b2': 0.20,
    'h': 1,
    'alpha': 15.0,
    'beta': 8.0,
    'z01': 5.0,
    'z02': 100.0
}
# ----------------------------------------------------------------------
# 1. Фундаментальные функции φ_i(t) с логарифмическим вычислением
#    и кэшированием для ускорения многократных вызовов (quad)
# ----------------------------------------------------------------------
@lru_cache(maxsize=100000)
def phi(t, a, b, h):
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
            # τ^0 / 0! = 1, b^0 = 1, exp(a*τ)
            term = math.exp(a * tau)
        elif tau <= 0.0:
            term = 0.0
        else:
            # Используем логарифмическое представление:
            # term = exp( k*log(b) + k*log(tau) - log(k!) + a*tau )
            log_term = k * math.log(b) + k * math.log(tau) - gammaln(k + 1) + a * tau
            term = math.exp(log_term)
        total += term
    return total

# ----------------------------------------------------------------------
# 2. Интегралы I(T) = ∫_0^T φ(s) ds
#    Разбиваем на отрезки [0, h], [h, 2h], ...
# ----------------------------------------------------------------------
def integral_phi(T, a, b, h, eps=1e-12):
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
def J_integral(T, a, b, h, eps=1e-12):
    lower = max(0.0, T - h)
    upper = T
    if lower >= upper:
        return 0.0
    # Разбиение по точкам, кратным h, для точности
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
# 4. Δ(T) и R(T)
# ----------------------------------------------------------------------
def Delta(T, x0, y0, a1, b1, a2, b2, h):
    J1 = J_integral(T, a1, b1, h)
    J2 = J_integral(T, a2, b2, h)
    term_integral = b1 * J1 * x0 - b2 * J2 * y0

    n = int(math.floor(T / h))
    term_sum = 0.0
    for k in range(n + 1):
        tau = T - k * h
        # при k=0 и tau=0: (tau^k)/k! = 1
        if k == 0:
            coeff = 1.0
        else:
            coeff = (tau ** k) / math.factorial(k)
        term_sum += coeff * ((b1 ** k) * math.exp(a1 * tau) * x0 -
                             (b2 ** k) * math.exp(a2 * tau) * y0)
    return term_integral + term_sum

def R(T, alpha, beta, a1, b1, a2, b2, h):
    I1 = integral_phi(T, a1, b1, h)
    I2 = integral_phi(T, a2, b2, h)
    return alpha * I1 - beta * I2

# ----------------------------------------------------------------------
# 5. Поиск T_crit – момента нарушения поточечного условия
# ----------------------------------------------------------------------
def find_Tcrit(alpha, beta, a1, b1, a2, b2, h, T_max=500.0, dt=None):
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
# 6. Основной алгоритм поиска T0
# ----------------------------------------------------------------------
def find_T0(x0, y0, alpha, beta, a1, b1, a2, b2, h,
            T_max=500.0, eps=1e-6, verbose=True):
    if abs(x0 - y0) < eps:
        if verbose:
            print("Начальные положения совпадают: T0 = 0")
        return 0.0

    # T_crit = find_Tcrit(alpha, beta, a1, b1, a2, b2, h, T_max)
    # if verbose:
    #     if np.isfinite(T_crit):
    #         print(f"T_crit = {T_crit:.6f} (нарушение поточечного условия)")
    #     else:
    #         print("Поточечное условие выполнено на всём [0, T_max]")

    T_low = 0.0
    T_high = None
    m = 1
    while True:
        T = m * h
        if T > T_max:
            if verbose:
                print(f"Достигнут T_max = {T_max}, преследование не найдено")
            return None
        # if np.isfinite(T_crit) and T > T_crit:
        #     if verbose:
        #         print(f"Превышен T_crit = {T_crit}, дальнейший поиск невозможен")
        #     return None

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

    # Бисекция
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
# Запуск расчёта
# ----------------------------------------------------------------------
if __name__ == "__main__":    
    # Уменьшаем T_max для демонстрации, при необходимости можно увеличить
    T0 = find_T0(
        x0=parameters['z01'],
        y0=parameters['z02'],
        alpha=parameters['alpha'],
        beta=parameters['beta'],
        a1=parameters['a1'],
        b1=parameters['b1'],
        a2=parameters['a2'],
        b2=parameters['b2'],
        h=parameters['h'],
        T_max=400.0,          # достаточно для поиска T0 (можно увеличить)
        eps=1e-6,
        verbose=True
    )
    if T0 is not None:
        print(f"\nРезультат: минимальное время преследования T0 = {T0:.6f}")
    else:
        print("\nПреследование не гарантировано в пределах T_max")
