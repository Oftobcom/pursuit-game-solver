"""
Модуль реализует алгоритм вычисления оптимального времени T0 по явным формулам
из раздела "Явные формулы для вычисления интегралов J_i(T)".
Предполагается одномерный случай (E = R) для простоты; для многомерного случая
требуется задавать нормы и скалярное произведение.
"""

import matplotlib.pyplot as plt
import numpy as np
import math
from typing import Callable, Tuple, Optional

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
            # при k=0: b^0=1, delta^0=1, 0!=1
            try:
                term = (b ** k) * (delta ** k) / math.factorial(k) * math.exp(a * delta)
            except OverflowError:
                # переключение на логарифмическую форму при больших значениях
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
    Используются аналитические формулы (2) и (3) из статьи.
    """
    if a > 0:
        if L == 0.0:
            # формула (3): J(0,U,k,a) = e^{aU} / a * Q_k(U) - (-1)^k / a^{k+1}
            return (math.exp(a * U) / a) * Qk(U, a, k) - ((-1) ** k) / (a ** (k + 1))
        else:
            # формула (2): J(L,U,k,a) = e^{aU}/a * Q_k(U) - e^{aL}/a * Q_k(L)
            return (math.exp(a * U) / a) * Qk(U, a, k) - (math.exp(a * L) / a) * Qk(L, a, k)
    elif a == 0:
        # степенной случай
        return (U ** (k + 1) / math.factorial(k + 1)) - (L ** (k + 1) / math.factorial(k + 1))
    else:
        raise ValueError("a должна быть неотрицательной")


def J_i(T: float, a: float, b: float, h: float) -> float:
    """
    Вычисляет J_i(T) = ∫_{T-h}^{T} ϕ_i(t) dt по явным формулам (6) или (7).
    """
    if h <= 0:
        raise ValueError("h > 0")
    n_full = int((T - h) // h) if T - h >= 0 else -1   # ⌊(T-h)/h⌋
    n_total = int(T // h)                               # ⌊T/h⌋
    total = 0.0
    if a > 0:
        # первая сумма: k = 0 … floor((T-h)/h)
        for k in range(0, n_full + 1):
            arg1 = T - k * h
            arg2 = T - h - k * h
            term1 = math.exp(a * arg1) / a * Qk(arg1, a, k)
            term2 = math.exp(a * arg2) / a * Qk(arg2, a, k)
            total += (b ** k) * (term1 - term2)
        # второе слагаемое
        arg = T - n_total * h
        term = (math.exp(a * arg) / a) * Qk(arg, a, n_total) - ((-1) ** n_total) / (a ** (n_total + 1))
        total += (b ** n_total) * term
    else:  # a == 0
        for k in range(0, n_full + 1):
            delta1 = T - k * h
            delta2 = T - h - k * h
            total += (b ** k) * ((delta1 ** (k + 1) - delta2 ** (k + 1)) / math.factorial(k + 1))

        delta = T - n_total * h
        total += (b ** n_total) * (delta ** (n_total + 1) / math.factorial(n_total + 1))
    return total


def compute_S(T: float, a: float, b: float, h: float, z0: float) -> float:
    """
    Вычисляет S_i(T) = Σ_{k=0}^{⌊T/h⌋} b^k * (T - kh)^k / k! * exp(a*(T - kh)) * z0
    для скалярного случая.
    """
    n = int(T // h)
    total = 0.0
    for k in range(0, n + 1):
        delta = T - k * h
        if delta == 0.0 and k > 0:
            term = 0.0
        else:
            term = (b ** k) * (delta ** k) / math.factorial(k) * math.exp(a * delta) * z0
        total += term
    return total


def compute_Delta(T: float, params: dict) -> float:
    """
    Вычисляет Δ(T) (скаляр) согласно формуле (Delta:main).
    params должен содержать ключи:
        a1, b1, a2, b2, h, z01, z02
    """
    a1 = params['a1']
    b1 = params['b1']
    a2 = params['a2']
    b2 = params['b2']
    h = params['h']
    z01 = params['z01']
    z02 = params['z02']
    S1 = compute_S(T, a1, b1, h, z01)
    S2 = compute_S(T, a2, b2, h, z02)
    J1 = J_i(T, a1, b1, h)
    J2 = J_i(T, a2, b2, h)
    return (S1 - S2) + (b1 * z01 * J1 - b2 * z02 * J2)


def compute_R(T: float, params: dict) -> float:
    """
    Вычисляет R(T) = ∫_0^T (α*ϕ1(s) - β*ϕ2(s)) ds
    (скалярный радиус).
    """
    alpha = params['alpha']
    beta = params['beta']
    # Используем формулу через интегралы от ϕ_i
    # По лемме int_phi_J0: ∫ϕ_i(s) ds = Σ_{k=0}^{⌊T/h⌋} b_i^k * J(0, T-kh, k, a_i)
    def int_phi(a, b, h, T):
        n = int(T // h)
        total = 0.0
        for k in range(n + 1):
            U = T - k * h
            total += (b ** k) * J_integral_aux(0.0, U, k, a)
        return total
    int_phi1 = int_phi(params['a1'], params['b1'], params['h'], T)
    int_phi2 = int_phi(params['a2'], params['b2'], params['h'], T)
    return alpha * int_phi1 - beta * int_phi2


def check_condition1(params: dict, T: float) -> bool:
    """
    Проверка поточечного условия 1 на отрезке [0, T].
    Используется дискретизация с шагом h/2.
    """
    h = params['h']
    a1, b1 = params['a1'], params['b1']
    a2, b2 = params['a2'], params['b2']
    alpha, beta = params['alpha'], params['beta']
    n = int(T // h)
    # узлы
    for k in range(n + 1):
        t = k * h
        phi1 = compute_phi(a1, b1, h, t)
        phi2 = compute_phi(a2, b2, h, t)
        if phi1 * alpha < phi2 * beta:
            return False
    # середины интервалов
    for k in range(n):
        t = (k + 0.5) * h
        phi1 = compute_phi(a1, b1, h, t)
        phi2 = compute_phi(a2, b2, h, t)
        if phi1 * alpha < phi2 * beta:
            return False
    # конечная точка если не кратна
    if abs(T - n * h) > 1e-12:
        phi1 = compute_phi(a1, b1, h, T)
        phi2 = compute_phi(a2, b2, h, T)
        if phi1 * alpha < phi2 * beta:
            return False
    return True


def find_T0(params: dict, eps: float = 1e-6, nmax: int = 1000) -> Optional[float]:
    """
    Основной алгоритм поиска минимального времени T0 по явным формулам.
    Параметры передаются словарём:
        a1, b1, a2, b2, h, alpha, beta, z01, z02
    Возвращает T0 либо None (сообщение о неразрешимости).
    """
    # Проверка тривиального случая
    delta0 = params['z01'] - params['z02']
    if abs(delta0) <= eps:
        return 0.0

    # Проверка условия 1 на [0, nmax*h]
    if not check_condition1(params, nmax * params['h']):
        print("Решение не найдено: поточечное условие не выполнено")
        return None

    # Поиск интервала [a, b] перебором T = n*h
    n = 1
    while n <= nmax:
        T = n * params['h']
        Delta_val = compute_Delta(T, params)
        R_val = compute_R(T, params)
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
        Delta_c = compute_Delta(c, params)
        R_c = compute_R(c, params)
        if abs(Delta_c) <= R_c:
            b = c
        else:
            a = c
    return (a + b) / 2.0

def plot_Delta_R(params, T_max=None, n_points=500, T0=None, save_path=None):
    """
    Строит графики |Δ(T)| и R(T) в зависимости от T.
    
    Параметры:
        params: словарь с параметрами модели
        T_max: максимальное время (если None, то T_max = nmax*h из params или 100)
        n_points: количество точек для дискретизации
        T0: опциональное найденное время (будет отмечено вертикальной линией)
        save_path: путь для сохранения графика (если None, показывается интерактивно)
    """
    h = params['h']
    if T_max is None:
        # Используем nmax из внешнего контекста? По умолчанию возьмём 500*h
        nmax = params.get('nmax', 500)
        T_max = nmax * h
    T_vals = np.linspace(0, T_max, n_points)
    Delta_abs = np.zeros_like(T_vals)
    R_vals = np.zeros_like(T_vals)
    
    for i, T in enumerate(T_vals):
        # Используем существующие функции compute_Delta и compute_R
        Delta_abs[i] = abs(compute_Delta(T, params))
        R_vals[i] = compute_R(T, params)
    
    plt.figure(figsize=(10, 6))
    plt.plot(T_vals, Delta_abs, label=r'$|\Delta(T)|$', linewidth=2)
    plt.plot(T_vals, R_vals, label=r'$R(T)$', linewidth=2)
    if T0 is not None and 0 <= T0 <= T_max:
        plt.axvline(x=T0, color='red', linestyle='--', label=f'$T_0 = {T0:.4f}$')
    plt.xlabel('Время T')
    plt.ylabel('Значение')
    # plt.title('Зависимость $|\Delta(T)|$ и $R(T)$')
    plt.title(r'Зависимость $|\Delta(T)|$ и $R(T)$')   # raw-строка
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"График сохранён в {save_path}")
    else:
        plt.show()

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

    # parameters = {
    #     'a1': 0.10,
    #     'b1': 0.40,
    #     'a2': 0.05,
    #     'b2': 0.20,
    #     'h': 5,
    #     'alpha': 15.0,
    #     'beta': 8.0,
    #     'z01': 5.0,
    #     'z02': 100.0
    # }

    parameters = {
        'a1': 0.0015,
        'b1': 0.005,
        'a2': 0.0008,
        'b2': 0.002,
        'h': 24.0,
        'alpha': 0.22,
        'beta': 0.08,
        'z01': 5.0,
        'z02': 100.0
    }

    results = {
        'h': [],
        'T0': [],
        'success': []
    }

    h = 20
    step = 2
    for i in range(1, 22):
        parameters['h'] = h  # можно менять для тестирования разных h
        T0 = find_T0(parameters, eps=1e-6, nmax=500)
        if T0 is not None:
            print(f"Оптимальное время преследования (явный метод): T0 = {T0:.6f}")
            results['h'].append(h)
            results['T0'].append(T0)
            results['success'].append(True)
        else:
            print("Задача неразрешима за разумное время.")
        h += step

    # ----------------------------------------------------------------------
    # Построение графика
    # ----------------------------------------------------------------------
    plt.figure(figsize=(10, 6))

    h_arr = np.array(results['h'])
    mask_success = np.array(results['success'])

    # Точки, где решение найдено
    if np.any(mask_success):
        h_success = h_arr[mask_success]
        T_success = np.array([results['T0'][i] for i in range(len(h_arr)) if mask_success[i]])
        plt.plot(h_success, T_success, 'o-', label='Рекуррентный метод', color='red', linewidth=2, markersize=8)

        # Подписи значений у точек
        for hi, ti in zip(h_success, T_success):
            plt.annotate(f'{ti:.3f}', (hi, ti), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

    # Точки, где решение не найдено – закрашиваем область
    mask_fail = ~mask_success
    if np.any(mask_fail):
        for hf in h_arr[mask_fail]:
            plt.axvspan(hf - step/2, hf + step/2, alpha=0.2, color='gray')
        # Подпись в первом таком интервале
        first_fail = h_arr[mask_fail][0]
        ylim = plt.ylim()
        plt.text(first_fail, ylim[1]*0.9, 'Неразрешимость\n(условие 1)', ha='center', color='black', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    plt.xlabel('Запаздывание $h$', fontsize=14)
    plt.ylabel('Минимальное время преследования $T_0$', fontsize=14)
    plt.title('Зависимость $T_0(h)$ (рекуррентный алгоритм, v02)', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)

    # Информация о параметрах
    params_text = (f"$a_1={parameters['a1']}, b_1={parameters['b1']}, "
                f"a_2={parameters['a2']}, b_2={parameters['b2']}, "
                f"\\alpha={parameters['alpha']}, \\beta={parameters['beta']}, "
                f"z_{{01}}={parameters['z01']}, z_{{02}}={parameters['z02']}$")
    plt.figtext(0.5, 0.01, params_text, ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig('T0_vs_h_recurrent_v02.png', dpi=150)
    plt.show()

    # ----------------------------------------------------------------------
    # Вывод таблицы результатов
    # ----------------------------------------------------------------------
    print("\nТаблица результатов (рекуррентный метод, v02):")
    print(" h       T0")
    print("--------------")
    for i, h in enumerate(results['h']):
        T0 = results['T0'][i]
        status = f"{T0:.6f}" if T0 is not None else "None"
        print(f"{h:5.2f}   {status}")        