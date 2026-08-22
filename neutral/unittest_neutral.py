#unittest_neutral.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Сравнение явного и рекуррентного методов для нейтральной системы (1D).
"""

import sys
from explicit_T0_solver_1D_neutral import phi_neutral_explicit, J_i_explicit, find_T0_explicit, Delta_explicit, R_explicit
from recurrent_T0_solver_1D_neutral import phi_neutral_recurrent, J_i_recurrent, find_T0_recurrent, Delta_recurrent, R_recurrent
from basic_T0_solver_1D_neutral import phi_neutral, J_integral

# PARAMS = {
#     'a1': 0.15, 'b1': 0.05,
#     'a2': 0.08, 'b2': 0.02,
#     'h': 2.4,
#     'alpha': 0.22, 'beta': 0.08,
#     'z01': 5.0, 'z02': 100.0
# }

PARAMS = {
    'a1': 0.2, 'b1': 0.3,
    'a2': 0.1, 'b2': 0.2,
    'h': 1.5,
    'alpha': 1.0, 'beta': 0.5,
    'z01': 0.0, 'z02': 10.0
}

EPS = 1e-6
NMAX = 500

def compare_T0():
   print("=== Сравнение T0 ===")
   T_rec = find_T0_recurrent(PARAMS, eps=EPS, nmax=NMAX)
   T_exp = find_T0_explicit(PARAMS, eps=EPS, nmax=NMAX)

   print(f"Рекуррентный: {T_rec:.10f}" if T_rec is not None else "Рекуррентный: None")
   print(f"Явный:        {T_exp:.10f}" if T_exp is not None else "Явный: None")

   if T_rec is not None and T_exp is not None:
       diff = abs(T_rec - T_exp)
       rel = diff / max(abs(T_rec), abs(T_exp), 1.0)
       print(f"Абс. разница: {diff:.2e}, Отн. разница: {rel:.2e}")
       if diff < 1e-6:
           print("✓ Совпадают в пределах 1e-6")
       else:
           print("✗ Различаются более чем на 1e-6")

def compare_Delta_R():
   print("\n=== Сравнение Δ и R в точках T = n*h ===")
   for n in range(1, 6):
       T = n * PARAMS['h']
       d_rec = Delta_recurrent(T, PARAMS['z01'], PARAMS['z02'],
                               PARAMS['a1'], PARAMS['b1'],
                               PARAMS['a2'], PARAMS['b2'], PARAMS['h'])
       d_exp = Delta_explicit(T, PARAMS['z01'], PARAMS['z02'],
                              PARAMS['a1'], PARAMS['b1'],
                              PARAMS['a2'], PARAMS['b2'], PARAMS['h'])
       r_rec = R_recurrent(T, PARAMS['alpha'], PARAMS['beta'],
                           PARAMS['a1'], PARAMS['b1'],
                           PARAMS['a2'], PARAMS['b2'], PARAMS['h'])
       r_exp = R_explicit(T, PARAMS['alpha'], PARAMS['beta'],
                          PARAMS['a1'], PARAMS['b1'],
                          PARAMS['a2'], PARAMS['b2'], PARAMS['h'])
       print(f"T={T:5.2f} | Δ_diff={abs(d_rec-d_exp):.3e} | R_diff={abs(r_rec-r_exp):.3e}")

if __name__ == "__main__":
    # T = [0.5, 1.5, 2.5, 3.5]
    # a = 0
    # b = 1
    # h = 1

    # T = [4, 5]
    # a = 0.2
    # b = 0.3
    # h = 1.5
    
    # T = [2.3]
    # a = 0.4
    # b = 0.5
    # h = 1.0
    
    # T = [2.5]
    # a = 0
    # b = 0.5
    # h = 1

    # for t in T:
    #     print("phi:")
    #     print(t, ":", phi_neutral(t, a, b, h), phi_neutral_recurrent(t, a, b, h), phi_neutral_recurrent(t, a, b, h))
    #     print()
    #     print("J_i:")
    #     print(t, ":", J_integral(t, a, b, h), J_i_recurrent(t, a, b, h), J_i_explicit(t, a, b, h))
    #     print()

    compare_T0()
    compare_Delta_R()

