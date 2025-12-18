from sympy import *
q_0 = Symbol('q_0')
x0 = -1.0*cos(q_0)
x1 = sin(q_0)
e = MutableDenseMatrix([[x0, x1, 0, Float('0.53500000000000003', precision=53)], [-Float('1.0', precision=53)*x1, x0, 0, 0], [0, 0, 1, Float('0.35800000000000004', precision=53)], [0, 0, 0, 1]])