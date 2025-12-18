from sympy import *
q_0 = Symbol('q_0')
q_1 = Symbol('q_1')
x0 = cos(q_1)
x1 = 1.0*cos(q_0)
x2 = sin(q_1)
x3 = sin(q_0)
x4 = 1.0*x0
x5 = 1.0*x2
e = MutableDenseMatrix([[-x0*x1, x1*x2, x3, Float('0.53500000000000003', precision=53)], [-x3*x4, x3*x5, -x1, 0], [-x5, -x4, 0, Float('0.35800000000000004', precision=53)], [0, 0, 0, Float('1.0', precision=53)]])