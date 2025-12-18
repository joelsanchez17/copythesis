from sympy import *
q_1 = Symbol('q_1')
q_0 = Symbol('q_0')
x0 = cos(q_0)
x1 = cos(q_1)
x2 = 1.0*x1
x3 = sin(q_1)
x4 = 1.0*x3
x5 = 1.0*sin(q_0)
e = MutableDenseMatrix([[x0*x2, -x0*x4, -x5, Float('-0.62', precision=53)], [x1*x5, -x3*x5, x0, 0], [-x4, -x2, 0, Float('0.35800000000000004', precision=53)], [0, 0, 0, Float('1.0', precision=53)]])