from sympy import *
q_0 = Symbol('q_0')
q_1 = Symbol('q_1')
q_2 = Symbol('q_2')
x0 = sin(q_2)
x1 = sin(q_0)
x2 = 1.0*x1
x3 = x0*x2
x4 = cos(q_1)
x5 = cos(q_2)
x6 = cos(q_0)
x7 = 1.0*x6
x8 = x5*x7
x9 = x2*x5
x10 = x0*x7
x11 = sin(q_1)
x12 = x11*x6
x13 = 1.0*x11
e = MutableDenseMatrix([[x3 - x4*x8, x10*x4 + x9, -Float('1.0', precision=53)*x12, Float('0.53500000000000003', precision=53) - Float('0.316', precision=53)*x12], [-x10 - x4*x9, x3*x4 - x8, -x11*x2, -Float('0.316', precision=53)*x1*x11], [-x13*x5, x0*x13, x4, Float('0.316', precision=53)*x4 + Float('0.35800000000000004', precision=53)], [0, 0, 0, Float('1.0', precision=53)]])