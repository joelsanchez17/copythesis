from sympy import *
q_2 = Symbol('q_2')
q_1 = Symbol('q_1')
q_0 = Symbol('q_0')
x0 = sin(q_2)
x1 = sin(q_0)
x2 = 1.0*x1
x3 = x0*x2
x4 = cos(q_0)
x5 = cos(q_1)
x6 = cos(q_2)
x7 = x2*x6
x8 = 1.0*x0*x4
x9 = sin(q_1)
x10 = x4*x9
x11 = 1.0*x9
e = MutableDenseMatrix([[-x3 + Float('1.0', precision=53)*x4*x5*x6, -x5*x8 - x7, Float('1.0', precision=53)*x10, Float('0.316', precision=53)*x10 + Float('-0.62', precision=53)], [x5*x7 + x8, -x3*x5 + Float('1.0', precision=53)*x4*x6, x2*x9, Float('0.316', precision=53)*x1*x9], [-x11*x6, x0*x11, x5, Float('0.316', precision=53)*x5 + Float('0.35800000000000004', precision=53)], [0, 0, 0, 1]])