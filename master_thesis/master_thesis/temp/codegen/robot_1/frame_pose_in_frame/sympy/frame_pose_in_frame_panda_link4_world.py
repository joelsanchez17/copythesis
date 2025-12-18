from sympy import *
q_2 = Symbol('q_2')
q_1 = Symbol('q_1')
q_0 = Symbol('q_0')
q_3 = Symbol('q_3')
x0 = cos(q_3)
x1 = sin(q_2)
x2 = sin(q_0)
x3 = 1.0*x2
x4 = x1*x3
x5 = cos(q_0)
x6 = cos(q_1)
x7 = cos(q_2)
x8 = -1.0*x4 + 1.0*x5*x6*x7
x9 = sin(q_3)
x10 = sin(q_1)
x11 = 1.0*x5
x12 = x3*x7
x13 = x1*x11
x14 = 0.0825*x1
x15 = 1.0*x12*x6 + 1.0*x13
x16 = 0.0825*x7
x17 = 1.0*x10
x18 = x17*x7
e = MutableDenseMatrix([[x0*x8 + x10*x11*x9, Float('1.0', precision=53)*x0*x10*x5 - x8*x9, x12 + x13*x6, Float('0.316', precision=53)*x10*x5 - x14*x2 + Float('0.082500000000000004', precision=53)*x5*x6*x7 + Float('-0.62', precision=53)], [x0*x15 + x10*x3*x9, Float('1.0', precision=53)*x0*x10*x2 - x15*x9, -x11*x7 + x4*x6, Float('0.316', precision=53)*x10*x2 + x14*x5 + x16*x2*x6], [-x0*x18 + Float('1.0', precision=53)*x6*x9, Float('1.0', precision=53)*x0*x6 + x18*x9, -x1*x17, -x10*x16 + Float('0.316', precision=53)*x6 + Float('0.35800000000000004', precision=53)], [0, 0, 0, Float('1.0', precision=53)]])