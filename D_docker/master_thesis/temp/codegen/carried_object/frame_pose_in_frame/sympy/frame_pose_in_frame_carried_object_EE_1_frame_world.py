from sympy import *
q_0 = Symbol('q_0')
q_6 = Symbol('q_6')
q_4 = Symbol('q_4')
q_2 = Symbol('q_2')
q_3 = Symbol('q_3')
q_1 = Symbol('q_1')
q_5 = Symbol('q_5')
x0 = q_2**2
x1 = q_1**2
x2 = q_3**2
x3 = 1/(q_0**2 + x0 + x1 + x2)
x4 = 2.0*x3
x5 = x0*x4
x6 = x2*x4 - 1.0
x7 = q_3*x4
x8 = q_0*x7
x9 = q_2*x4
x10 = q_1*x9
x11 = q_0*x9
x12 = q_1*x7
x13 = 0.4*x3
x14 = x1*x4
x15 = q_0*q_1*x4
x16 = q_3*x9
x17 = q_3*x13
x18 = q_2*x13
e = MutableDenseMatrix([[x5 + x6, x10 - x8, -x11 - x12, q_4 - x0*x13 - x13*x2 + Float('0.20000000000000001', precision=53)], [-x10 - x8, -x14 - x6, x15 - x16, q_0*x17 + q_1*x18 + q_5], [x11 - x12, x15 + x16, x14 + x5 + Float('-1.0', precision=53), -q_0*x18 + q_1*x17 + q_6], [0, 0, 0, 1]])