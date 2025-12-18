from sympy import *
q_2 = Symbol('q_2')
q_4 = Symbol('q_4')
q_5 = Symbol('q_5')
q_1 = Symbol('q_1')
q_3 = Symbol('q_3')
q_0 = Symbol('q_0')
q_6 = Symbol('q_6')
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
x10 = q_0*x9
x11 = x1*x4
x12 = q_0*q_1*x4
e = MutableDenseMatrix([[-x5 - x6, Float('2.0', precision=53)*q_1*q_2*x3 - x8, q_1*x7 + x10, q_4], [q_1*x9 + x8, -x11 - x6, Float('2.0', precision=53)*q_2*q_3*x3 - x12, q_5], [Float('2.0', precision=53)*q_1*q_3*x3 - x10, q_3*x9 + x12, -x11 - x5 + Float('1.0', precision=53), q_6], [0, 0, 0, 1]])