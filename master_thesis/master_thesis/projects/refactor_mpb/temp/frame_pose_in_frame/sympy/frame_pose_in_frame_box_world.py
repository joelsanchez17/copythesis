from sympy import *
q_8 = Symbol('q_8')
q_9 = Symbol('q_9')
q_11 = Symbol('q_11')
q_7 = Symbol('q_7')
q_13 = Symbol('q_13')
q_12 = Symbol('q_12')
q_10 = Symbol('q_10')
x0 = 1/(q_10**2.0 + q_7**2.0 + q_8**2.0 + q_9**2.0)
x1 = 2.0*x0
x2 = q_9**2*x1
x3 = q_10**2*x1 - 1.0
x4 = q_7*x1
x5 = q_10*x4
x6 = q_8*x1
x7 = q_10*x6
x8 = q_9*x4
x9 = q_8**2*x1
x10 = q_10*q_9*x1
x11 = q_7*x6
e = MutableDenseMatrix([[-x2 - x3, Float('2.0', precision=53)*q_8*q_9*x0 - x5, x7 + x8, q_11], [q_9*x6 + x5, -x3 - x9, x10 - x11, q_12], [x7 - x8, x10 + x11, -x2 - x9 + Float('1.0', precision=53), q_13], [0, 0, 0, Float('1.0', precision=53)]])