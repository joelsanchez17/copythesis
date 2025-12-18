import sympy

rotation_z = lambda angle: sympy.matrices.dense.rot_givens(1, 0, angle, dim=3)
rotation_y = lambda angle: sympy.matrices.dense.rot_givens(0, 2, angle, dim=3)
rotation_x = lambda angle: sympy.matrices.dense.rot_givens(2, 1, angle, dim=3)

def euler_derivatives_to_angular_velocity_matrix(order:str, variables:list):
    
    """
    Calculates the transformation matrix that transforms the euler angle derivatives to the angular velocity vector using SymPy.
    ```
    alpha = euler_angles
    
    T = euler_derivatives_to_angular_velocity_matrix('zyz',alpha)
    
    angular_velocity = T @ alpha_dot
    ```
    Intrisinc rotation, so rotations around the axes of the moving body.
    
    Example:
    ```
    psi,theta,phi = sympy.symbols('psi,theta,phi')
    psi_dot, theta_dot, phi_dot = sympy.symbols('\\dot{\\psi},\\dot{\\theta},\\dot{\\phi}')
    w_x, w_y, w_z = sympy.symbols('w_x,w_y,w_z')
    display(sympy.Eq(sympy.UnevaluatedExpr(sympy.Matrix([w_x,w_y,w_z])),sympy.UnevaluatedExpr(euler_derivatives_to_angular_velocity_matrix('zyz',[psi,theta,phi]))*sympy.UnevaluatedExpr(sympy.Matrix([psi_dot,theta_dot,phi_dot]))))
    ```
    """
    assert len(order) == 3
    assert order[0] != order[1] and order[1] != order[2]
    assert len(variables) >= 3
    matrices = []
    vectors = []
    new_vars = [sympy.symbols(f'var_{i}') for i in range(0,3)]
    for i in range(0,3):
        char = order[2-i]
        if char == 'x':
            # vectors.append(sympy.Matrix([[new_vars[i],0,0]]).T)
            vectors.append(sympy.Matrix([[1,0,0]]).T)
            matrices.append(rotation_x(variables[i]))
        if char == 'y':
            # vectors.append(sympy.Matrix([[0,new_vars[i],0]]).T)
            vectors.append(sympy.Matrix([[0,1,0]]).T)
            matrices.append(rotation_y(variables[i]))
        if char == 'z':
            # vectors.append(sympy.Matrix([[0,0,new_vars[i]]]).T)
            vectors.append(sympy.Matrix([[0,0,1]]).T)
            matrices.append(rotation_z(variables[i]))
    # R = matrix[0] @ matrix[1] @ matrix[2]
    # result = vectors[0] + matrices[0]@vectors[1] + matrices[0]@matrices[1]@vectors[2]
    # a,b = sympy.linear_eq_to_matrix(result, *new_vars)
    a = sympy.Matrix((vectors[0],matrices[0]@vectors[1],matrices[0]@matrices[1]@vectors[2])).reshape(3,3).T
    return a