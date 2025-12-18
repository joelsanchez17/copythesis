import sympy as sp
from sympy import ( 
                   Symbol, symbols, IndexedBase,
                   )
from typing import Union,List  


def subs_variables_with_value(expr: 'SympyExpressionLike', variable: Union[List[Union[IndexedBase,Symbol]],Symbol,IndexedBase] ,value:float):
    result = expr
    if isinstance(variable,IndexedBase) or isinstance(variable,Symbol):
        variable = [variable]
    for var in variable:
        result = result.subs({var[i]:value for i in range(var.shape[0])})
    return result

def jacobian_wrt_indexedbase(expression:sp.Matrix,var: IndexedBase) -> sp.Matrix:
    if var.shape is None:
        raise ValueError('IndexedBase must have a shape')
        
    return expression.jacobian([var[i] for i in range(var.shape[0])])

def expressions_to_latex(expression_dict,symbol_names = {}):
    latex_code = "\\begin{align*}\n\\\\~\\\\\n\\\\~\\\\\n"
    for name, expression in expression_dict.items():
        latex_code += f"& {name} = {sp.latex(expression, symbol_names=symbol_names)} \\\\~\\\\\n" 
    latex_code += "\\end{align*}"
    return latex_code