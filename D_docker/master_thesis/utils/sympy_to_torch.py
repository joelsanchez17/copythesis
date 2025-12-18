import sympy
from sympy import Function, Symbol, symbols, IndexedBase, Idx, parse_expr
from sympy.codegen.ast import CodeBlock, Assignment,Return
import logging

def sympy_to_torch(sympy_expressions,name,sympy_states,sympy_inputs = None,squeeze = True):

    torch_functions = {'torch.tensor': 'torch.tensor', 'sin': 'torch.sin', 'cos': 'torch.cos', 'pow': 'torch.pow', 'log': 'torch.log'}

    def constants_to_tensors(expr):
        # wild symbol to select all numbers
        w = sympy.Wild("w", properties=[lambda t: isinstance(t, sympy.Number)])
        # w2 = sympy.Wild("w2", properties=[lambda t: isinstance(t, sympy.IndexedBase)])
        # extract the numbers from the expression
        n = expr.find(w)
        if len(n) == 0:
            return expr
        # n2 = expr.find(w2)
        # f1 = next(iter(n2))
        # print(f1.name)
        # get a lowercase alphabet
        # alphabet = list(string.ascii_lowercase)
        alphabet = [str(i) for i in range(len(n))]
        # create a symbol for each number
        s = sympy.symbols("a a".join(alphabet))
        if len(n) == 1:
            s = (s,)
        # create a dictionary mapping a number to a symbol
        z = sympy.Function('torch.tensor')
        # d = {k: z(k) for k, v in zip(n, s)}
        d = {k: v for k, v in zip(n, s)}
        d2 = {v: z(k,'dtype') for k, v in zip(n, s)}
        # expr.subs(d) | log
        # print(d)
        return expr.subs(d).subs(d2)
    
    num_states = sympy_states.shape[0]
    temp_states = [sympy.Symbol('x'+str(i)) for i in range(num_states)]
    if sympy_inputs:
        num_inputs = sympy_inputs.shape[0]
        temp_inputs = [sympy.Symbol('u'+str(i)) for i in range(num_inputs)]
    
    
    
    torch_expressions = [] 
    if not isinstance(sympy_expressions, sympy.Matrix):
        sympy_expressions = sympy.Matrix(sympy_expressions)
        
    num_rows = sympy_expressions.shape[0]
    num_cols = sympy_expressions.shape[1]
    # for i,expression in enumerate(sympy_expressions):
    for row_idx in range(num_rows):
        col = []
        for col_idx in range(num_cols):
            expression = sympy_expressions[row_idx,col_idx]
            
            if sympy_inputs:
                temp_expression = expression.replace(sympy_states,tuple(temp_states)).replace(sympy_inputs,tuple(temp_inputs))
            else:
                temp_expression = expression.replace(sympy_states,tuple(temp_states))
            temp_expression_2 = constants_to_tensors(temp_expression)
            temp_expression_3 = temp_expression_2
            for i,state in enumerate(temp_states):
                temp_expression_3 = temp_expression_3.replace(state,sympy_states[i])
            if sympy_inputs:
                for i,input in enumerate(temp_inputs):
                    temp_expression_3 = temp_expression_3.replace(input,sympy_inputs[i])
            logging.debug(f"{expression} \n->\n{temp_expression_3}\n")
            # torch_expressions.append(temp_expression_3)
            col.append(temp_expression_3)
        torch_expressions.append(col)
    assignments = []
    for i in range(num_rows):
        for j in range(num_cols):
            # assignments.append(Assignment(sympy.Symbol(f'dx_{i}_{j}'), torch_expressions[i][j]))
            if squeeze:
                assignments.append(Assignment(sympy.IndexedBase(f'dx',shape=(num_rows,))[i], torch_expressions[i][j]))
            else:
                assignments.append(Assignment(sympy.IndexedBase(f'dx',shape=(num_rows,num_cols))[i,j], torch_expressions[i][j]))
    # codeblock = CodeBlock(
    #     *(Assignment(sympy.Symbol(f'dx_{i}_{j}'), expr) for (i,j),expr in zip(torch_expressions) )    
    # )
    codeblock = CodeBlock(*assignments)
    # dx = torch.empty_like(x).resize_(4,4)
    optimized_codeblock = CodeBlock(f'dx = torch.empty_like(x).resize_{sympy_expressions.shape}',*(cb for cb in codeblock.cse()),Return(sympy.IndexedBase(f'dx{".squeeze()" if squeeze else ""}')) )
    # optimized_codeblock = CodeBlock(,*(cb for cb in codeblock.cse()),Return(sympy.IndexedBase(f'dx{".squeeze()" if squeeze else ""}')) ) | log
    
    function_body = sympy.pycode(optimized_codeblock, user_functions= torch_functions,).replace('dtype','dtype=x.dtype' )
    function_body_tabbed = '\n'.join(['\t'+line for line in function_body.split('\n')])
    # code = f"""def {name}(x{',u' if sympy_inputs else ''}):\n{function_body_tabbed}"""
    code = f"""def {name}(x{',u' if sympy_inputs else ''}):\n{function_body_tabbed}"""
    return code