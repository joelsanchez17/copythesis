import sympy as sp
from sympy import (Function, 
                   Symbol, IndexedBase,
                   Idx, parse_expr)
from typing import Union,List,NamedTuple,Literal    ,Iterable,Dict,Optional,Callable
import random, string
from toolz import compose_left
from sympy.codegen.ast import  Assignment,CodeBlock, Return
class ParsingVariable(NamedTuple):
    name: str
    shape: List[int] 
class ParsingResult(NamedTuple):
    variables: Dict[str,IndexedBase]
    expression: sp.Matrix 

def parse_string_expressions(expressions: Union[List[str],str], variables: Union[ParsingVariable,List[ParsingVariable]], index_notation = Literal['parenthesis', 'bracket']
                             ) -> ParsingResult:
    """
    Parses a list of string expressions into a sympy matrix.

    Args:
        expressions (Union[List[str],str]): A list of string expressions or a single string expression to be parsed.
        variables (Union[ParsingVariable,List[ParsingVariable]]): A list of ParsingVariable objects or a single ParsingVariable object representing the variables in the expressions.
        index_notation (Literal['parenthesis', 'bracket'], optional): The index notation to be used in the parsed expressions. Defaults to 'parenthesis'.

    Returns:
        ParsingResult: A ParsingResult object containing the parsed variables and expressions as a sympy matrix.
    """
def parse_string_expressions(expressions: Union[List[str],str], variables: Union[ParsingVariable,List[ParsingVariable]], index_notation = Literal['parenthesis', 'bracket']
                             ) -> ParsingResult:
    """Parses a list of string expressions into a sympy matrix."""
    
    if isinstance(expressions,str):
        expressions = [expressions]
    if isinstance(variables,ParsingVariable):
        variables = [variables]
    
    try:
        # Change the index notation to fit sympy's convention.
        if index_notation == 'parenthesis':
            # Create a dictionary of local variables, replacing the names with the functions.
            local_dict = {var.name:Function(var.name) for var in variables if len(var.shape) != 0}
            # Parse the strings into sympy expressions.
            sympy_expressions = [parse_expr(string,local_dict=local_dict) for string in expressions]
            
            # Replace the functions with the indexed bases in the expressions.
            for current_depth in range(len(sympy_expressions)):
                for var in variables:
                    name = var.name
                    shape = var.shape
                    sympy_expressions[current_depth] = sympy_expressions[current_depth].replace(Symbol(name),lambda: IndexedBase(name,shape=shape)) #for those that weren't replaced by the local_dict
                    sympy_expressions[current_depth] = sympy_expressions[current_depth].replace(Function(name),lambda arg: IndexedBase(name,shape=shape)[arg])
            
            # Replace the functions with the indexed bases in the variables.
            variables = {var.name:IndexedBase(var.name, var.shape) for var in variables}
        elif index_notation == 'bracket':
            # Create a dictionary of local variables, replacing the names with the indexed bases.
            local_dict = {var.name:IndexedBase(var.name, var.shape) for var in variables}
            # Replace the functions with the indexed bases in the variables.
            variables = local_dict
            # Parse the strings into sympy expressions.
            sympy_expressions = [parse_expr(string,local_dict=local_dict) for string in expressions]
        
        # variables = [IndexedBase(var.name, var.shape) for var in variables]
    except TypeError as e:
        print(f'Protip: change index notation from {index_notation} to {"bracket" if index_notation=="parenthesis" else "parenthesis"}')
        raise e
    sympy_expressions = sp.Matrix(sympy_expressions)
    return ParsingResult(variables,sympy_expressions)
def batch_expression(expression: sp.Expr,variables_to_batch: Union[List[sp.IndexedBase],sp.IndexedBase],new_dimension_name:str = 'batch'
                     ) -> Dict:
    if not isinstance(variables_to_batch,Iterable):
        variables_to_batch = (variables_to_batch,) 
    
    batch_dimension = sp.Symbol(new_dimension_name+'_dimension')
    # batch_slice = sp.Symbol(new_dimension_name+'_slice')
    batch_slice = sp.Symbol(':')
    new_vars = []
    for var in variables_to_batch:
        # if isinstance(var,(sp.IndexedBase,sp.Indexed)):
        if hasattr(var,'shape'):
            new_var = sp.IndexedBase(str(var),shape=((batch_dimension,) + var.shape) )
            new_vars.append(new_var)
            # wildcard for the indices of the variable
            wildcards = tuple(map(compose_left(lambda x: 'wildcard_'+str(x),sp.Wild),range(len(var.shape))))
            # replace doesn't work for some reason
            matches = expression.find(var[wildcards])
            for match in matches:
                expression = expression.replace(var[match.indices],new_var[[batch_slice]+list(match.indices)])
        # else:
        #     new_var = sp.IndexedBase(str(var),shape=(batch_dimension,) )
        #     new_vars.append(new_var)
        #     # wildcard for the indices of the variable
        #     # wildcards = list(map(compose_left(lambda x: 'wildcard_'+str(x),sp.Wild),range(len(var.shape))))
        #     # replace doesn't work for some reason
        #     matches = expression.find(var)
        #     for match in matches:
        #         expression = expression.replace(var,new_var[batch_slice])
    return {'expression':expression,'batched_variables':new_vars,'batch_dimension':batch_dimension,'batch_slice':batch_slice}


def multi_dim_indices(shape):
    if len(shape) == 1:
        for i in range(shape[0]):
            yield (i,)
    else:
        for i in range(shape[0]):
            for sub_index in multi_dim_indices(shape[1:]):
                yield (i,) + sub_index
    
def codeblock_to_batched_codeblock(codeblock: CodeBlock,variables_to_batch: List[IndexedBase]) -> CodeBlock:
    """

    """ 
    assignments = tuple(cb for cb in codeblock)
    variables_to_batch = variables_to_batch
    assignments = sp.Array(assignments)
    result = batch_expression(assignments,variables_to_batch=variables_to_batch,new_dimension_name='batch')
    
    return CodeBlock(*result["expression"])


def sympy_to_pytorch(expression: sp.Expr,in_args: Dict[str,List[sp.Basic]],function_name:str,condense_inputs:Optional[bool]=False,has_self:Optional[bool]=False,batched:Optional[List[bool]]=False,tab = "    ") -> str:
    sympy_torch_func_lookup = {
        
        "Abs": "torch.abs",
        "sign": "torch.sign",
        # Note: May raise error for ints.
        "ceiling": "torch.ceil",
        "floor": "torch.floor",
        "log": "torch.log",
        "exp": "torch.exp",
        "Sqrt": "torch.sqrt",
        "cos": "torch.cos",
        "acos": "torch.acos",
        "sin": "torch.sin",
        "asin": "torch.asin",
        "tan": "torch.tan",
        "atan": "torch.atan",
        "atan2": "torch.atan2",
        # Note: May give NaN for complex results.
        "cosh": "torch.cosh",
        "acosh": "torch.acosh",
        "sinh": "torch.sinh",
        "asinh": "torch.asinh",
        "tanh": "torch.tanh",
        "atanh": "torch.atanh",
        "Pow": "torch.pow",
        "re": "torch.real",
        "im": "torch.imag",
        "arg": "torch.angle",
        # Note: May raise error for ints and complexes
        "erf": "torch.erf",
        "loggamma": "torch.lgamma",
        "Eq": "torch.eq",
        "Ne": "torch.ne",
        "StrictGreaterThan": "torch.gt",
        "StrictLessThan": "torch.lt",
        "LessThan": "torch.le",
        "GreaterThan": "torch.ge",
        "And": "torch.logical_and",
        "Or": "torch.logical_or",
        "Not": "torch.logical_not",
        "Max": "torch.max",
        "Min": "torch.min",
        # Matrices
        "MatAdd": "torch.add",
        "HadamardProduct": "torch.mul",
        "Trace": "torch.trace",
        # Note: May raise error for integer matrices.
        "Determinant": "torch.det",
        
        "conjugate": "torch.conj",
    }
    if isinstance(batched, bool):
        batched = [batched]*len(in_args)
    # {arg:arg.name for arg in (states+parameters)}
    variables = []
    for key,value in in_args.items():
        variables += value
    
    first_var_name = variables[0].name
    if isinstance(expression,sp.Matrix) or isinstance(expression,sp.Array):
        shape = expression.shape
        indices = list(multi_dim_indices(shape))
        array_elements = [sp.Symbol(f"_x{i}") for i,indice in enumerate(indices)]
        # assirhs=
        assignments = [Assignment(var,expression[indices]) for var,indices in zip(array_elements,indices)]
        # n = sp.Symbol('n')
        out_variable = sp.IndexedBase(f'out',shape = shape)
        # if len(shape) == 1:
        #     out_variable = sp.IndexedBase(f'out',shape[0])
        # elif len(shape) == 2:
        #     out_variable = sp.MatrixSymbol(f'out',*shape)
        # else:
        #     raise ValueError(f'Expected shape to be of length 1 or 2, but got {len(shape)} (gotta implement still for higher dimensions)')
        output_assignments = [Assignment(out_variable[indice],var) for var,indice in zip(array_elements,indices)]
        
        optimized_code = CodeBlock(*assignments).cse()
        
        final_codeblock = CodeBlock(*(list(optimized_code) + output_assignments + [Return(out_variable)]))
        if any(batched):
            for batch,(key,value) in zip(batched,in_args.items()):
                if batch:
                    final_codeblock = codeblock_to_batched_codeblock(final_codeblock,variables_to_batch=[out_variable] + value)
            
            # final_codeblock = CodeBlock(*[Assignment(assignment.lhs,sp.Function('torch.squeeze')(assignment.rhs)) if isinstance(assignment,Assignment) else assignment for assignment in final_codeblock])
            out_var_string = f"out = torch.empty_like({first_var_name}).resize_({first_var_name + '.shape[0],' + ','.join(map(str,expression.shape))})\n"
        else:
            out_var_string = f"out = torch.empty_like({first_var_name}).resize_({','.join(map(str,expression.shape))})\n"
        _,_,body = sp.pycode(final_codeblock,user_functions=sympy_torch_func_lookup,allow_unknown_functions=True,human=False)
        
        # body = out_var_string + body
    elif isinstance(expression,sp.Expr):
        
        out_variable = sp.Symbol('out')
        assignments = [Assignment(out_variable,expression)]
        final_codeblock = CodeBlock(*assignments).cse()
            
        for batch,(key,value) in zip(batched,in_args.items()):
            if batch:
                final_codeblock = codeblock_to_batched_codeblock(final_codeblock,variables_to_batch=value)
                # final_codeblock = CodeBlock(*[Assignment(assignment.lhs,sp.Function('torch.squeeze')(assignment.rhs)) if isinstance(assignment,Assignment) else assignment for assignment in final_codeblock])
        out_var_string = ""
        _,_,body = sp.pycode(CodeBlock(*(list(final_codeblock) + [Return(out_variable)])),user_functions=sympy_torch_func_lookup,allow_unknown_functions=True,human=False)
        
    else:
        raise TypeError(f'Expected expression to be of type Matrix, Array, or Expr, but got {type(expression)} this might be fixable by adding another if else statement and adapting the code to the new type')

    if condense_inputs:
        unpacking_str = ""
        args = ("self," if has_self else "")
        for batch,(key,value) in zip(batched,in_args.items()):
            if batch:
                unpacking_str += f"{','.join([var.name for var in value])} = {key}.transpose(1,0)\n"
            else:
                unpacking_str += f"{','.join([var.name for var in value])} = {key}\n"
            args += f"{key},"
        # unpacking_str = f"{','.join([var.name for var in variables])} = args\n"
        # args = ("self," if has_self else "") + "args"
    else:
        unpacking_str = ""
        args = ("self," if has_self else "")
        for key,value in in_args.items():
            args += ','.join([var.name for var in value])
            args += ','
        # args = ("self," if has_self else "") + ','.join([var.name for var in variables])
    function_body_tabbed = '\n'.join([tab+line for line in (unpacking_str + out_var_string + body).split('\n')])
    
    func = """def {function_name:s}({args:s}):
{body:s}""".format(function_name=function_name,
                    args=args,
                    body=function_body_tabbed).replace("math.sqrt","torch.sqrt")
    return func

# class BatchedResult(NamedTuple):
#     expression: sp.Expr
#     batched_variables: List[sp.IndexedBase]
#     batch_dimension: sp.Symbol
#     batch_slice: sp.Symbol

# def batch_expression(expression: sp.Expr,variables_to_batch: Union[List[sp.IndexedBase],sp.IndexedBase],new_dimension_name:str = 'batch'
#                      ) -> BatchedResult:
#     if not isinstance(variables_to_batch,Iterable):
#         variables_to_batch = (variables_to_batch,) 
    
#     batch_dimension = sp.Symbol(new_dimension_name+'_dimension')
#     batch_slice = sp.Symbol(new_dimension_name+'_slice')
#     new_vars = []
#     for var in variables_to_batch:
#         new_var = sp.IndexedBase(str(var),shape=(batch_dimension,)+var.shape)
#         new_vars.append(new_var)
#         # wildcard for the indices of the variable
#         wildcards = tuple(map(compose_left(lambda x: 'wildcard_'+str(x),sp.Wild),range(len(var.shape))))
#         # replace doesn't work for some reason
#         matches = expression.find(var[wildcards])
#         for match in matches:
#             expression = expression.replace(var[match.indices],new_var[[batch_slice]+list(match.indices)])
#     return BatchedResult(expression,new_vars,batch_dimension,batch_slice)

# def multi_dim_indices(shape):
#     if len(shape) == 1:
#         for i in range(shape[0]):
#             yield (i,)
#     else:
#         for i in range(shape[0]):
#             for sub_index in multi_dim_indices(shape[1:]):
#                 yield (i,) + sub_index
# def create_assignments(expression: Union[sp.Matrix, sp.Array, sp.Expr]) -> List[Assignment]:
#     """
#     Create a list of assignments for each element of the input expression.

#     Args:
#         expression (Union[sp.Matrix, sp.Array, sp.Expr]): The input expression.

#     Returns:
#         List[Assignment]: A list of assignments where each assignment is of the form
#         dx[i,j,...] = expression[i,j,...], where dx is an indexed base with shape [m,n,...]
#         and expression is an array of shape [m,n,...].

#     Raises:
#         TypeError: If the input expression is not of type Matrix, Array, or Expr.
#     TODO: use this?
#     class sympy.codegen.ast.Element(*args, **kwargs)
#     Gets a [m,n,...] array of sympy expressions (or just one) and create assignments for each element, returning a list of assignments where each assignment is 
#     of the form dx[i,j,...] = expression[i,j,...] for example, where RESULT is an array of shape [m,n,..] and declared before hand.    
#     """

#     if isinstance(expression,sp.Matrix) or isinstance(expression,sp.Array):
#         return [Assignment(sp.IndexedBase(f'out',shape=expression.shape)[indices],expression[indices]) for indices in multi_dim_indices(expression.shape)]
#     elif isinstance(expression,sp.Expr):
#         return [Assignment(sp.Symbol('out'),expression)]
#     else:
#         raise TypeError(f'Expected expression to be of type Matrix, Array, or Expr, but got {type(expression)}')
    
# def codeblock_to_batched_codeblock(codeblock: CodeBlock,variables_to_batch: List[IndexedBase]) -> CodeBlock:
#     """

#     """ 
#     assignments = tuple(cb for cb in codeblock)
#     variables_to_batch = variables_to_batch
#     assignments = sp.Array(assignments)
#     result = batch_expression(assignments,variables_to_batch=variables_to_batch,new_dimension_name='batch')
    
#     return CodeBlock(*([Assignment(result.batch_slice,sp.Function('SLICE')('NONE'))] + list(result.expression)))

# def replace_numbers_with_something(expression: sp.Expr,something: Callable[[sp.Expr],sp.Expr]) -> sp.Expr:
#     """
#     Replaces the numbers in a expression with something(number) if the number is not an index
#     """
#     def make_random_string(N):
#         return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(N))
    
#     w1 = sp.Wild('w1',properties=[lambda x: (len(x.args) > 0) and isinstance(x,sp.Indexed)])
#     w2 = sp.Wild('w2',properties=[lambda x: isinstance(x,sp.Number)])
#     def idx_to_symbol(w1):
#         name = str(w1.args[0])+make_random_string(5)+'_'+'_'.join((map(str,w1.args[1:])))
#         return sp.Symbol(name)
#     map_idx_to_symbol = {match: idx_to_symbol(match) for match in expression.find(w1)}
#     expression = expression.subs(map_idx_to_symbol)
#     expression = expression.replace(w2,something(w2))
#     expression = expression.subs({value:key for key,value in map_idx_to_symbol.items()})
#     return expression

# def write_python_function(expression: Union[sp.Matrix,sp.Array,sp.Expr],variables: Dict[str,IndexedBase],function_name:str,user_functions:Optional[Dict[str,str]] = {},batched:Optional[bool]=False, functions_to_apply: Optional[List[Callable[[sp.Expr],sp.Expr]]] = []) -> str:
#     """
#     TODO:
#     do something that can replace sp.Pow,Mul etc with custom functions
#     basically:
#     - expression.replace(sp.Pow,sp.Function('POW'))
#     - pycode(...,user_functions={'POW':'custom_pow'})
#     """
    
#     if isinstance(expression,sp.Expr):
#         expression = sp.Array([expression])
#     assignments = create_assignments(expression)
#     out_variable = sp.IndexedBase('out',shape=expression.shape)
#     codeblock = CodeBlock(*assignments).cse()
#     if batched:
#         variables_to_batch = [out_variable] + list(variables.values())
#         codeblock = codeblock_to_batched_codeblock(codeblock,variables_to_batch)
        
#     for function in functions_to_apply:
#         codeblock = (function(codeblock))
#     if len(functions_to_apply) > 0:
#         codeblock = codeblock.cse()
#     function_template = """def {function_name:s}({args:s}):
# {body:s}"""
#     user_func = {}
#     user_func.update(user_functions)
#     user_func.update({'SLICE':'SLICE'})
    
#     function_body = sp.pycode(codeblock,user_functions=user_func).replace('SLICE(NONE)','slice(None)')
#     function_body_tabbed = '\n'.join(['\t'+line for line in function_body.split('\n')])
#     func = function_template.format(function_name=function_name,args=','.join([str(var) for var in variables.values()] + ["out"]),body=function_body_tabbed)
#     return func