from pydrake.systems.framework import BasicVector, LeafSystem, Context
import torch
import numpy as np
from typing import Callable, Dict, Iterable, List, Literal, NamedTuple, Optional, Tuple, Union
class FunctionInput(NamedTuple):
    """
    Represents a generic function input in the form of a named tuple, encompassing details for functions that can be used to describe dynamics or additional outputs.

    Attributes:
        name (str): A unique name for the function. This is used to identify the function within a system.

        function (Callable): The function to be represented. This should be a callable object that describes either the dynamics or additional outputs of a system.

        inputs (Dict[Literal['x','t','others'], int]): A dictionary specifying the inputs of the function.
            'x': The state variable.
            't': The time variable.
            x and t are special variables that are automatically gotten from the system context.
            'others': Any other input variables that are required by the function.
            The dictionary maps each input to its dimension.

        output_len (int): Specifies the length of the output from the function. This must match the actual output length of the provided function.

        to_numpy (Callable, optional): A conversion function that takes the input of the defined `function` and converts it into a NumPy array if necessary. Defaults to the identity function.

        from_numpy (Callable, optional): A conversion function that takes a NumPy array and converts it back into the expected input format for the defined `function`. Defaults to the identity function.

    Notes:
        - Ensure that the defined input dictionary and output length are consistent with the actual function implementation.
        - The `to_numpy` and `from_numpy` methods can be used to ensure compatibility between different data types such as PyTorch tensors and NumPy arrays.
    """
    name: str
    function: Callable
    inputs: Dict[Literal['x','t','others'], int]
    output_len: int
    to_numpy: Callable = lambda x: x
    from_numpy: Callable = lambda x: x
    
def create_drake_dynamical_system(dynamics:Optional[FunctionInput] = None, 
                                  discrete_or_continuous: Optional[Literal['continuous','discrete']] = None, 
                                  output_functions: Optional[Iterable[FunctionInput]] = None, 
                                  time_step:int=None) -> LeafSystem:
    """
    Creates a Drake LeafSystem representing a dynamical system, defining the system dynamics and output functions.

    Parameters:
        dynamics (Optional[Union[PytorchFunction, NumpyFunction]]): Describes the dynamics function. 
            The function can be a PyTorch or a NumPy function and must have the inputs and output length defined.
            If None, the system will not have dynamics.
            
        discrete_or_continuous (Optional[Literal['continuous', 'discrete']]): Specifies whether the system is 
            continuous or discrete. If 'discrete', a time_step must also be specified.
            If None, the system will not have dynamics, and this argument will be ignored.

        output_functions (Optional[Iterable[Union[PytorchFunction, NumpyFunction]]]): A list of output functions that 
            the system should have. Each output function must be defined similarly to the dynamics function. 
            If None, no additional output functions will be added to the system.

        time_step (int): Required if discrete_or_continuous is 'discrete'. Specifies the time step for the discrete 
            system update.

    Returns:
        DynamicalLeafSystem: A custom LeafSystem object representing the dynamical system.

    Notes:
        - If the system is discrete, a time_step must be provided. If not, a ValueError will be raised.
        - The provided dynamics and output_functions must match the specified input and output dimensions. Inconsistent dimensions will lead to unexpected behavior.
        - The `to_numpy` and `from_numpy` conversion functions within FunctionInput objects must be consistent with the expected data types in the defined functions (e.g., converting between PyTorch tensors and NumPy arrays).
        - If dynamics is None, the system will not have dynamics, and it will ignore the discrete_or_continuous argument.
        - Extra inputs not defined in the 'x' and 't' can be defined in the PytorchFunction or NumpyFunction.
            These will be automatically declared as input ports.
        - Ensure that the output length of the functions is consistent with the actual output shape.
        - The implementation relies on specific keys for inputs such as 'x' for the state and 't' for time. Ensure that these are 
            followed.
        - This function is designed to be flexible but may be sensitive to inconsistencies between the defined 
            shapes and the actual input and output shapes of the functions.
        - The FunctionInput is defined as follows; anything that has those fields can be used as functions::
        
            class FunctionInput(NamedTuple):
                name: str
                function: Callable
                inputs: Dict[Literal['x','t','others'], int]
                output_len: int
                to_numpy: Callable = lambda x: x
                from_numpy: Callable = lambda x: x
                
    Example:: 
    
        def pytorch_dynamics(x, u,x_ref):
            return (x + u[0]*3)[0:2]
        layer = torch.nn.Linear(5,5)
        linear_layer = lambda x: -abs(layer(x))*x

        def numpy_output_function(x, u,other_input):
            return (np.sin(x) + u[0]*3)[0:2]
        torch_to_numpy = lambda x: x.detach().numpy()
        numpy_to_torch = lambda x: torch.tensor(x,dtype=torch.float32)
        dynamics = FunctionInput(name='dynamics',function=linear_layer,inputs={'x':5},output_len=5,to_numpy=torch_to_numpy,from_numpy=numpy_to_torch)
        output_functions = [FunctionInput(name='output1',function=pytorch_dynamics,inputs={'x':5,'u':2,'x_ref':2},output_len=2,to_numpy=torch_to_numpy,from_numpy=numpy_to_torch),
                            FunctionInput(name='output2',function=numpy_output_function,inputs={'x':5,'u':2,'other_input':2},output_len=2),
                            FunctionInput(name='output3',function=linear_layer,inputs={'x':5},output_len=5,to_numpy=torch_to_numpy,from_numpy=numpy_to_torch),
                            FunctionInput(name='states',function=lambda x:x,inputs={'x':5},output_len=5),
                            ]
        leaf_system = create_drake_dynamical_system(dynamics, 'continuous', output_functions)
        context = leaf_system.CreateDefaultContext()
        context.SetContinuousState([9,18,3,3,3])
        leaf_system.GetInputPort('u').FixValue(context,[1,2])
        leaf_system.GetInputPort('x_ref').FixValue(context,[1,2])
        leaf_system.GetInputPort('other_input').FixValue(context,[1,2])
        print(leaf_system.GetOutputPort('output1').Eval(context))
        print(leaf_system.GetOutputPort('output2').Eval(context))
        print(leaf_system.GetOutputPort('output3').Eval(context))
        print(leaf_system.EvalTimeDerivatives(context).CopyToVector())
        # plant.EvalUniquePeriodicDiscreteUpdate(plant_context_).get_data()
    """
    if output_functions is not None and not isinstance(output_functions, Iterable):
        output_functions = [output_functions]
    if dynamics is not None:
        num_states = dynamics.inputs['x']
        
    if output_functions is None:
        output_functions = []
        
    is_discrete = discrete_or_continuous.lower() == 'discrete' if discrete_or_continuous is not None else False
    if is_discrete and time_step is None:
        raise ValueError("time_step must be specified for discrete systems")
    
    class DynamicalLeafSystem(LeafSystem):
        def __init__(self):
            LeafSystem.__init__(self)
            if dynamics is not None:
                self.dynamics = dynamics
                self.is_discrete = is_discrete
                
                diff = dynamics.inputs.keys() - {'x','t'}
                if len(diff)>0:
                    for key in diff:
                        self.DeclareVectorInputPort(key, BasicVector(dynamics.inputs[key]))
                
                self.update_function = self.MakeOutputFunction(dynamics.function,dynamics.inputs.keys(),from_numpy=dynamics.from_numpy,to_numpy=dynamics.to_numpy)
                
                if self.is_discrete:
                    self.DeclareDiscreteState(num_states)
                    self.DeclarePeriodicDiscreteUpdateEvent(
                    period_sec=time_step,
                    offset_sec=0.0,
                    update=self.update_function)
                else:
                    self.DeclareContinuousState(num_states)
                    self.DoCalcTimeDerivatives = self.update_function

            # Define output ports for each output function
            for output_function in output_functions:                
                name,function,inputs,output_shape,to_numpy,from_numpy = output_function
                diff = inputs.keys() - {'x','t'}
                if len(diff)>0:
                    for key in diff:
                        try:
                            self.GetInputPort(key)
                        except:
                            self.DeclareVectorInputPort(key, BasicVector(inputs[key]))
                self.DeclareVectorOutputPort(name, BasicVector(output_shape), self.MakeOutputFunction(function,inputs.keys(),from_numpy=from_numpy,to_numpy=to_numpy))
                
        def MakeOutputFunction(self, output_function:Callable, input_variables:set = {},from_numpy:Callable = lambda x: x,to_numpy: Callable = lambda x: x) -> Callable:
            def output_function_wrapper(context: Context, output):
                x = context.get_continuous_state_vector().CopyToVector() if not self.is_discrete else context.get_discrete_state_vector().CopyToVector()
                t = context.get_time()
                inputs = {'x':x,'t':t}
                inputs.update({key:self.GetInputPort(key).Eval(context) for key in input_variables - {'x','t'}}) 
                inputs = {key:from_numpy(inputs[key]) for key in input_variables}
                output_val = to_numpy(output_function(**inputs))
                output.SetFromVector(output_val)
            return output_function_wrapper
        
                
    return DynamicalLeafSystem()