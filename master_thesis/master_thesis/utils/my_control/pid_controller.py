
import numpy as np
from typing import List, Tuple, Dict, Callable, Any, Optional, Union

class PIDController:
    def __init__(self, kp: Union[np.double,List[np.double]], ki: Union[np.double,List[np.double]], kd: Union[np.double,List[np.double]], num_variables: Optional[int] = None, integral_saturation: Optional[Union[np.double,List[np.double]]] = None):
        #
        
        if isinstance(kp, (float,np.double,int,np.double,np.float64)):
            kp = [kp]
        if isinstance(ki, (float,np.double,int,np.double,np.float64)):
            ki = [ki]
        if isinstance(kd, (float,np.double,int,np.double,np.float64)):
            kd = [kd]
        if num_variables is None:
            num_variables = len(kp)
        if len(kp) != num_variables or len(ki) != num_variables or len(kd) != num_variables:
            if len(kp) == 1:
                kp = kp * num_variables
                ki = ki * num_variables
                kd = kd * num_variables
            else:
                raise ValueError('kp, ki, kd must have the same length = num_variables')
        if integral_saturation is None:
            integral_saturation = [np.inf] * num_variables
        if isinstance(integral_saturation, (float,np.double,int,np.double,np.float64)):
            integral_saturation = [integral_saturation]
        if len(integral_saturation) != num_variables:
            if len(integral_saturation) == 1:
                integral_saturation = integral_saturation * num_variables
            else:
                raise ValueError('integral_saturation must have the same length = num_variables')
        self.integral_saturation = np.asarray(integral_saturation)
        kp = np.array(kp)
        ki = np.array(ki)
        kd = np.array(kd)
        
        self.current_control = np.array([0.0] * num_variables)
        #
        self.kp = kp
        self.ki = ki
        self.kd = kd
        #
        self.reference = np.array([0.0] * num_variables)
        #
        self.current_error  = np.array([0.0] * num_variables)
        self.last_error = np.array([0.0] * num_variables)
        self.error_rate = np.array([0.0] * num_variables)
        self.summed_error   = np.array([0.0] * num_variables)
        #
        self.current_time: np.double = None
        self.last_time: np.double = None
        self.elapsed_time: np.double = None

    def update_time(self, current_time:float):
        if self.current_time is None:
            self.current_time = current_time
            self.last_time = current_time
            self.elapsed_time = current_time - current_time
            return
        self.last_time = self.current_time
        self.current_time = current_time
        self.elapsed_time = self.current_time - self.last_time
        
    def update_errors(self, measurement):
        if self.current_time is None:
            raise ValueError('Time not initialized')
        measurement = np.asarray(measurement)
        elapsed_time = self.elapsed_time
        self.last_error = self.current_error
        self.current_error = self.reference - measurement
        self.summed_error += self.current_error * elapsed_time
        for i in range(len(self.summed_error)):
            if abs(self.summed_error[i]) > self.integral_saturation[i]:
                self.summed_error[i] = self.summed_error[i]/abs(self.summed_error[i])*self.integral_saturation[i]
        self.error_rate = (self.current_error - self.last_error) / (elapsed_time+1e-16)

    def update_reference(self, next_reference):
        self.reference = np.asarray(next_reference)

    def update_control(self):
        if self.current_time is None:
            raise ValueError('Time not initialized')
        proportional_control = self.kp * self.current_error
        integral_control = self.ki * self.summed_error
        derivative_control = self.kd * self.error_rate
        self.current_control = proportional_control + integral_control + derivative_control

    def get_control(self)  -> np.ndarray:
        
        return self.current_control

    def update_and_calculate_control(self, measurement:np.double, next_reference:np.double, current_time:float) -> np.ndarray:
        self.update_time(current_time)
        self.update_reference(next_reference)
        self.update_errors(measurement)
        self.update_control()
        return self.get_control()


        
    
        

