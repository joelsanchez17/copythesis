import numpy as np
from typing import List, Tuple, Dict, Callable, Any, Optional, Union
from pydrake.all import (AffineSystem, ValueProducer,AbstractValue,LeafSystem)
import copy

def MovingAverageSystem(number_of_inputs, window_length, time_period):
    t = np.eye(window_length-2)
    A = np.hstack((np.vstack((np.zeros((1,window_length-2)),t,)),np.zeros((window_length-1,1))))
    B = np.array([1,*([0]*(window_length-2))]).reshape(-1,1)
    C = np.array([[1/window_length]*(window_length-1)])
    D = np.array([[1/window_length]])
    
    # stack the matrices so we have num_inputs systems in parallel
    Al = np.zeros((number_of_inputs*(window_length-1),number_of_inputs*(window_length-1)))
    Bl = np.zeros((number_of_inputs*(window_length-1),number_of_inputs))
    Cl = np.zeros((number_of_inputs,number_of_inputs*(window_length-1)))
    Dl = np.zeros((number_of_inputs,number_of_inputs))
    for i in range(number_of_inputs):
        Al[i*(window_length-1):(i+1)*(window_length-1),i*(window_length-1):(i+1)*(window_length-1)] = A
        Bl[i*(window_length-1):(i+1)*(window_length-1),i] = B.flatten()
        Cl[i,i*(window_length-1):(i+1)*(window_length-1)] = C.flatten()
        Dl[i,i] = D.flatten()
        
    # AffineSystem.__init__(self,A=Al,B=Bl,C=Cl,D=Dl,time_period=time_period)
    return (affine_system := AffineSystem(A=Al,B=Bl,C=Cl,D=Dl,time_period=time_period))


class AbstractLogger(LeafSystem):
    class Log():
        def __init__(self,preallocated_size):
            self._sample_times = [None]*preallocated_size
            self._values = [None]*preallocated_size
            self.index = 0
            self.preallocated_size = preallocated_size
        def record(self, time,value):
            if self.index >= len(self._sample_times):
                self._sample_times += [None]*self.preallocated_size
                self._values += [None]*self.preallocated_size
            self._sample_times[self.index] = time
            self._values[self.index] = value
            self.index += 1
        @property
        def sample_times(self):
            return self._sample_times[:self.index]
        @property
        def values(self):
            return self._values[:self.index]
    def __init__(self, value_type, publish_period_seconds: float = 0,preallocated_size=10000):
        super().__init__()
        if publish_period_seconds > 0:
            self.DeclarePeriodicPublishEvent(publish_period_seconds,0,self.record)
        else:
            self.DeclarePerStepPublishEvent(self.record)
        self._input_port = self.DeclareAbstractInputPort(
            'input', AbstractValue.Make(value_type))
        # self._input_port = self.DeclareAbstractInputPort(
        #     'input', Value(value_type))
        # self.preallocated_size = preallocated_size
        # self.sample_times = [None]*preallocated_size
        # self.values = []*preallocated_size
        self.log_cache_index = self.DeclareCacheEntry(description = "log",
                                                      value_producer = ValueProducer(AbstractValue.Make(AbstractLogger.Log(preallocated_size)).Clone,calc= ValueProducer.NoopCalc),
                                                      prerequisites_of_calc={self.nothing_ticket()}).cache_index()
        
    def record(self, context):
        time = context.get_time()
        input_value = self._input_port.Eval(context)
        value = self.get_input_port().Allocate()
        value.set_value(input_value)
        self.get_cache_entry(self.log_cache_index).get_mutable_cache_entry_value(context).GetMutableValueOrThrow().record(time,value.get_value())
    def FindLog(self,context):
        return self.get_cache_entry(self.log_cache_index).get_mutable_cache_entry_value(context).GetMutableValueOrThrow()
        # return self.get_cache_entry(self.log_cache_index).get_mutable_cache_entry_value(context).GetMutableValueOrThrow()
    @classmethod
    def LogAbstractOutput(cls,output_port,builder,publish_period = 0.0,preallocated_size=10000):
        x = output_port.Allocate()
        var_type = type(x.get_value())()
        # var_type = ImageRgba8U()
        # var_type = output_port.Allocate().Make()
        abstract_logger = builder.AddSystem(AbstractLogger(var_type,publish_period,preallocated_size))
        builder.Connect(output_port,abstract_logger.get_input_port())
        return abstract_logger