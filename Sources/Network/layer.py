import numpy as np
from .propagatinContext import PropagationContext

class Layer():
    def __init__(self):
        pass

    def forward(self, input : np.ndarray) -> PropagationContext:
        raise NotImplementedError()
    
    def backprop(self, dL_dOut : np.ndarray, forwardContext : PropagationContext) -> np.ndarray:
        raise NotImplementedError
    