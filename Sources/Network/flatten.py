from .layer import Layer
import numpy as np
from .propagatinContext import PropagationContext

class Flatten(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, input : np.ndarray) -> PropagationContext:
        # input shape (count_batch, count_APN, depth, height, width)
        # output shape is (count_batch, count_APN, length)
        m, apn, d, h, w = input.shape
        output_activations = input.reshape(m, apn, d*h*w)

        result = PropagationContext(input, output_activations)
        result.input_shape = input.shape
        return result
    
    def backprop(self, dL_dOut : np.ndarray, forwardContext : PropagationContext, step_number : int) -> np.ndarray:
        # input_activations & dL_dOut shape (count_batch, count_APN, depth, height, width)
        input_shape = forwardContext.input_shape
        return dL_dOut.reshape(input_shape)
