from .layer import Layer
import numpy as np
from .propagatinContext import PropagationContext

class MaxPool(Layer):
    def __init__(self, size):
        super().__init__()

        self.size = size

    def forward(self, input : np.ndarray) -> np.ndarray:
        # input shape (count_batch, count_APN, depth, height, width)

        # shape the array to allow np.nanmax
        ny, nx = input.shape[3]//self.size, input.shape[4]//self.size
        trimmed_input = input[..., :ny*self.size, :nx*self.size]
        shaped_array = trimmed_input.reshape(input.shape[:3] + (ny,nx,self.size,self.size))

        # do the max pooling
        output_activations = np.nanmax(shaped_array,axis=(5,6), keepdims=True)

        max_pos_mask = np.where(output_activations == shaped_array, 1, 0)
        output_activations = np.squeeze(output_activations, axis=(5,6))

        result = PropagationContext(input, output_activations)
        result.pos_max = max_pos_mask
        return result
    
    def backprop(self, dL_dOut : np.ndarray, forwardContext : PropagationContext) -> np.ndarray:
        pos_max = forwardContext.pos_max
        ib, iapn, ic, ih, iw, iy, ix = np.where(pos_max == 1)
        values = dL_dOut[ib, iapn, ic, ih, iw].flatten()
        result = np.zeros(forwardContext.input_activations.shape)
        result[ib, iapn, ic, ih*self.size + iy, iw*self.size + ix] = values
        return result

