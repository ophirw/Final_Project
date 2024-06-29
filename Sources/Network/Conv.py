from .trainableLayer import TrainableLayer
import numpy as np
from scipy.signal import fftconvolve

class Conv(TrainableLayer):
    def __init__(self, layer_shape):
        super().__init__(layer_shape)

    def calc(self, v : np.ndarray) -> np.ndarray:
        return self.Convolve3D(inputs=v, kernels=self.weights)
    
    def Convolve3D(self, inputs : np.ndarray, kernels : np.ndarray, mode : str="same") -> np.ndarray:
        # kernels shape is (count, depth, height, width)
        # inputs shape is (count_batch, count_APN, depths, height, width)
        result = np.zeros(inputs.shape[:2]+(kernels.shape[0],)+inputs.shape[3:])

        for i, triplet in enumerate(inputs[:, :, np.newaxis, ...]):
            for j, sample in enumerate(triplet):
                result[i, j] = np.sum(fftconvolve(sample, kernels, "same", [-2, -1]), -3)

        return result

    def get_gradient_weights(self, dL_dO: np.ndarray, input_activ: np.ndarray) -> np.ndarray: 
        # dL_dO & input_activ shape is (count_batch, count_APN, depth, height, width)
        # returns shape (count, depth, height, width)
        padded_input = self.padding(input_activ, (self.weights.shape[-1]-1)/2.0)

        dL_dO_reshaped = dL_dO.reshape(dL_dO.shape[:3] + (1,) + dL_dO.shape[3:])
        padded_input_reshaped = padded_input.reshape(padded_input.shape[:2] + (1,) + padded_input.shape[2:])

        dL_dW = np.mean(np.sum(fftconvolve(padded_input_reshaped, dL_dO_reshaped, mode='valid', axes=[-2, -1]), 1), 0)

        return dL_dW
    
    def get_gradient_inputs(self, dL_dO: np.ndarray, input_shape) -> np.ndarray:
        # dL_dO.shape & input_shape is (count_batch, count_APN, depth, height, width)
        # returns shape (count, depth, height, width)

        dL_dO_reshaped = dL_dO.reshape(dL_dO.shape[:3] + (1,) + dL_dO.shape[3:])
        kernels_reshaped = self.weights.reshape((1, 1) + self.weights.shape)
        kernels_reshaped = kernels_reshaped[..., ::-1, ::-1]
        dL_dInput = np.sum(fftconvolve(dL_dO_reshaped, kernels_reshaped, mode="full", axes=[-2, -1]), 2)

        trim_size = (dL_dInput.shape[-1]-input_shape[-1])/2.0
        dL_dInput_trimmed = dL_dInput[...,
                              int(np.floor(trim_size)):int(dL_dInput.shape[-2]-np.ceil(trim_size)), 
                              int(np.floor(trim_size)):int(dL_dInput.shape[-1]-np.ceil(trim_size))]
        
        return dL_dInput_trimmed

    def get_gradient_biases(self, dL_dO: np.ndarray) -> np.ndarray:
        # dL_dO shape is (count_batch, count_APN, depth, height, width)
        # returns shape (length)
        return np.mean(np.sum(dL_dO, axis=(1, 3, 4)), 0)

    def padding(self, activations, pad):
        return np.pad(activations, ((0, 0), (0, 0), (0, 0), (int(np.floor(pad)), int(np.ceil(pad))), (int(np.floor(pad)), int(np.ceil(pad)))), 'constant')
