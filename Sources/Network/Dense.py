from .trainableLayer import TrainableLayer
import numpy as np

class Dense(TrainableLayer):
    def __init__(self, layer_shape):
        super().__init__(layer_shape)

    def calc(self, v : np.ndarray) -> np.ndarray:
        # v shape is (count_batch, count_APN, length)
        return np.tensordot(v, self.weights, axes=([-1], [-1]))
    
    def get_gradient_weights(self, dL_dO: np.ndarray, input_activ: np.ndarray) -> np.ndarray:
        # dL_dO & input_activ shape is (count_batch, count_APN, length)
        dW_all = np.einsum('ijk,ijl->ijkl', dL_dO, input_activ)
        return np.mean(np.sum(dW_all, axis=1), axis=0)
    
    def get_gradient_biases(self, dL_dO: np.ndarray) -> np.ndarray:
        # dL_dO shape is (count_batch, count_APN, length)
        # result shape is (length,)
        return np.mean(np.sum(dL_dO, 1), 0)

    def get_gradient_inputs(self, dL_dO: np.ndarray, input_shape) -> np.ndarray:
        # dL_dO shape is (count_batch, count_APN, length)
        # weights shape is (height, width)
        return np.squeeze(self.weights.transpose([1, 0])@dL_dO[..., None], axis=-1)