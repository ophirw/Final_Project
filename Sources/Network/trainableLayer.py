import numpy as np
from .layer import Layer
from .globalVariables import *
from .propagatinContext import PropagationContext

### super class for all layers with trainable parameters
class TrainableLayer(Layer):
    def __init__(self, layer_shape):

        self.layer_shape = layer_shape

        self.weights : np.ndarray = self.weight_init_func()
        self.biases : np.ndarray = np.zeros(layer_shape[0],)

        self.weights_first_moments = np.zeros(self.weights.shape)
        self.weights_second_moments = np.zeros(self.weights.shape)

        self.biases_first_moments = np.zeros(self.biases.shape)
        self.biases_second_moments = np.zeros(self.biases.shape)
                

    def forward(self, input : np.ndarray) -> np.ndarray:
        output = self.calc(input)

        # reshaping biases to allow addition
        new_bias_shape = (1, 1, len(self.biases),) + (1,)*(len(output.shape[2:])-1)
        b = self.biases.reshape(new_bias_shape)

        result_Z = output+b
        result_output = self.activ_func(result_Z)

        result = PropagationContext(input, result_output)
        result.output_Zs = result_Z
        return result
    
    # implements Adam learning for layer, and returns the backpropagating tensor for the next layer
    def backprop(self, dL_dOut : np.ndarray, forwardContext : PropagationContext) -> np.ndarray:
        input_activations = forwardContext.input_activations
        output_Zs = forwardContext.output_Zs

        dL_dZ = dL_dOut*self.dactiv_func(output_Zs)
        dL_dW = self.get_gradient_weights(dL_dZ, input_activations)
        dL_dB = self.get_gradient_biases(dL_dZ)
        
        self.weights_first_moments = beta1*self.weights_first_moments + (1-beta1)*dL_dW
        self.weights_second_moments = beta2*self.weights_second_moments + (1-beta2)*(dL_dW**2)

        self.biases_first_moments = beta1*self.biases_first_moments + (1-beta1)*dL_dB
        self.biases_second_moments = beta2*self.biases_second_moments + (1-beta2)*(dL_dB**2)

        alpha_t = alpha*np.sqrt(1-(beta2**epoch_number))/(1-(beta1**epoch_number))

        self.weights -= alpha_t*self.weights_first_moments/(np.sqrt(self.weights_second_moments)+eps)
        self.biases -= alpha_t*self.biases_first_moments/(np.sqrt(self.biases_second_moments)+eps)

        return self.get_gradient_inputs(dL_dZ, forwardContext.input_activations.shape)
    
    def weight_init_func(self):
        n = np.prod(self.layer_shape[1:])
        return he_init(np.zeros((self.layer_shape)), [n])
    
    def set_activ_func(self, func, dfunc):
        self.activ_func = func
        self.dactiv_func = dfunc
        return self
    
    
    def __getstate__(self) -> object:
        state = self.__dict__.copy()
        del state['activ_func']
        del state['dactiv_func']
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)

    
    def calc(self, v : np.ndarray) -> np.ndarray:
        raise NotImplementedError()
    
    def get_gradient_weights(self, dL_dO: np.ndarray, input_activ : np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def get_gradient_biases(self, dL_dO: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def get_gradient_inputs(self, dL_dO: np.ndarray, input_shape) -> np.ndarray:
        raise NotImplementedError