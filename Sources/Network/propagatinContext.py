import numpy as np

class PropagationContext():
    def __init__(self, input_activations : np.ndarray, output_activations : np.ndarray) -> None:
        self.input_activations : np.ndarray = input_activations
        self.output_activations : np.ndarray = output_activations