import numpy as np
from Network.network import Network
from Network.globalVariables import epoch_number
import gc

class Trainer():
    def __init__(self, input_shape) -> None:
        self.net = Network(input_shape)

    def Train(self, net : Network, epochs : int):
        global epoch_number
        epoch_number = 0

        costs = []
        accuracy = 0
        for i in range(epoch_number):
            training_data_batch = DataPreProccessor.get_training_batch()
            net.backprop_step(training_data_batch)
            del training_data_batch
            gc.collect()