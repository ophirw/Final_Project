from Network.network import Network
from Network.globalVariables import mini_batch_size
import gc
from DataPreProccessor import DataPreProccessor
import threading

class Trainer():
    def __init__(self) -> None:
        self.data = DataPreProccessor()

    def Train(self, net : Network, epochs : int):
        thread = threading.Thread(target=self.train_threaded, args=(net, epochs))
        thread.run()

    def train_threaded(self, net : Network, epochs : int):
        training_costs = []
        backprop_step_number = 0
        for i in range(epochs):
            for j in range(len(self.data.triplets)//mini_batch_size):
                backprop_step_number += 1
                training_data_batch = self.data.get_training_batch(j)
                print(f"starting batch number {j}")
                cost_before_backprop = net.backprop_step(training_data_batch, backprop_step_number)
                print(f"cost from last batch is: {cost_before_backprop}")
                training_costs.append((backprop_step_number, cost_before_backprop))
                del training_data_batch
                gc.collect()

        net.saveNetwork()