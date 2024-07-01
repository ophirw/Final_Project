from Network.network import Network
from Network.globalVariables import mini_batch_size
import gc
from DataPreProccessor import DataPreProccessor
import threading
from Models.trainerModel import TrainerModel

class Trainer():
    def __init__(self, model : TrainerModel) -> None:
        self.data = DataPreProccessor()
        self.model = model

    def Train(self, net : Network, epochs : int):
        thread = threading.Thread(target=self.train_threaded, args=(net, epochs))
        thread.start()

    def train_threaded(self, net : Network, epochs : int):
        backprop_step_number = 0

        for i in range(epochs):
            for j in range(len(self.data.triplets)//mini_batch_size):

                if self.model.stop_train:
                    net.saveNetwork()
                    return
                
                backprop_step_number += 1

                training_data_batch = self.data.get_training_batch(j)
                print(f"starting batch number {j}")
                cost_before_backprop = net.backprop_step(training_data_batch, backprop_step_number)

                self.model.costs.append(cost_before_backprop)
                print(cost_before_backprop)

                del training_data_batch
                gc.collect()
    

        net.saveNetwork()