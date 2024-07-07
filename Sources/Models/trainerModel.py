from .observableModel import ObservableModel
from Network.network import Network

class TrainerModel(ObservableModel):
    def __init__(self, network : Network):
        super().__init__()
        self.network = network
        self.costs = []
        self.batches_learned = 0
        self.train_existing_net : bool = False
        self.stop_train : bool = False
    
    def add_cost(self, cost):
        self.costs.append(cost)
        self.trigger_event("added_cost")