from .observableModel import ObservableModel
from Network.network import Network

class TrainerModel(ObservableModel):
    def __init__(self, network : Network):
        super().__init__()
        self.network = network
        self.costs = []
        self.train_existing_net : bool = False