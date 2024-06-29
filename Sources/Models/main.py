from .user import User
from .framesCaptured import FramesCaptured
from Network.network import Network
from .trainerModel import TrainerModel

class Model:
    def __init__(self, network : Network):
        self.user = User()
        self.framescaptured = FramesCaptured()
        self.network = network
