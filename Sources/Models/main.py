from .user import User
from .framesCaptured import FramesCaptured
from Network.network import Network

class Model:
    def __init__(self, network : Network):
        self.user = User()
        self.framescaptured = FramesCaptured()

        self.network = network
