from Models.main import Model
from Views.main import View
from Controllers.main import Controller
from Network.network import Network
import numpy as np
from .Trainer import Trainer

def main():
    net = Network.loadNetwork()   # net is None if network isn't trained yet

    model = Model(net)
    view = View(model)
    controller = Controller(model, view)
    controller.start()

if __name__ == "__main__":
    main()