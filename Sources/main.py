from Models.main import Model
from Views.main import View
from Controllers.main import Controller
from Network.network import Network
import numpy as np

def main():
    net = Network.loadNetwork()   # net is None if network isn't trained yet

    model = Model(net)
    view = View(model)
    controller = Controller(model, view)
    controller.start()

if __name__ == "__main__":
    #main()
    print("start")
    input = np.full((1, 1, 3, 250, 250), 1)
    net = Network(input.shape)
    print(np.squeeze(net.feedforward(input), (0, 1)))
    #print("back")
    #net.backprop_step(input)
