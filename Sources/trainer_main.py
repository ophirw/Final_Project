from Trainer import Trainer
from Models.trainerModel import TrainerModel
from Views.trainerView import TrainerView
from Controllers.trainerController import TrainerController
from Network.network import Network
from Views.root import Root

def main():
    net = Network.loadNetwork()
    trainer = Trainer()

    root = Root()
    model = TrainerModel(net)
    view = TrainerView(root, model)
    controller = TrainerController(model, view, trainer)
    view.tkraise()
    root.mainloop()


if __name__ == '__main__':
    main()