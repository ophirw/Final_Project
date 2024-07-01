from Models.trainerModel import TrainerModel
from Views.trainerView import TrainerView
from Trainer import Trainer
from Network.network import Network
from Network.globalVariables import *

class TrainerController:
    def __init__(self, model : TrainerModel, view : TrainerView, trainer : Trainer):
        self.model = model
        self.view = view
        self.trainer = trainer
        self._bind()

    def _bind(self):
        self.view.train_existing_checkbox.config(variable=self.model.train_existing_net)
        self.view.starttrain_buttn.config(command=self.start_training)
        self.view.stoptrain_buttn.config(command=self.stop_training)

    def start_training(self):
        if not self.model.train_existing_net or self.model.network is None:
            self.model.network = Network((mini_batch_size, 3) + image_size)

        self.view.stoptrain_buttn.config(state='normal')
        self.trainer.Train(self.model.network, amount_of_epochs)

    def stop_training(self):
        self.model.stop_train = True
