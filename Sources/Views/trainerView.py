from tkinter import Frame, Label, Button, Checkbutton
from Models.trainerModel import TrainerModel

class TrainerView(Frame):
    def __init__(self, master, model=None):
        super().__init__(master=master)
        self.root = master
        self.model : TrainerModel = model

        self.header = Label(self, text="Network Trainer")
        self.header.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

        state = 'disabled' if self.model.network is None else 'normal'
        self.train_existing_checkbox = Checkbutton(self, text="train existing network", state=state)
        self.train_existing_checkbox.grid(row=0, column=2)

        self.starttrain_buttn = Button(self, text="Train")
        self.starttrain_buttn.grid(row=1, column=1)

        self.grid(row=0, column=0, sticky="nsew")