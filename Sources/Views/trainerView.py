from tkinter import Frame, Label, Button, Checkbutton
from Models.trainerModel import TrainerModel
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.ticker import MaxNLocator
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
        self.starttrain_buttn.grid(row=1, column=0)

        self.stoptrain_buttn = Button(self, text="stop training", state='disabled')
        self.stoptrain_buttn.grid(row=1, column=1)

        self.figure, self.ax = plt.subplots()
        self.ax.set_xlabel("mini-batch number")
        self.ax.set_ylabel("cost")
        self.graph = FigureCanvasTkAgg(self.figure, self)
        self.graph.get_tk_widget().grid(row=2, column=0, columnspan=3, pady=10)

        self.grid(row=0, column=0, sticky="nsew")

    def update_graph(self):
        self.ax.clear()

        self.ax.set_xlabel("mini-batch number")
        self.ax.set_ylabel("cost")

        y_data = self.model.costs
        x_data = range(len(y_data))

        self.ax.plot(x_data, y_data, marker='o')
        self.ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        xlim_high = 10 if max(x_data) < 10 else max(x_data)
        ylim_high = 0.7 if max(y_data) < 0.7 else max(y_data)

        self.ax.set_xlim(0, xlim_high)
        self.ax.set_ylim(0, ylim_high)

        self.graph.draw()
