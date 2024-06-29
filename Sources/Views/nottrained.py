from tkinter import Frame, Label, Button

class NotTrainedView(Frame):
    def __init__(self, master, model=None):
        super().__init__(master=master)

        self.header = Label(self, text="No model currently trained. Please use trainer before attempting to use system.")
        self.header.grid(row=0, column=0, columnspan=2, padx=10, pady=10)