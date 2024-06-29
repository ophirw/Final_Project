from tkinter import Tk

class Root(Tk):
    def __init__(self):
        super().__init__()

        start_width = int(self.winfo_screenwidth()*0.5)
        min_width = 400
        start_height = int(self.winfo_screenheight()*0.7)
        min_height = 250

        self.geometry(f"{start_width}x{start_height}")
        self.minsize(width=min_width, height=min_height)
        self.title("Facial Recognition Project, Ophir Wesley")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)