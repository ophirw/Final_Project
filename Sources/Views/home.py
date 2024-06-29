from tkinter import Frame, Label, Canvas
from Models.main import Model
import PIL.Image, PIL.ImageTk

class HomeView(Frame):
    def __init__(self, master, model : Model=None):
        super().__init__(master=master)

        self.model = model
        self.profile_picture = None
        self.header = Label(self)
       


    def load(self):
        pass ### place the header and the profile_pic on-screen