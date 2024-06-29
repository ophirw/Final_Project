from tkinter import Frame, Label, Button
from .camerafeed import CameraFeedView

class SignUpView(Frame):
    def __init__(self, master, model=None):
        super().__init__(master=master)

        self.header = Label(self, text="Sign Up by clicking the 'capture video' button.\nPlease make sure you are well lit and in front of the camera.")
        self.header.grid(row=0, column=0)

        self.video = CameraFeedView(self, model)
        self.video.grid(row=1, column=0)

        self.capture_btn = Button(self, text="capture video")
        self.capture_btn.grid(row=2, column=0)