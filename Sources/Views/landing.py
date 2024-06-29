from tkinter import Frame, Label, Button

class LandingView(Frame):
    def __init__(self, master, model=None):
        super().__init__(master=master)

        self.header = Label(self, text="Sign In or Sign Up")
        self.header.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

        self.signin_btn = Button(self, text="Sign In")
        self.signin_btn.grid(row=1, column=0, sticky="w")

        self.signup_btn = Button(self, text="Sign Up")
        self.signup_btn.grid(row=1, column=1, sticky="w")