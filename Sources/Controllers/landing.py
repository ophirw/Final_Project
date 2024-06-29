from Models.main import Model
from Views.main import View

class LandingController:
    def __init__(self, model : Model, view : View):
        self.model = model
        self.view = view
        self.frame = self.view.frames["landing"]
        self._bind()

    def _bind(self):
        self.frame.signin_btn.config(command=self.signin)
        self.frame.signup_btn.config(command=self.signup)

    def signin(self):
        self.view.switch("signin")
        
    def signup(self):
        self.view.switch("signup")
