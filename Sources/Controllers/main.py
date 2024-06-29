from .landing import LandingController
from .signin import SignInController
from .signup import SignUpController
from .home import HomeController
from Views.main import View
from Models.main import Model

class Controller:
    def __init__(self, model: Model, view: View):
        self.view = view
        self.model = model
        self.landing_controller = LandingController(model, view)
        self.signin_controller = SignInController(model, view)
        self.signup_controller = SignUpController(model, view)
        self.home_controller = HomeController(model, view)

    def start(self):
        if self.model.network is not None:
            self.view.switch("landing")
        else:
            self.view.switch("nottrained")
        self.view.start_mainloop()