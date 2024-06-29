from Models.main import Model
from Views.main import View
from Views.home import HomeView
from Database import database

class HomeController():
    def __init__(self, model : Model, view : View):
        self.model = model
        self.view = view
        self.frame : HomeView = self.view.frames["home"]
        self._bind()
    
    def _bind(self):
        self.view.add_frame_chagne_listener("home", self.frame.load)
        self.model.user.add_event_listener(
            "user_changed", self.user_state_listener
        )

    def user_state_listener(self):
        if self.model.user.user:
            self.update_view()
            self.view.switch("home")
        else:
            self.view.switch("landing")

    def update_view(self):
        db = database()
        self.frame.profile_picture = db.get_image(self.model.user.user)
        self.frame.header.configure(text=f"Welcome{self.model.user.user}.")