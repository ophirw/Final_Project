from .root import Root
from .landing import LandingView
from .signup import SignUpView
from .signin import SignInView
from .home import HomeView
from .nottrained import NotTrainedView
from Models.main import Model

class View:
    def __init__(self, model: Model):
        self.root = Root()
        self.frames = {}

        self.frame_chagne_listeners = {}

        self._add_frame(LandingView, "landing")
        self._add_frame(SignUpView, "signup", model)
        self._add_frame(SignInView, "signin", model)
        self._add_frame(HomeView, "home", model)
        self._add_frame(NotTrainedView, "nottrained")

    def _add_frame(self, Frame, name, model=None):
        self.frames[name] = Frame(self.root, model)
        self.frames[name].grid(row=0, column=0, sticky="nsew")

    def add_frame_chagne_listener(self, frame_name, func):
        try:
            self.frame_chagne_listeners[frame_name].append(func)
        except KeyError:
            self.frame_chagne_listeners[frame_name] = [func]

        return lambda: self.frame_chagne_listeners[frame_name].remove(func)

    def trigger_frame_change(self, frame_name):
        if frame_name not in self.frame_chagne_listeners.keys():
            return

        for func in self.frame_chagne_listeners[frame_name]:
            func()

    def switch(self, name):
        frame = self.frames[name]
        frame.tkraise()
        self.trigger_frame_change(name)

    def start_mainloop(self):
        self.root.mainloop()


