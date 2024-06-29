from Views.main import View
from Models.main import Model
from tkinter import Frame
from Database import database
from ..DataPreProccessor import DataPreProccessor
import numpy as np

class SignInController:
    def __init__(self, model : Model, view : View):
        self.model = model
        self.view = view
        self.frame : Frame = self.view.frames["signin"]

        self._bind()
    
    def _bind(self):
        self.frame.capture_btn.configure(command=self.start_capture)
        self.model.framescaptured.add_event_listener("all_images_added", self.attempt_sign_in)
        self.view.add_frame_chagne_listener("signin", self.start_camera_feed)
    
    def start_camera_feed(self):
        self.frame.video.update_image(self.view.root)

    def start_capture(self):
        self.model.framescaptured.start_capture(5, 5000)

    def attempt_sign_in(self):
        db = database()
        feature_vectors = []
        for im in self.model.framescaptured.images:
            feature_vectors.append(np.squeeze(self.model.network.feedforward(DataPreProccessor.from_camera(im)), axis=(0, 1)))
        user = db.find_match(feature_vectors) # None if not found
        self.model.user.set_user(user)