from Models.main import Model
from Views.main import View
from Database import database
import numpy as np
#import dataPreProccessor

class SignUpController():
    def __init__(self, model : Model, view : View):
        self.model = model
        self.view = view
        self.frame = self.view.frames["signup"]

        self._bind()

    def _bind(self):
        self.view.add_frame_chagne_listener("signup", self.start_camera_feed)
        self.frame.capture_btn.configure(command=self.start_capture)
        self.model.framescaptured.add_event_listener("all_images_added", self.sign_up)

    def start_camera_feed(self):
        self.frame.video.update_image(self.view.root)

    def start_capture(self):
        self.model.framescaptured.start_capture(10, 5000)
    
    def sign_up(self):
        db = database()
        avg_feature_vector = np.zeros((1, 1, 128))
        #for im in self.model.framescaptured.images:
        #    avg_feature_vector += self.model.network.feedforward(dataPreProccessor.from_camera(im))
        avg_feature_vector /= len(self.model.framescaptured.images)
        name = "find a way to get the user's name"
        pic = "get a profile pic from the user"
        db.add_user(name, pic, avg_feature_vector)