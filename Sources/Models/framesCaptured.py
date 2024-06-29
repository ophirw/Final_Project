from .observableModel import ObservableModel
from time import time

class FramesCaptured(ObservableModel):
    def __init__(self):
        super().__init__()
        
        self.images = []
        self.requested_num_of_images : int = 0
        self.requested_capture_time_ms : int = 0
        self.start_time_ms = None
        self.is_started = False
    
    def add_image(self, image):
        self.images.append(image)
        self.trigger_event("image_added")
        print("image added, count = ", len(self.images))
        if len(self.images) >= self.requested_num_of_images:
            self.trigger_event("all_images_added")
            self.is_started = False
    
    def start_capture(self, num_of_images, capture_time_ms):
        self.images = []
        self.requested_num_of_images = num_of_images
        self.requested_capture_time_ms = capture_time_ms
        self.start_time_ms = int(time()*1000)
        self.is_started = True

        self.trigger_event("start_capture")