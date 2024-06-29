from tkinter import Frame, Canvas
import cv2
from Models.main import Model
import PIL.Image, PIL.ImageTk
from time import time


class CameraFeedView(Frame):
    def __init__(self, master, model : Model=None):
        super().__init__(master=master)

        self.model = model

        self.vid_source = cv2.VideoCapture(0)
        if not self.vid_source.isOpened():
            raise Exception("could not open camera")
        
        self.width = self.vid_source.get(3) #camera width
        self.height = self.vid_source.get(4) #camera height
        self.delay = 5

        self.canvas = Canvas(master=self, highlightthickness=1, highlightbackground="black", bg="gray", width=self.width, height=self.height)
        self.canvas.pack(expand=True, fill='both')
    
    def update_image(self, root):

        image = self.getImage(self.vid_source)

        self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
        self.canvas.create_image(0, 0, image = self.photo, anchor='nw') # show frame on screen
        
        data = self.model.framescaptured # shortening var name for convenience

        if (data.is_started and self.is_time_to_take_image(data)):
            data.add_image(image)

        if self.winfo_viewable() and (len(data.images) < data.requested_num_of_images or not data.is_started):
            root.after(self.delay, lambda: self.update_image(root))

    def getImage(self, capturer):
        ret, image = capturer.read()
        if not ret:
            raise Exception("Can't recieve image")
        return cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    def is_time_to_take_image(self, data_model) -> bool:
        return ((int(time()*1000)) >= 
                (data_model.start_time_ms + 
                 ((data_model.requested_capture_time_ms)*(len(data_model.images))/(data_model.requested_num_of_images-1))))   