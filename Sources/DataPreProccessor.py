import PIL.Image
import numpy as np
from Network.globalVariables import mini_batch_size, image_size
import cv2
import random

class ImageRef():
    def __init__(self, name, index) -> None:
        self.name = name
        self.index = index

    def get_path(self):
        return f"lfw/{self.name}/{self.name}_{self.index:0>4}.jpg"

class DataPreProccessor():
    def __init__(self) -> None:
        self.PATH ="Data/"
        self.negativePairs: dict[str, tuple[int, list[ImageRef]]] = self.build_negative_dict()
        self.positivePairs: list[tuple[ImageRef]] = self.build_positive_pairs()
        random.shuffle(self.positivePairs)
        self.triplets : list[tuple[ImageRef]] = self.build_triplets()

    def get_image(self, im_ref : ImageRef):
        im = PIL.Image.open(self.PATH+im_ref.get_path())
        im_arr = np.array(im)  # im_arr is RGB
        im_arr = self.pre_proccess(im_arr, im_ref)
        return im_arr # is RGB
    
    @staticmethod
    def pre_proccess(arr : np.ndarray, im_ref : ImageRef):
        face_arr = DataPreProccessor.detectface(arr, im_ref)[0] # arr and face_arr are RGB
        if face_arr.shape[0] != face_arr.shape[1]:
            raise ValueError("cropped face isn't square")
        elif face_arr.shape[0] > 250:
            raise ValueError("face too close to camera")
        cropped_face_arr_CHW = cv2.resize(face_arr, image_size[:-3:-1]).transpose([2, 0, 1])
        return cropped_face_arr_CHW.reshape(image_size)/255.0
    
    @staticmethod
    def detectface(image : np.ndarray, im_ref : ImageRef) -> tuple[np.ndarray, np.ndarray]:
        # image is RGB
        BGRimage = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        grayscaleimage = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(grayscaleimage, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        faces = sorted(faces, key=lambda rect: rect[2] * rect[3], reverse=True)
        if (len(faces) == 0):
            raise ValueError("No face detected in image.")
        elif (len(faces) > 1):
            print(f"More than one face detected in image {im_ref.name}.") #TODO: consider raising exception for whoever has the im_ref to print the name of the image.
            #border_color = (0,0,255)
            #for face in faces:
            #    (x, y, w, h) = face
            #    cv2.rectangle(BGRimage, (x, y), (x+w, y+h), border_color, 2)
            #    border_color=(0,0,0)
            #cv2.imshow('More than one face in image detected', BGRimage)
            #cv2.waitKey(0)

        (x, y, w, h) = faces[0]
        cv2.rectangle(BGRimage, (x, y), (x+w, y+h), (0, 0, 0), 2)
        faceBGR = BGRimage[y:y+h, x:x+w]

        faceRGB = cv2.cvtColor(faceBGR, cv2.COLOR_BGR2RGB)
        boundedImageRGB = cv2.cvtColor(BGRimage, cv2.COLOR_BGR2RGB)

        return faceRGB, boundedImageRGB

    def get_training_batch(self, batch_num):
        batch = self.triplets[batch_num*mini_batch_size : (batch_num+1)*mini_batch_size]
        batch_array = []
        for i, triplet in enumerate(batch):
            batch_array.append(np.array([self.get_image(im_ref) for im_ref in triplet]))
        return np.array(batch_array)

    def build_triplets(self):
        triplets : list[tuple[ImageRef]] = []

        for posPair in self.positivePairs:
            try:
                nextNegative, negativesList = self.negativePairs[posPair[0].name]
                negative = negativesList[nextNegative]
                new_nextNegative = (self.negativePairs[posPair[0].name][0]+1)%len(negativesList)
                self.negativePairs[posPair[0].name] = (new_nextNegative, negativesList)
            except KeyError:
                name = posPair[0].name
                while (name == posPair[0].name):
                    i = np.random.randint(0, len(self.positivePairs))
                    name = self.positivePairs[i][0].name
                negative = self.positivePairs[i][0]
            triplets.append(posPair + (negative,))
        return triplets

    def build_negative_dict(self):
        with open(self.PATH+"negativePairsTrain.txt", 'r') as file:
            negativePairs : list[str] = file.readlines()
        negativePairs = [line.strip() for line in negativePairs]

        negative_dict : dict[str, tuple[int, list[ImageRef]]]= {}
        for pair in negativePairs:
            refs = [tuple(person.split(' ')) for person in pair.split('\t')]
            people = [ImageRef(name_index[0], name_index[1]) for name_index in refs]
    
            if people[0].name not in negative_dict.keys():
                negative_dict[people[0].name] = (0, [])
            if people[1].name not in negative_dict.keys():
                negative_dict[people[1].name] = (0, [])
            
            negative_dict[people[0].name][1].append(people[1])
            negative_dict[people[1].name][1].append(people[0])
        return negative_dict
    
    def build_positive_pairs(self):
        with open(self.PATH+"positivePairsTrain.txt", 'r') as file:
            positivePairs = file.readlines()
        positivePairs = [line.strip() for line in positivePairs]

        positive_pairs_img_refs : list[tuple[ImageRef]] = []
        for pair in positivePairs:
            pair_split = pair.split('\t')
            positive_pairs_img_refs.append((ImageRef(pair_split[0], pair_split[1]), ImageRef(pair_split[0], pair_split[2])))
        
        return positive_pairs_img_refs