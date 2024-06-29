import PIL.Image
import numpy as np
from Network.globalVariables import mini_batch_size, image_size
from cv2.typing import MatLike
import random

class ImageRef():
    def __init__(self, name, index) -> None:
        self.name = name
        self.index = index

    def get_path(self):
        return f"lfw/{self.name}/{self.name}_{self.index:0>4}.jpg"

class DataPreProccessor():
    def __init__(self) -> None:
        self.PATH ="C:/Final_Project/Data/"
        self.negativePairs: dict[str, tuple[int, list[ImageRef]]] = self.build_negative_dict()
        self.positivePairs: list[tuple[ImageRef]] = self.build_positive_pairs()
        random.shuffle(self.positivePairs)
        self.triplets : list[tuple[ImageRef]] = self.build_triplets()

    def get_image(self, im_ref : ImageRef):
        im = PIL.Image.open(self.PATH+im_ref.get_path())
        im_arr = np.array(im).transpose([2, 0, 1])
        im_arr = self.pre_proccess(im_arr)
        return im_arr
    
    def get_training_batch(self, batch_num):
        batch = self.triplets[batch_num*mini_batch_size : (batch_num+1)*mini_batch_size]
        batch_array = []
        for i, triplet in enumerate(batch):
            batch_array.append(np.array([self.get_image(im_ref) for im_ref in triplet]))
        return np.array(batch_array)
    
    def from_camera(self, image : MatLike):
        arr = self.pre_proccess(image)
        return arr

    def pre_proccess(self, arr : np.ndarray):
        return arr.reshape(image_size)

    def build_triplets(self):
        triplets : list[tuple[ImageRef]] = []

        for index, posPair in enumerate(self.positivePairs):
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