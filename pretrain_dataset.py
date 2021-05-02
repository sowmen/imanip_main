import os
import numpy as np
import cv2

from torch.utils.data import Dataset
from albumentations import augmentations
from torchvision import transforms
from utils import get_ela


class DATASET(Dataset):
    def __init__(self, dataframe, mode, imgaug_augment=None,
                 transforms_normalize=None, geo_augment=None
    ):

        super().__init__()
        self.dataframe = dataframe
        self.mode = mode
        self.resize = 256
        self.imgaug_augment = imgaug_augment
        self.geo_augment = geo_augment
        self.transforms_normalize = transforms_normalize
        self.root_folder = "Image_Manipulation_Dataset/FODB/extracted_images"

        self.data = self.dataframe.values
        np.random.shuffle(self.data)

        print(f"{mode} -> {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):

        _, label, type_name, parameter, image_name = self.data[index]

        image_path = os.path.join(self.root_folder, image_name)

        if(not os.path.exists(image_path)):
            print(f"Image Not Found : {image_path}")

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ela_image = get_ela(image, 25)


        if self.imgaug_augment:
            try :
                image = self.imgaug_augment.augment_image(image)
            except Exception as e:
                print(image_path, e)
        
        if self.geo_augment:
            data = self.geo_augment(image=image, ela=ela_image)
            image = data["image"]
            ela_image = data["ela"]
        

        image = augmentations.geometric.functional.resize(image, self.resize, self.resize, cv2.INTER_AREA)
        ela_image = augmentations.geometric.functional.resize(ela_image, self.resize, self.resize, cv2.INTER_AREA)


        ##########------Normalize-----##########
        image_normalize = {
                'mean' : [0.4194640884489183, 0.4285149314597214, 0.4092076864902631],
                'std' : [0.2735709837194346, 0.2759520370791159, 0.2985429137632295]
        }
        # image_normalize = {
        #     "mean": [0.485, 0.456, 0.406],
        #     "std": [0.229, 0.224, 0.225],
        # }
        transNormalize = transforms.Normalize(mean=image_normalize['mean'], std=image_normalize['std'])
        transTensor = transforms.ToTensor()

        tensor_image = transTensor(image)
        tensor_ela = transTensor(ela_image)

        tensor_image = transNormalize(tensor_image)
        tensor_ela = transNormalize(tensor_ela)
        ########################################

        # if self.transforms_normalize:
        #     data = self.transforms_normalize(image=image, ela=ela_image)
        #     image = data["image"]
        #     ela_image = data["ela"]


        return {
            # "raw_image" : image,
            # "raw_ela" : ela_image,
            "image": tensor_image,
            "image_path" : image_path, 
            "label": label, 
            "ela" : tensor_ela,
            "type" : type_name+parameter
        }
