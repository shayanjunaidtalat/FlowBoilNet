import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from torchvision.datasets import ImageFolder
import torch.multiprocessing
from torchvision import transforms as T
torch.multiprocessing.set_sharing_strategy('file_system')
import matplotlib.pyplot as plt
import pandas as pd


def add_value(dict_obj, key, value):
    ''' Adds a key-value pair to the dictionary.
        If the key already exists in the dictionary, 
        it will associate multiple values with that 
        key instead of overwritting its value'''
    if key not in dict_obj:
        dict_obj[key] = value
    elif isinstance(dict_obj[key], list):
        dict_obj[key].append(value)
    else:
        dict_obj[key] = [dict_obj[key], value]



class BubbleDataset(Dataset):
    def __init__(self, image_dir,transform=None):
        self.image_dir = image_dir
        #self.images = os.listdir(image_dir)
        self.dataset = ImageFolder(image_dir)
        self.images = [self.dataset.imgs[i][0] for i in range(len(self.dataset))]
        self.transform = transform
        self.image_list=[]
        self.labels=[]
        self.mydict = dict()


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        image_name = os.path.basename(self.images[index])

        subdirectories = [x[0] for x in os.walk(self.image_dir)][1:]
        image = np.array(self.dataset[index][0])
        y= self.dataset[index][1]
        if self.transform is not None:
            augmentations = self.transform(image=image)
            image = augmentations["image"]
        label = np.zeros((3,1))
        for i in range(len(subdirectories)):
            if os.path.exists((os.path.join(subdirectories[i], image_name))):
                label[i] = 1

        if image_name not in self.image_list:
            self.image_list.append(image_name)
            self.labels.append(label)
            add_value(self.mydict, image_name,label)
        else:
            add_value(self.mydict, image_name,label)
            list_of_duplicate_images = [key
            for key, list_of_values in self.mydict.items()
            if np.array(list_of_values).shape[1] > 1]
            duplicate_labels = self.mydict[list_of_duplicate_images[-1]]
            if np.all(duplicate_labels == duplicate_labels[0]) == False:
                print("Image", image_name, "has mulitple labels") 
        return image, label



class SlugDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("RGB"))
        label = np.array([[0] [1] [0]])

        if self.transform is not None:
            augmentations = self.transform(image=image)
            image = augmentations["image"]

        return image, label
