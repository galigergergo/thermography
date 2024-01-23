#!/usr/bin/env python
# coding: utf-8


from torch.utils.data import Dataset
from pathlib import Path
import os
import numpy as np
import torch
from skimage import io
import random
from torchvision import transforms


class ThermUnetDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_transform=None, mask_transform=None,
                 start=None, stop=None, augment=False,
                prefixes=["V_true_", "measurements_"]):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        mask_full_path = self.mask_dir #todo: not needed anymore

        self.mask_file_list = [f for f in os.listdir(mask_full_path) if os.path.isfile(os.path.join(mask_full_path, f))]
        if(start != None and stop != None):
            self.mask_file_list = self.mask_file_list[start:stop]
            
        # generate image file list based on mask files in order to avoid confusion btw. both    
        self.img_file_list = [f.rsplit('.', 1)[0].replace(prefixes[0], prefixes[1])+'.png' for f in self.mask_file_list]

        # add full path to the filenames
        self.mask_file_list = [os.path.join(self.mask_dir, f) for f in self.mask_file_list]
        self.img_file_list = [os.path.join(self.image_dir, f) for f in self.img_file_list]
            
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        
        self.augment = augment
        
        self.prefixes = prefixes

        self.len = len(self.mask_file_list)
        

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
            
        img_name = self.img_file_list[index]
        mask_name = self.mask_file_list[index]

        image = io.imread(img_name)
        mask = io.imread(mask_name)   

        # todo: handle via external transform
        # Random horizontal flipping as simple augmentation technique
        if(self.augment):
            if random.random() > 0.5:
                image = np.fliplr(image)
                mask = np.fliplr(mask)

        if self.img_transform:
            image = self.img_transform(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)

        sample = {'image': image, 'mask': mask, 'image_file': img_name, 'mask_file': mask_name}

        return sample

    
    def __len__(self):
        return self.len
    

# in the real world we don't have masks available! 
class InferenceDataset(Dataset):
    def __init__(self, image_dir, root_dir=None, img_transform=None, start=None, stop=None):            
        self.image_dir = Path(image_dir)
        
        if(root_dir == None):
            self.dataset_path = os.path.dirname(image_dir)
        else:
            self.dataset_path = Path(root_dir)
        
        img_full_path = os.path.join(self.dataset_path, self.image_dir)

        self.img_file_list = [f for f in os.listdir(img_full_path) if os.path.isfile(os.path.join(img_full_path, f))]
        if(start != None and stop != None):
            self.img_file_list = self.img_file_list[start:stop]

        self.img_transform = img_transform

        self.len = len(self.img_file_list)
        

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
            
        file_name =  self.img_file_list[index]
        img_name = os.path.join(self.dataset_path, self.image_dir, file_name)

        image = io.imread(img_name)    

        if self.img_transform:
            image = self.img_transform(image)

        sample = {'image': image, 'image_file': img_name}

        return sample



    def __len__(self):
        return self.len
