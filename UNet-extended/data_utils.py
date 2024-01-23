#!/usr/bin/env python
# coding: utf-8


import os
from pathlib import Path
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.utils.data import ConcatDataset

from transforms import *
from dataset import *


# generates the location of the results given the model file and the location of the input files for inference
def get_result_path(model_file, input_path):
    filecomponents = input_path.split(os.sep)
        
    tmp_path = model_file.replace('.pth', '')

    # combine with prefix and remove ':' from the path
    num_filecomponents = len(filecomponents)
    for comp in range(num_filecomponents):
        tmp_path = os.path.join(tmp_path, filecomponents[comp].replace(":", ""))

    return tmp_path


def get_pipelines(normalizer):
    img_transform_pipeline = Compose([
                ToFloat32(),
                ToTensor(),
                Normalize(mean=normalizer['norm_mean'], std=normalizer['norm_std']),
                ])

    mask_transform_pipeline = Compose([
                ToUint8(),
                ToTensor(),
                ])
    
    return img_transform_pipeline, mask_transform_pipeline


def load_e2e_data(train_dir, mask_dir, start, stop, normalizer, augment=False):
    return load_data(train_dir, mask_dir, start, stop, normalizer, augment=augment, prefixes=["V_true_", "measurements_"])


def load_hybrid_data(train_dir, mask_dir, start, stop, normalizer, augment=False):
    return load_data(train_dir, mask_dir, start, stop, normalizer, augment=augment, prefixes=["V_true_", "T_virt_abel_"])


def load_data(train_dir, mask_dir, start, stop, normalizer, augment=False, prefixes=["V_true_", "measurements_"]):
    img_transform_pipeline, mask_transform_pipeline = get_pipelines(normalizer)
    
    image_dir = os.path.join(train_dir, "SNR_70")
    data_SNR_70 = ThermUnetDataset(image_dir, mask_dir, 
                                         img_transform=img_transform_pipeline,
                                         mask_transform=mask_transform_pipeline,
                                         start=start, stop=stop,
                                         augment=augment,
                                         prefixes=prefixes)

    image_dir = os.path.join(train_dir, "SNR_60")
    data_SNR_60 = ThermUnetDataset(image_dir, mask_dir, 
                                         img_transform=img_transform_pipeline,
                                         mask_transform=mask_transform_pipeline,
                                         start=start, stop=stop,
                                         augment=augment,
                                         prefixes=prefixes)

    image_dir = os.path.join(train_dir, "SNR_50")
    data_SNR_50 = ThermUnetDataset(image_dir, mask_dir, 
                                         img_transform=img_transform_pipeline,
                                         mask_transform=mask_transform_pipeline,
                                         start=start, stop=stop,
                                         augment=augment,
                                         prefixes=prefixes)

    image_dir = os.path.join(train_dir, "SNR_40")
    data_SNR_40 = ThermUnetDataset(image_dir, mask_dir, 
                                         img_transform=img_transform_pipeline,
                                         mask_transform=mask_transform_pipeline,
                                         start=start, stop=stop,
                                         augment=augment,
                                         prefixes=prefixes)

    image_dir = os.path.join(train_dir, "SNR_30")
    data_SNR_30 = ThermUnetDataset(image_dir, mask_dir, 
                                         img_transform=img_transform_pipeline,
                                         mask_transform=mask_transform_pipeline,
                                         start=start, stop=stop,
                                         augment=augment,
                                         prefixes=prefixes)

    image_dir = os.path.join(train_dir, "SNR_20")
    data_SNR_20 = ThermUnetDataset(image_dir, mask_dir, 
                                         img_transform=img_transform_pipeline,
                                         mask_transform=mask_transform_pipeline,
                                         start=start, stop=stop,
                                         augment=augment,
                                         prefixes=prefixes)

    image_dir = os.path.join(train_dir, "SNR_10")
    data_SNR_10 = ThermUnetDataset(image_dir, mask_dir, 
                                         img_transform=img_transform_pipeline,
                                         mask_transform=mask_transform_pipeline,
                                         start=start, stop=stop,
                                         augment=augment,
                                         prefixes=prefixes)

    image_dir = os.path.join(train_dir, "SNR_0")
    data_SNR_0 = ThermUnetDataset(image_dir, mask_dir, 
                                         img_transform=img_transform_pipeline,
                                         mask_transform=mask_transform_pipeline,
                                         start=start, stop=stop,
                                         augment=augment,
                                         prefixes=prefixes)

    image_dir = os.path.join(train_dir, "SNR_-10")
    data_SNR_m10 = ThermUnetDataset(image_dir, mask_dir, 
                                         img_transform=img_transform_pipeline,
                                         mask_transform=mask_transform_pipeline,
                                         start=start, stop=stop,
                                         augment=augment,
                                         prefixes=prefixes)

    image_dir = os.path.join(train_dir, "SNR_-20")
    data_SNR_m20 = ThermUnetDataset(image_dir, mask_dir, 
                                         img_transform=img_transform_pipeline,
                                         mask_transform=mask_transform_pipeline,
                                         start=start, stop=stop,
                                         augment=augment,
                                         prefixes=prefixes)
    
    return ConcatDataset([data_SNR_70, data_SNR_60,
                           data_SNR_50, data_SNR_40,
                           data_SNR_30, data_SNR_20,
                           data_SNR_10, data_SNR_0,
                           data_SNR_m10, data_SNR_m20,])


def load_inference_data(input_dir, normalizer, start=None, stop=None):
    img_transform_pipeline,_ = get_pipelines(normalizer)
    dataset = InferenceDataset(image_dir=input_dir, img_transform=img_transform_pipeline,
                              start=start, stop=stop)
    return dataset


def load_inference_test_data(input_dir, normalizer, start=None, stop=None):
    img_transform_pipeline,_ = get_pipelines(normalizer)
    
    image_dir = os.path.join(input_dir, "SNR_70")
    data_SNR_70 = InferenceDataset(image_dir=image_dir, img_transform=img_transform_pipeline,
                              start=start, stop=stop)
    image_dir = os.path.join(input_dir, "SNR_60")
    data_SNR_60 = InferenceDataset(image_dir=image_dir, img_transform=img_transform_pipeline,
                              start=start, stop=stop)
    image_dir = os.path.join(input_dir, "SNR_50")
    data_SNR_50 = InferenceDataset(image_dir=image_dir, img_transform=img_transform_pipeline,
                              start=start, stop=stop)
    image_dir = os.path.join(input_dir, "SNR_40")
    data_SNR_40 = InferenceDataset(image_dir=image_dir, img_transform=img_transform_pipeline,
                              start=start, stop=stop)
    image_dir = os.path.join(input_dir, "SNR_30")
    data_SNR_30 = InferenceDataset(image_dir=image_dir, img_transform=img_transform_pipeline,
                              start=start, stop=stop)
    image_dir = os.path.join(input_dir, "SNR_20")
    data_SNR_20 = InferenceDataset(image_dir=image_dir, img_transform=img_transform_pipeline,
                              start=start, stop=stop)
    image_dir = os.path.join(input_dir, "SNR_10")
    data_SNR_10 = InferenceDataset(image_dir=image_dir, img_transform=img_transform_pipeline,
                              start=start, stop=stop)
    image_dir = os.path.join(input_dir, "SNR_0")
    data_SNR_0 = InferenceDataset(image_dir=image_dir, img_transform=img_transform_pipeline,
                              start=start, stop=stop)
    image_dir = os.path.join(input_dir, "SNR_-10")
    data_SNR_m10 = InferenceDataset(image_dir=image_dir, img_transform=img_transform_pipeline,
                              start=start, stop=stop)
    image_dir = os.path.join(input_dir, "SNR_-20")
    data_SNR_m20 = InferenceDataset(image_dir=image_dir, img_transform=img_transform_pipeline,
                              start=start, stop=stop)

    # full data set with 10 different SNRs
    dataset = ConcatDataset([data_SNR_70, data_SNR_60,
                            data_SNR_50, data_SNR_40,
                            data_SNR_30, data_SNR_20,
                            data_SNR_10, data_SNR_0,
                            data_SNR_m10, data_SNR_m20,])
    
    return dataset


def get_eval_pipelines():
    img_transform_pipeline = Compose([
                DivideBy255(),
                ToFloat32(),
                ToTensor(),
                ])

    mask_transform_pipeline = img_transform_pipeline
    
    return img_transform_pipeline, mask_transform_pipeline


def load_results(result_dir, mask_dir, prefixes):
    img_transform_pipeline, mask_transform_pipeline = get_eval_pipelines()
    
    image_dir = os.path.join(result_dir, "SNR_70")
    data_SNR_70 = ThermUnetDataset(image_dir, mask_dir,
                                        img_transform=img_transform_pipeline,
                                        mask_transform=mask_transform_pipeline,
                                        prefixes=prefixes)

    image_dir = os.path.join(result_dir, "SNR_60")
    data_SNR_60 = ThermUnetDataset(image_dir, mask_dir,
                                        img_transform=img_transform_pipeline,
                                        mask_transform=mask_transform_pipeline,
                                        prefixes=prefixes)

    image_dir = os.path.join(result_dir, "SNR_50")
    data_SNR_50 = ThermUnetDataset(image_dir, mask_dir,
                                        img_transform=img_transform_pipeline,
                                        mask_transform=mask_transform_pipeline,
                                        prefixes=prefixes)

    image_dir = os.path.join(result_dir, "SNR_40")
    data_SNR_40 = ThermUnetDataset(image_dir, mask_dir,
                                        img_transform=img_transform_pipeline,
                                        mask_transform=mask_transform_pipeline,
                                        prefixes=prefixes)

    image_dir = os.path.join(result_dir, "SNR_30")
    data_SNR_30 = ThermUnetDataset(image_dir, mask_dir,
                                        img_transform=img_transform_pipeline,
                                        mask_transform=mask_transform_pipeline,
                                        prefixes=prefixes)

    image_dir = os.path.join(result_dir, "SNR_20")
    data_SNR_20 = ThermUnetDataset(image_dir, mask_dir,
                                        img_transform=img_transform_pipeline,
                                        mask_transform=mask_transform_pipeline,
                                        prefixes=prefixes)

    image_dir = os.path.join(result_dir, "SNR_10")
    data_SNR_10 = ThermUnetDataset(image_dir, mask_dir,
                                        img_transform=img_transform_pipeline,
                                        mask_transform=mask_transform_pipeline,
                                        prefixes=prefixes)

    image_dir = os.path.join(result_dir, "SNR_0")
    data_SNR_0 = ThermUnetDataset(image_dir, mask_dir,
                                        img_transform=img_transform_pipeline,
                                        mask_transform=mask_transform_pipeline,
                                        prefixes=prefixes)

    image_dir = os.path.join(result_dir, "SNR_-10")
    data_SNR_m10 = ThermUnetDataset(image_dir, mask_dir,
                                        img_transform=img_transform_pipeline,
                                        mask_transform=mask_transform_pipeline,
                                        prefixes=prefixes)

    image_dir = os.path.join(result_dir, "SNR_-20")
    data_SNR_m20 = ThermUnetDataset(image_dir, mask_dir,
                                        img_transform=img_transform_pipeline,
                                        mask_transform=mask_transform_pipeline,
                                        prefixes=prefixes)

    all_results = {"SNR_70": data_SNR_70,
                  "SNR_60": data_SNR_60,
                  "SNR_50": data_SNR_50,
                  "SNR_40": data_SNR_40,
                  "SNR_30": data_SNR_30,
                  "SNR_20": data_SNR_20,
                  "SNR_10": data_SNR_10,
                  "SNR_0": data_SNR_0,
                  "SNR_-10": data_SNR_m10,
                  "SNR_-20": data_SNR_m20,
                  }
    
    return all_results


def load_realworld_data(input_dir, normalizer, theta_dir="Deg_0", start=None, stop=None):
    img_transform_pipeline,_ = get_pipelines(normalizer)
    
    image_dir = os.path.join(input_dir, theta_dir)
    data_deg_0 = InferenceDataset(image_dir=image_dir, img_transform=img_transform_pipeline,
                              start=start, stop=stop)
    #image_dir = os.path.join(input_dir, "Deg_10")
    #data_deg_10 = InferenceDataset(image_dir=image_dir, img_transform=img_transform_pipeline,
    #                          start=start, stop=stop)
    #image_dir = os.path.join(input_dir, "Deg_25")
    #data_deg_25 = InferenceDataset(image_dir=image_dir, img_transform=img_transform_pipeline,
    #                          start=start, stop=stop)
    #image_dir = os.path.join(input_dir, "Deg_45")
    #data_deg_45 = InferenceDataset(image_dir=image_dir, img_transform=img_transform_pipeline,
    #                          start=start, stop=stop)

    # full data set with 4 different degrees of rotation
    dataset = ConcatDataset([data_deg_0#, data_deg_10,
                            #data_deg_25, data_deg_45,
    ])
    
    return dataset
