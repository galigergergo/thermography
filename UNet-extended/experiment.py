import os
from os import makedirs
import random
import numpy as np
import skimage 
from skimage import io
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize

import matplotlib.pyplot as plt

from thermunet import ThermUNet
from dataset import ThermUnetDataset
from model_utils import save_checkpoint, load_checkpoint, get_normalizer, compile_model
from data_utils import load_inference_test_data, load_realworld_data



def get_thermunet(depth, wf):
    return ThermUNet(depth=depth, wf=wf)


def get_cmp_thermunet():
    return get_thermunet(3, 4)
    
    
def get_lrg_thermunet():
    return get_thermunet(5, 4)


def get_optimizer(model, lr=0.001, weight_decay=0.0, amsgrad=True):    
    optimizer = torch.optim.Adam(model.parameters(), amsgrad=amsgrad,
                                 weight_decay=weight_decay, lr=lr)
        
    return optimizer


# todo: move to Dataset class?
def extract_normalizer(dataset):
    mean = -1
    std = -1
    for tfs in dataset.datasets[0].__getattribute__('img_transform').transforms:
        if isinstance(tfs, Normalize):
            mean = tfs.mean
            std = tfs.std
            
    assert mean is not -1 and std is not -1, 'Normalize transform not detected!'
    
    normalizer = {
        'norm_mean': mean,
        'norm_std': std,
        }

    return normalizer


def get_dataloader(dataset, batch_size=10, shuffle=True, pin_memory=True, num_workers=0):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                             num_workers=num_workers, pin_memory=pin_memory)
    
    return data_loader
    
    
def train(model, train_dataset, val_dataset, epochs=100, visualization_lvl=0):
    # this is just used to be stored along the model within a checkpoint
    normalizer = extract_normalizer(train_dataset)

    # used to put data on the same device as the model
    device = next(model.parameters()).device
                  
    # create result folder    
    dateTimeObj = datetime.now()
    uid = '{}-{}-{}_{}-{}-{}.{}'.format(dateTimeObj.year, dateTimeObj.month, dateTimeObj.day,
                                        dateTimeObj.hour, dateTimeObj.minute, dateTimeObj.second,
                                        dateTimeObj.microsecond)
    
    train_dir = train_dataset.datasets[0].__getattribute__('image_dir').parent
    model_path = os.path.join(train_dir, uid, "models")
    if(os.path.isdir(model_path)==False):
        try:
            os.makedirs(model_path)
        except OSError:
            print ("Creation of the directory %s failed" % model_path)
        else:
            print ("Successfully created the directory %s " % model_path)

    train_dataset_file = os.path.join(train_dir, uid, "train_data.pth")
    torch.save({
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            }, train_dataset_file)
    
    # used to minimize...
    optimizer = get_optimizer(model)
    # ...the loss
    criterion = torch.nn.BCEWithLogitsLoss()

    # make data iterable
    train_loader = get_dataloader(train_dataset)
    val_loader = get_dataloader(val_dataset)
        
    # used to keep track of the best model so far
    best_val_loss = np.Inf
    best_epoch = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        val_epoch_loss = 0.0
        #=====================================================================================
        # BEGIN: validation
        #=====================================================================================            
        for step, b in enumerate(val_loader):        
            optimizer.zero_grad()

            X = b['image'].to(device)  # [N, 1, H, W]
            y = b['mask'].to(device)  # [N, H, W]

            prediction = model(X)  # [N, nb_classes, H, W]

            #==========================================================================
            # BEGIN: visualization of some samples from the validation data
            #==========================================================================            
            if visualization_lvl >= 1:
                if step==0 and (epoch%5)==0:
                    print("epoch ",epoch)
                    prediction_sigmoid = torch.sigmoid(prediction.detach())
                    test = prediction_sigmoid.cpu().detach().numpy()
                    input_batch = X.cpu().detach().numpy()
                    target_batch = y.cpu().detach().numpy()

                    fig = plt.figure(figsize=(12,18))
                    num_examples = 8

                    # we need to limit the range to avoid some plots not showing up...
                    for i in range(num_examples-2):
                        idx = (i+1)*3-2

                        input_sample = input_batch[i]

                        ax = plt.subplot(num_examples, 3, idx)
                        plt.tight_layout()
                        ax.set_title('val input #{}'.format(i))
                        ax.axis('off')
                        plt.imshow(input_sample.reshape(64, 256))
                        plt.subplots_adjust(wspace=None, hspace=None)

                        sample = test[i]

                        ax = plt.subplot(num_examples, 3, idx+1)
                        plt.tight_layout()
                        ax.set_title('val output #{}'.format(i))
                        ax.axis('off')
                        plt.imshow(sample.reshape(64, 256))
                        plt.subplots_adjust(wspace=None, hspace=None)

                        sample = target_batch[i]

                        ax = plt.subplot(num_examples, 3, idx+2)
                        plt.tight_layout()
                        ax.set_title('val target #{}'.format(i))
                        ax.axis('off')
                        plt.imshow(sample.reshape(64, 256))
                        plt.subplots_adjust(wspace=None, hspace=None)

                plt.show()
                #======================================================================
                # END: visualization of some samples from the validation data
                #======================================================================            

            val_loss = criterion(prediction, y)
            val_epoch_loss += prediction.shape[0] * val_loss.item()
        #=====================================================================================
        # END: validation
        #=====================================================================================        

            
        #=====================================================================================
        # BEGIN: actual training (adapting weights/parameters)
        #=====================================================================================            
        for step, b in enumerate(train_loader):        
            optimizer.zero_grad()

            X = b['image'].to(device)  # [N, 1, H, W]
            y = b['mask'].to(device)  # [N, H, W]

            prediction = model(X)  # [N, nb_classes, H, W]

            #=============================================================================
            # BEGIN: visualization of some samples from the training data
            #=============================================================================           
            if visualization_lvl >= 2:
                if step==0 and (epoch%5)==0:
                    print("epoch ",epoch)
                    prediction_sigmoid = torch.sigmoid(prediction.detach())
                    test = prediction_sigmoid.cpu().detach().numpy()
                    input_batch = X.cpu().detach().numpy()
                    target_batch = y.cpu().detach().numpy()
                    fig = plt.figure(figsize=(12,18))
                    num_examples = 8

                    # we need to limit the range to avoid some plots not showing up...
                    for i in range(num_examples-2):
                        idx = (i+1)*3-2

                        input_sample = input_batch[i]

                        ax = plt.subplot(num_examples, 3, idx)
                        plt.tight_layout()
                        ax.set_title('train input #{}'.format(i))
                        ax.axis('off')
                        plt.imshow(input_sample.reshape(64, 256))
                        plt.subplots_adjust(wspace=None, hspace=None)

                        sample = test[i]

                        ax = plt.subplot(num_examples, 3, idx+1)
                        plt.tight_layout()
                        ax.set_title('train output #{}'.format(i))
                        ax.axis('off')
                        plt.imshow(sample.reshape(64, 256))
                        plt.subplots_adjust(wspace=None, hspace=None)

                        sample = target_batch[i]

                        ax = plt.subplot(num_examples, 3, idx+2)
                        plt.tight_layout()
                        ax.set_title('train target #{}'.format(i))
                        ax.axis('off')
                        plt.imshow(sample.reshape(64, 256))
                        plt.subplots_adjust(wspace=None, hspace=None)

                plt.show()
                #=========================================================================
                # END: visualization of some samples from the training data
                #=========================================================================            

            loss = criterion(prediction, y)
            epoch_loss += prediction.shape[0] * loss.item()

            loss.backward()
            optimizer.step()
        #=====================================================================================
        # END: actual training (adapting weights/parameters)
        #=====================================================================================            


        #======================================================================
        # BEGIN: some housekeeping
        #======================================================================            
        running_val_loss = val_epoch_loss/len(val_dataset)
        running_train_loss = epoch_loss/len(train_dataset)

        if(running_val_loss < best_val_loss):
            best_val_loss = running_val_loss
            best_epoch = epoch

            model_file = os.path.join(model_path, 'best_model.pth')
            save_checkpoint(model_file, epoch, model, optimizer,
                            running_train_loss, running_val_loss,
                            normalizer)

            
        model_file = os.path.join(model_path, 'model_{}.pth'.format(epoch))
        save_checkpoint(model_file, epoch, model, optimizer,
                        running_train_loss, running_val_loss,
                        normalizer)
        #======================================================================
        # END: some housekeeping
        #======================================================================            

    
        print("======================= epoch {}".format(epoch) + "==========================")
        print("train loss: {}".format(running_train_loss))
        print("val loss: {}".format(running_val_loss))
        print("best val loss so far: {} (epoch {})".format(best_val_loss, best_epoch))
    
    
    
def inference(model, dataset, result_prefix, verbosity=0):
    # used to put data on the same device as the model
    device = next(model.parameters()).device

    if(verbosity>0):
        print("using " + str(device) + " for inference...")
    
    data_loader = get_dataloader(dataset, shuffle=False)

    for step, b in enumerate(data_loader):
        image_files = b['image_file']

        X = b['image'].to(device)  # [N, 1, H, W]

        prediction = model(X)  # [N, nb_classes, H, W]

        prediction_sigmoid = torch.sigmoid(prediction.detach())
        output = prediction_sigmoid.cpu().detach().numpy()
        num_results = len(output)

        for f in range(num_results):
            # figure out result filename
            filecomponents = image_files[f].split(os.sep)

            num_filecomponents = len(filecomponents)
#            tmp_path = model_file.replace(".pth", "")
            tmp_path = result_prefix
            
            # combine with prefix and remove ':' from the path
            for comp in range(num_filecomponents-1):
                tmp_path = os.path.join(tmp_path, filecomponents[comp].replace(":", ""))

            result_path = tmp_path

            # create folder for results.
            # (we need to check this more than once, since a dataset can contain files from
            # several directories)
            if(os.path.isdir(result_path)==False):
                try:
                    os.makedirs(result_path)
                except OSError:
                    print ("Creation of the directory %s failed" % result_path)
                else:
                    if(verbosity>0):
                        print ("Successfully created the directory %s " % result_path)


            result_file = os.path.join(result_path, filecomponents[-1])
                
            if(verbosity>1):
                print(result_file.replace(os.getcwd(), '..'))


            input_sample = (output[f]*255).astype(np.uint8)
            io.imsave(result_file, input_sample.reshape(64, 256), check_contrast=False)

            
        
    
def inference_test(input_dir, model_file, device, verbosity=1):
    checkpoint = load_checkpoint(model_file)
    model = compile_model(checkpoint)
    model.to(device)

    normalizer = get_normalizer(checkpoint)
    dataset = load_inference_test_data(input_dir, normalizer)
    
    prefix_path = model_file.replace(".pth", "")

    inference(model, dataset, prefix_path, verbosity=verbosity)

    
# same dataset with multiple models
def multi_inference_test(input_dir, model_files, device, verbosity=1):
    for model_file in model_files:
        inference_test(input_dir, model_file, device, verbosity=verbosity)
    
    

def inference_realworld(input_dir, model_file, device, theta_dir="Deg_0", verbosity=1):
    checkpoint = load_checkpoint(model_file)
    model = compile_model(checkpoint)
    model.to(device)

    normalizer = get_normalizer(checkpoint)
    dataset = load_realworld_data(input_dir, normalizer, theta_dir)
    
    prefix_path = model_file.replace(".pth", "")

    inference(model, dataset, prefix_path, verbosity=verbosity)
    