#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import torch
#import os.path.isdir

def load_model_losses(model_file):
    checkpoint = torch.load(model_file)

    depth = checkpoint['unet_depth']
    wf = checkpoint['unet_wf']

    # meta info about the model
    train_loss = checkpoint['train_loss']
    val_loss = checkpoint['val_loss']
    
    return [train_loss, val_loss, depth, wf]
    

def load_model_loss_curve(model_path):
    file_list = [f for f in os.listdir(model_path) if (f.startswith("model_") and not os.path.isdir(os.path.join(model_path, f)))]

    depth = 0
    wf = 0

    num_epochs = len(file_list)
    
    train_loss_curve = np.zeros((num_epochs, 1))
    val_loss_curve = np.zeros((num_epochs, 1))
    min_val_loss = -1

    for epoch in range(num_epochs):
        model_file = os.path.join(model_path, 'model_{}.pth'.format(epoch))
        print("loading " + model_file)

        if(epoch==0):
            tmp = load_model_losses(model_file)
            train_loss_curve[epoch] = tmp[0]
            val_loss_curve[epoch] = tmp[1]
            depth = tmp[2]
            wf = tmp[3]
        else:
            tmp = load_model_losses(model_file)
            train_loss_curve[epoch] = tmp[0]
            val_loss_curve[epoch] = tmp[1]

        min_val_loss = np.min(val_loss_curve)

    return {"train_loss_curve": train_loss_curve, 
             "val_loss_curve": val_loss_curve,
             "min_val_loss": min_val_loss,
             "wf": wf,
             "depth": depth,
             "num_epochs": num_epochs,
            }


