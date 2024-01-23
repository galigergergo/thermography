#!/usr/bin/env python
# coding: utf-8

# In[1]:
import torch

from thermunet import ThermUNet

def save_checkpoint(file_name, epoch, model, optimizer, train_loss, val_loss, normalizer):        
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'mean': normalizer['norm_mean'],
                'std': normalizer['norm_std'],
                'unet_in_channels': model.__getattribute__('in_channels'),
                'unet_n_classes': model.__getattribute__('n_classes'),
                'unet_depth': model.__getattribute__('depth'),
                'unet_wf': model.__getattribute__('wf'),
                'unet_padding': model.__getattribute__('padding'),
                'unet_batch_norm': model.__getattribute__('batch_norm'),
                'unet_up_mode': model.__getattribute__('up_mode'),
                }, file_name)    


    
def load_checkpoint(file_name):
    checkpoint = torch.load(file_name)
    
    return checkpoint


def get_normalizer(checkpoint):
    normalizer = {
        'norm_mean': checkpoint['mean'],
        'norm_std': checkpoint['std'],
        }

    return normalizer
 

def compile_model(checkpoint):
    # mandatory hyperparameters
    in_channels = checkpoint['unet_in_channels']
    n_classes = checkpoint['unet_n_classes']
    depth = checkpoint['unet_depth']
    wf = checkpoint['unet_wf']

    # optional hyperparameters
    if checkpoint.get("unet_batch_norm") == None:
        batch_norm = False
    else:
        batch_norm = checkpoint['unet_batch_norm']

    if checkpoint.get("unet_padding") == None:
        padding = True
    else:
        padding = checkpoint['unet_padding']

    if checkpoint.get("up_mode") == None:
        up_mode = 'upsample'
    else:
        up_mode = checkpoint['up_mode']

    model = ThermUNet(batch_norm=batch_norm, n_classes=n_classes, in_channels=in_channels, depth=depth, wf=wf,
                 padding=padding, up_mode=up_mode)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval() # put model into inference mode
    
    return model
    