#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch

# scales a given numpy array to have values btw. [0, 1]
def normalize(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))


# counts non-zero (=1) pixels
def L0_norm(X):
    l0_norm = np.sum(X==1)
    
    return l0_norm


# creates binarized mask according to threshold for a single 2d numpy array
def mask(X, thresh=0.5):
    X_masked = X.copy()
    X_masked[X_masked<=thresh] = 0.
    X_masked[X_masked>thresh] = 1.
    
    return X_masked


# computes uncertainty involing MSE on an estimated ground truth for a single 2d numpy array
# bounded [0-1]
def ucrt1(X, sens_thresh=0.01, conf_thresh=0.95, criterion=torch.nn.MSELoss()):
    X_sens = mask(X, sens_thresh)
    X_conf = mask(X, conf_thresh)
    
    l0_sens = L0_norm(X_sens)
    l0_conf = L0_norm(X_conf)
    
    ratio = l0_conf/l0_sens
    
    loss_estimate = criterion(torch.Tensor(X[X_sens==1]), torch.Tensor(X_sens[X_sens==1])).item()
    
    uncertainty = loss_estimate/(1+ratio)
    
    return uncertainty


# computes uncertainty involing estimated area of defect for a single 2d numpy array
# unbounded
def ucrt2(X, sens_thresh=0.01, conf_thresh=0.95):
    X_sens = mask(X, sens_thresh)
    X_conf = mask(X, conf_thresh)
    
    l0_sens = L0_norm(X_sens)
    l0_conf = L0_norm(X_conf)
    
    ratio = l0_conf/l0_sens
    
    num_pixels = X.shape[0]*X.shape[1]
    
    uncertainty = (1-ratio)*(l0_sens/num_pixels)
    
    return uncertainty


# computes uncertainty involing estimated area of defect for a single 2d numpy array
# bounded [0-1]
def ucrt3(X, sens_thresh=0.01, conf_thresh=0.95):
    X_sens = mask(X, sens_thresh)
    X_conf = mask(X, conf_thresh)
    
    l0_sens = L0_norm(X_sens)
    l0_conf = L0_norm(X_conf)
    
    ratio = l0_conf/l0_sens
    
    num_pixels = X.shape[0]*X.shape[1]
    
    uncertainty = (l0_sens/num_pixels)/(1+ratio)
    
    return uncertainty


# computes uncertainties involing MSE on an estimated ground truth for a dataset
def getUCRT1(dataset, sens_thresh=0.01, conf_thresh=0.95):
    uncertainty = np.zeros(len(dataset),)
    
    for i in range(len(dataset)):
        X = dataset[i]['image'].numpy()[0,:,:]
        uncertainty[i] = ucrt1(X, sens_thresh, conf_thresh)

    return uncertainty


# computes uncertainties involing estimated area of defect for a single 2d numpy array
def getUCRT2(dataset, sens_thresh=0.01, conf_thresh=0.95):
    uncertainty = np.zeros(len(dataset),)
    
    for i in range(len(dataset)):
        X = dataset[i]['image'].numpy()[0,:,:]
        uncertainty[i] = ucrt2(X, sens_thresh, conf_thresh)

    return uncertainty


# computes uncertainties involing estimated area of defect for a single 2d numpy array
# same characteristics as ucrt2, but bounded [0, 1]
def getUCRT3(dataset, sens_thresh=0.01, conf_thresh=0.95):
    uncertainty = np.zeros(len(dataset),)
    
    for i in range(len(dataset)):
        X = dataset[i]['image'].numpy()[0,:,:]
        uncertainty[i] = ucrt3(X, sens_thresh, conf_thresh)

    return uncertainty


# computes loss for each item in the dataset according to the given criterion
def getLosses(dataset, criterion):
    losses = np.zeros(len(dataset),)

    for idx in range(len(dataset)):
        X = torch.Tensor(dataset[idx]['image'])
        y = torch.Tensor(dataset[idx]['mask'])

        losses[idx] = criterion(X, y).item()
        
    return losses


# computes BCE loss for each item in the dataset
def getBCELosses(dataset):
    return getLosses(dataset, torch.nn.BCELoss())


# computes MSE loss for each item in the dataset
def getMSELosses(dataset):
    return getLosses(dataset, torch.nn.MSELoss())    
