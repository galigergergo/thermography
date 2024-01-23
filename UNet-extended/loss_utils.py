#!/usr/bin/env python
# coding: utf-8


import torch

def getBCELoss(dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    criterion = torch.nn.BCELoss()

    running_loss = 0.0

    for step, b in enumerate(dataloader):    
        X = b['image'].to(device)  # [N, 1, H, W]
        y = b['mask'].to(device)  # [N, H, W]

        loss = criterion(X.detach(), y)
#        loss[torch.isnan(loss)] = 0
      #  print(loss)

        running_loss += X.shape[0] * loss.item()
        
    return running_loss/len(dataloader.dataset)



def getMSELoss(dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    criterion = torch.nn.MSELoss()

    running_loss = 0.0

    for step, b in enumerate(dataloader):    
        X = b['image'].to(device)  # [N, 1, H, W]
        y = b['mask'].to(device)  # [N, H, W]

        loss = criterion(X.detach(), y)

        running_loss += X.shape[0] * loss.item()
        
    return running_loss/len(dataloader.dataset)



def getMSELossValues(dataloader, method=''):
    from time import sleep
    from skimage import io
    import matplotlib.pyplot as plt
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    criterion = torch.nn.MSELoss()

    losses = []
    
    for step, b in enumerate(dataloader):    
        X = b['image'].to(device)  # [N, 1, H, W]
        y = b['mask'].to(device)  # [N, H, W]
    
        N = X.shape[0]
        for i in range(N):
            X_i = X.detach()[i, :, :, :]
            y_i = y[i, :, :]
            loss = criterion(X_i, y_i).item()
            losses.append(loss)
            
#             X_np = X_i.cpu().detach().numpy()[0]
#             y_np = y_i.cpu().detach().numpy()[0]
            
#             Nx, Ny = X_np.shape            
#             loss_verif = 0
#             for j in range(Nx):
#                 for k in range(Ny):
#                     loss_verif += (X_np[j][k] - y_np[j][k])**2
#             loss_verif /= (Nx * Ny)
            
            #print('%f        %f' % (loss, loss_verif))
            
            #fig = plt.figure(figsize =(10, 7))
            #plt.imsave('./data/testing/%s/%d_pred.png' % (method, step*10+i+1), X_i.cpu().detach().numpy()[0])
            #fig = plt.figure(figsize =(10, 7))
            #plt.imsave('./data/testing/%s/%d_truth.png' % (method, step*10+i+1), y_i.cpu().detach().numpy()[0])
            
            
        
    return losses





