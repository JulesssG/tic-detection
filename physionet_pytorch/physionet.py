import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import datasets, transforms
from utils.data_utils import TransformSubset, PhysionetMMMI
from utils.models import EEGNet
import random
from tqdm import tqdm
from torch.autograd import Variable
import sys
import numpy as np
from sklearn.model_selection import KFold

from utils.preprocess_utils import TimeWindowPostCue, ReshapeTensor

# torch.manual_seed(9823752)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# np.random.seed(1528102)

import os

from contextlib import redirect_stdout, redirect_stderr



DIR_DATA = "/usr/scratch/bismantova/xiaywang/Projects/BCI/datasets/PhysionetMMMI/QuantLab/PhysionetMMMI/data"
CV = True
N_FOLDS = 5

verbose = False

class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)
    def update(self, val):
        self.sum += val.cpu()
        self.n += 1
    @property
    def avg(self):
        return self.sum / self.n

def train(model, device, train_loader, optimizer, scheduler, epoch, verbose=True):
    model.train()
    train_loss = Metric('train_loss')
    with tqdm(total=len(train_loader),
          desc='Train Epoch     #{}'.format(epoch + 1),
          disable=not verbose) as t:
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data.float())
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            train_loss.update(loss)
            t.set_postfix({'train loss': train_loss.avg.item()})
            t.update(1)
        scheduler.step()
        #print(optimizer.param_groups[0]['lr'])
    return train_loss.avg.item()

def validate(model,device,val_loader,verbose=True):
    global min_loss
    model.eval()
    val_loss = Metric('val_loss')

    with tqdm(total=len(val_loader),
          desc='Validation Epoch     #{}'.format(epoch + 1),
          disable=not verbose) as t:
        val_loss_curr = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            output = model(data.float())
            loss = F.cross_entropy(output, target)
            t.set_postfix({'loss': val_loss_curr})
            t.update(1)
            val_loss_curr += loss.item()
            val_loss.update(loss)

    return val_loss.avg.item()

def test(model, device, test_loader, verbose=True):
    model.eval()
    test_acc = Metric('test_acc')
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data.float())
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    return correct/len(test_loader.dataset) * 100.


import time

start = time.time()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
#torch.backends.cudnn.benchmark = True
params = {'batch_size': 16,
          'shuffle': True}
          #'num_workers': 4}

transform = transforms.Compose([ReshapeTensor()])
data_set = PhysionetMMMI(datapath=DIR_DATA, transform=transform)

kf = KFold(n_splits = N_FOLDS)

cv_acc = np.zeros((N_FOLDS, 1))

with tqdm(desc=f'{N_FOLDS} fold cross validation', total=N_FOLDS, ascii=True) as bar:
    for fold, (train_idx, valid_idx) in enumerate(kf.split(data_set)):

        #print(f"training samples {len(train_idx)}, validation samples {len(valid_idx)}")

        train_set = TransformSubset(data_set, train_idx, transform)
        valid_set = TransformSubset(data_set, valid_idx, transform)

        train_loader  = torch.utils.data.DataLoader(train_set, **params)
        val_loader = torch.utils.data.DataLoader(valid_set, **params)


        model = EEGNet().to(device)
        lr = 1e-2
        optimizer = Adam(model.parameters(), lr=lr, eps=1e-7)
        scheduler = MultiStepLR(optimizer, milestones=[20,50], gamma=0.1)
        max_epochs = 100

        for epoch in range(max_epochs):
            train(model, device, train_loader, optimizer, scheduler, epoch, verbose=True)
            validate(model, device, val_loader)

        cv_acc[fold] = test(model, device, val_loader)

        bar.update()

print("\nCV accuracy: %.02f%%" % cv_acc.mean())
print("\nCV accuracy std: %.02f%%" % cv_acc.std())

end = time.time()
print("time used: ", end - start)
