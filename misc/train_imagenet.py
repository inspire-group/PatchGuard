#######################################################################################
# Adapted from https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
# Used for training models on ImageNette
#######################################################################################

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
from tqdm import tqdm
import random
import nets.bagnet
import nets.resnet
import argparse
from utils.cutout import Cutout


parser = argparse.ArgumentParser()
parser.add_argument("--model_dir",default='checkpoints',type=str)
parser.add_argument("--data_dir",default='/data/imagenette',type=str)
parser.add_argument("--model_name",default='bagnet17.pth',type=str)
parser.add_argument("--clip",default=-1,type=int)
parser.add_argument("--epoch",default=20,type=int)
parser.add_argument("--cutout_size",default=31,type=int)
parser.add_argument("--aggr",default='mean',type=str)
parser.add_argument("--resume",action='store_true')
parser.add_argument("--cutout",action='store_true',help="use CUTOUT during the training")
parser.add_argument("--fc",action='store_true',help="only retrain the fully-connected layer")
args = parser.parse_args()

MODEL_DIR=os.path.join('.',args.model_dir)
DATA_DIR=os.path.join(args.data_dir)

if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

mean_vec=[0.485, 0.456, 0.406]
std_vec=[0.229, 0.224, 0.225]

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean_vec, std_vec)
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean_vec,std_vec)
    ]),
}

if args.cutout:
    data_transforms['train'].transforms.append(Cutout(n_holes=1, length=args.cutout_size))

train_dir=os.path.join(DATA_DIR,'train')
val_dir=os.path.join(DATA_DIR,'val')

train_dataset = datasets.ImageFolder(train_dir,data_transforms['train'])
val_dataset = datasets.ImageFolder(val_dir,data_transforms['val'])

print('train_dataset.size',len(train_dataset.samples))
print('val_dataset.size',len(val_dataset.samples))
image_datasets = {'train':train_dataset,'val':val_dataset}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
print('class_names:',class_names)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64,shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64,shuffle=False)

dataloaders={'train':train_loader,'val':val_loader}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('device:',device)

def train_model(model, criterion, optimizer, scheduler, num_epochs=20 ,mask=False):

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in tqdm(range(num_epochs)):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print('saving...')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_model_wts,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict':scheduler.state_dict()
                    }, os.path.join(MODEL_DIR,args.model_name))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

if args.clip > 0:
	clip_range = [0,args.clip]
else:
	clip_range = None
    
if 'bagnet17' in args.model_name:
    model_conv = nets.bagnet.bagnet17(pretrained=True,clip_range=clip_range,aggregation=args.aggr)
elif 'bagnet33' in args.model_name:
    model_conv = nets.bagnet.bagnet33(pretrained=True,clip_range=clip_range,aggregation=args.aggr)
elif 'bagnet9' in args.model_name:
    model_conv = nets.bagnet.bagnet9(pretrained=True,clip_range=clip_range,aggregation=args.aggr)
elif 'resnet50' in args.model_name:
    model_conv = nets.resnet.resnet50(pretrained=True,clip_range=clip_range,aggregation=args.aggr)

if args.fc: #only retrain the fully-connected layer
	for param in model_conv.parameters():
	    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, len(class_names))
model_conv = torch.nn.DataParallel(model_conv)
model_conv = model_conv.to(device)
criterion = nn.CrossEntropyLoss()

if args.fc:
	optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
else:
	optimizer_conv = optim.SGD(model_conv.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
#print(optimizer_conv.state_dict())
#https://pytorch.org/tutorials/beginner/saving_loading_models.html
if args.resume:
    print('restoring model from checkpoint...')
    checkpoint = torch.load(os.path.join(MODEL_DIR,args.model_name))
    model_conv.load_state_dict(checkpoint['model_state_dict'])
    model_conv = model_conv.to(device)
    #https://discuss.pytorch.org/t/code-that-loads-sgd-fails-to-load-adam-state-to-gpu/61783/3
    optimizer_conv.load_state_dict(checkpoint['optimizer_state_dict'])
    exp_lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    #print(checkpoint['optimizer_state_dict'])
    #print(checkpoint['scheduler_state_dict'])


model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=args.epoch)

