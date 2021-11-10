##############################################################################################################
# Part of code adapted from https://github.com/alevine0/patchSmoothing/blob/master/certify_imagenet_band.py
##############################################################################################################

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import nets.dsresnet_imgnt as resnet_imgnt
import nets.dsresnet_cifar as resnet_cifar
from torchvision import datasets,transforms
from tqdm import tqdm
from utils.defense_utils import *

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir",default='checkpoints',type=str,help="path to checkpoints")
parser.add_argument('--band_size', default=-1, type=int, help='size of each smoothing band')
parser.add_argument('--patch_size', default=-1, type=int, help='patch_size')
parser.add_argument('--thres', default=0.0, type=float, help='detection threshold for robus masking')
parser.add_argument('--dataset', default='imagenette', choices=('imagenette','imagenet','cifar'),type=str,help="dataset")
parser.add_argument('--data_dir', default='data', type=str,help="path to data")

parser.add_argument('--skip', default=1,type=int, help='Number of images to skip')
parser.add_argument("--m",action='store_true',help="use robust masking")
parser.add_argument("--ds",action='store_true',help="use derandomized smoothing")

args = parser.parse_args()

MODEL_DIR=os.path.join('.',args.model_dir)
DATA_DIR=os.path.join(args.data_dir,args.dataset)
DATASET = args.dataset

device = 'cuda' #if torch.cuda.is_available() else 'cpu'

cudnn.benchmark = True

def get_dataset(ds,data_dir):
    if ds in ['imagenette','imagenet']:
        ds_dir=os.path.join(data_dir,'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        dataset_ = datasets.ImageFolder(ds_dir, transforms.Compose([
                transforms.Resize((299,299)), #note that here input size if 299x299 instead of 224x224
                transforms.ToTensor(),
                normalize,
            ]))
    elif ds == 'cifar':
        transform_test = transforms.Compose([
                         transforms.ToTensor(),
                         #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                         ])
        dataset_ = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
    return dataset_,dataset_.classes

val_dataset_,class_names = get_dataset(DATASET,DATA_DIR)
skips = list(range(0, len(val_dataset_), args.skip))
val_dataset = torch.utils.data.Subset(val_dataset_, skips)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32,shuffle=False)

num_cls = len(class_names)

# Model
print('==> Building model..')



if DATASET == 'imagenette':
    net = resnet_imgnt.resnet50()
    net = torch.nn.DataParallel(net)
    num_ftrs = net.module.fc.in_features
    net.module.fc = nn.Linear(num_ftrs, num_cls)  
    checkpoint = torch.load(os.path.join(MODEL_DIR,'ds_nette.pth'))
    args.band_size = args.band_size if args.band_size>0 else 25
    args.patch_size = args.patch_size if args.patch_size>0 else 42
elif DATASET == 'imagenet':
    net = resnet_imgnt.resnet50()
    net = torch.nn.DataParallel(net)
    checkpoint = torch.load(os.path.join(MODEL_DIR,'ds_net.pth'))
    args.band_size = args.band_size if args.band_size>0 else 25
    args.patch_size = args.patch_size if args.patch_size>0 else 42
elif DATASET == 'cifar':
    net = resnet_cifar.ResNet18()
    net = torch.nn.DataParallel(net)
    checkpoint = torch.load(os.path.join(MODEL_DIR,'ds_cifar.pth'))
    args.band_size = args.band_size if args.band_size>0 else 4
    args.patch_size = args.patch_size if args.patch_size>0 else 5

print(args.band_size,args.patch_size)


net.load_state_dict(checkpoint['net'])

net = net.to(device)
net.eval()


if args.ds:#ds
    correct = 0
    cert_correct = 0
    cert_incorrect = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)
            predictions,  certyn = ds(inputs, net,args.band_size, args.patch_size, num_cls,threshold = 0.2)
            correct += (predictions.eq(targets)).sum().item()
            cert_correct += (predictions.eq(targets) & certyn).sum().item()
            cert_incorrect += (~predictions.eq(targets) & certyn).sum().item()
    print('Results for Derandomized Smoothing')
    print('Using band size ' + str(args.band_size) + ' with threshhold ' + str(0.2))
    print('Certifying For Patch ' +str(args.patch_size) + '*'+str(args.patch_size))
    print('Total images: ' + str(total))
    print('Correct: ' + str(correct) + ' (' + str((100.*correct)/total)+'%)')
    print('Certified Correct class: ' + str(cert_correct) + ' (' + str((100.*cert_correct)/total)+'%)')
    print('Certified Wrong class: ' + str(cert_incorrect) + ' (' + str((100.*cert_incorrect)/total)+'%)')

if args.m:#mask-ds
    result_list=[]
    clean_corr_list=[]
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader):
            inputs = inputs.to(device)
            targets = targets.numpy()
            result,clean_corr = masking_ds(inputs,targets,net,args.band_size, args.patch_size,thres=args.thres)
            result_list+=result
            clean_corr_list+=clean_corr

    cases,cnt=np.unique(result_list,return_counts=True)
    print('Results for Mask-DS')
    print("Provable robust accuracy:",cnt[-1]/len(result_list) if len(cnt)==3 else 0)
    print("Clean accuracy with defense:",np.mean(clean_corr_list))
    print("------------------------------")
    print("Provable analysis cases (0: incorrect prediction; 1: vulnerable; 2: provably robust):",cases)
    print("Provable analysis breakdown:",cnt/len(result_list))


