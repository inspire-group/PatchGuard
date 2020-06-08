##############################################################################################################
# Part of code adapted from https://github.com/alevine0/patchSmoothing/blob/master/certify_imagenet_band.py
##############################################################################################################

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import nets.dsresnet_imgnt as resnet
from torchvision import datasets,transforms
from tqdm import tqdm
from utils.defense_utils import *

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--band_size', default=25, type=int, help='size of each smoothing band')
parser.add_argument('--patch_size', default=42, type=int, help='patch_size')
parser.add_argument('--thres', default=0.0, type=float, help='detection threshold for robus masking')
parser.add_argument('--model', default='resnet50', type=str, help='model')
parser.add_argument('--checkpoint', default='imagenette.pth', type=str,help='checkpoint')
parser.add_argument('--valpath', default='/data/imagenette/val', type=str, help='Path to validation set')
#parser.add_argument('--checkpoint', default='imagenet.pth', type=str,help='checkpoint')
#parser.add_argument('--valpath', default='/data/imagenet_data/val', type=str, help='Path to validation set')
parser.add_argument('--skip', default=1,type=int, help='Number of images to skip')
parser.add_argument("--m",action='store_true',help="use robust masking")
parser.add_argument("--ds",action='store_true',help="use derandomized smoothing")

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
valdir = args.valpath
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

valset = datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize((299,299)), #note that here input size if 299x299 instead of 224x224
        transforms.ToTensor(),
        normalize,
    ]))

skips = list(range(0, len(valset), args.skip))

valset_1 = torch.utils.data.Subset(valset, skips)
testloader = torch.utils.data.DataLoader(valset_1,batch_size=16)

# Model
print('==> Building model..')
checkpoint_dir = 'checkpoints'
if not os.path.exists('./checkpoints'):
    os.makedirs('./checkpoints')
if (args.model == 'resnet50'):
    net = resnet.resnet50()
elif (args.model == 'resnet18'):
    net = resnet.resnet18()

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

num_cls=1000
if 'imagenette' in args.valpath:
    num_ftrs = net.module.fc.in_features
    net.module.fc = nn.Linear(num_ftrs, 10)
    net = net.to(device)
    num_cls=10
print('==> Resuming from checkpoint..')
#assert os.path.isdir(checkpoint_dir), 'Error: no checkpoint directory found!'
resume_file = '{}/{}'.format(checkpoint_dir, args.checkpoint)
print(resume_file)
assert os.path.isfile(resume_file)
checkpoint = torch.load(resume_file)
net.load_state_dict(checkpoint['net'])
net.eval()


if args.ds:#ds
    correct = 0
    cert_correct = 0
    cert_incorrect = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in tqdm(testloader):
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
    clean_fp_list=[]
    with torch.no_grad():
        for inputs, targets in tqdm(testloader):
            inputs = inputs.to(device)
            targets = targets.numpy()
            result,clean_corr,clean_fp = masking_ds(inputs,targets,net,args.band_size, args.patch_size,thres=args.thres)
            result_list+=result
            clean_corr_list+=clean_corr
            clean_fp_list+=clean_fp

    cases,cnt=np.unique(result_list,return_counts=True)
    print('Results for Mask-DS')
    print("Provable analysis cases:",cases)
    print("Provable analysis breakdown",cnt/len(result_list))
    print("Clean accuracy with defense:",np.mean(clean_corr_list))    
    print("Detection FP:",np.mean(clean_fp_list))


