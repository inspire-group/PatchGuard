import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from  torchvision import datasets, transforms

import nets.bagnet
import nets.resnet
from utils.defense_utils import *

import os 
import argparse
from tqdm import tqdm
import numpy as np 
import PIL
from PatchAttacker import PatchAttacker


parser = argparse.ArgumentParser()
parser.add_argument("--dump_dir",default='patch_adv',type=str,help="directory to save attack results")

parser.add_argument("--model_dir",default='checkpoints',type=str,help="path to checkpoints")
parser.add_argument('--data_dir', default='data', type=str,help="path to data")
parser.add_argument('--dataset', default='imagenette', choices=('imagenette','imagenet','cifar'),type=str,help="dataset")
parser.add_argument("--model",default='bagnet17',type=str,help="model name")
parser.add_argument("--clip",default=-1,type=int,help="clipping value; do clipping when this argument is set to positive")
parser.add_argument("--aggr",default='mean',type=str,help="aggregation methods. set to none for local feature")
parser.add_argument("--patch_size",type=int,help="size of the adversarial patch")

args = parser.parse_args()

MODEL_DIR=os.path.join('.',args.model_dir)
DATA_DIR=os.path.join(args.data_dir,args.dataset)
DATASET = args.dataset
DUMP_DIR=os.path.join('dump',args.dump_dir+'_{}_{}'.format(args.model,args.dataset))
if not os.path.exists('dump'):
    os.mkdir('dump')
if not os.path.exists(DUMP_DIR):
    os.mkdir(DUMP_DIR)



if DATASET in ['imagenette','imagenet']:
    DATA_DIR=os.path.join(DATA_DIR,'val')
    mean_vec = [0.485, 0.456, 0.406]
    std_vec =  [0.229, 0.224, 0.225]
    ds_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean_vec,std_vec)
        ])
    val_dataset = datasets.ImageFolder(DATA_DIR,ds_transforms)
    class_names = val_dataset.classes
elif DATASET == 'cifar':
    mean_vec = [0.4914, 0.4822, 0.4465]
    std_vec = [0.2023, 0.1994, 0.2010]
    ds_transforms = transforms.Compose([
        transforms.Resize(192, interpolation=PIL.Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean_vec,std_vec),
    ])
    val_dataset = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=ds_transforms)
    class_names = val_dataset.classes

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8,shuffle=False)

#build and initialize model
device = 'cuda' #if torch.cuda.is_available() else 'cpu'

if args.clip > 0:
    clip_range = [0,args.clip]
else:
    clip_range = None

if 'bagnet17' in args.model:
    model = nets.bagnet.bagnet17(pretrained=True,clip_range=clip_range,aggregation=args.aggr)
    rf_size=17
elif 'bagnet33' in args.model:
    model = nets.bagnet.bagnet33(pretrained=True,clip_range=clip_range,aggregation=args.aggr)
    rf_size=33
elif 'bagnet9' in args.model:
    model = nets.bagnet.bagnet9(pretrained=True,clip_range=clip_range,aggregation=args.aggr)
    rf_size=9


if DATASET == 'imagenette':
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model = torch.nn.DataParallel(model)
    checkpoint = torch.load(os.path.join(MODEL_DIR,args.model+'_nette.pth'))
    model.load_state_dict(checkpoint['model_state_dict']) 
elif  DATASET == 'imagenet':
    model = torch.nn.DataParallel(model)
    checkpoint = torch.load(os.path.join(MODEL_DIR,args.model+'_net.pth'))
    model.load_state_dict(checkpoint['state_dict'])
elif  DATASET == 'cifar':
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model = torch.nn.DataParallel(model)
    checkpoint = torch.load(os.path.join(MODEL_DIR,args.model+'_192_cifar.pth'))
    model.load_state_dict(checkpoint['net'])
    
model = model.to(device)
model.eval()
cudnn.benchmark = True


attacker = PatchAttacker(model, mean_vec, std_vec,patch_size=args.patch_size,step_size=0.05,steps=500)

adv_list=[]
error_list=[]
accuracy_list=[]
patch_loc_list=[]

for data,labels in tqdm(val_loader):
    
    data,labels=data.to(device),labels.to(device)
    data_adv,patch_loc = attacker.perturb(data, labels)

    output_adv = model(data_adv)
    error_adv=torch.sum(torch.argmax(output_adv, dim=1) != labels).cpu().detach().numpy()
    output_clean = model(data)
    acc_clean=torch.sum(torch.argmax(output_clean, dim=1) == labels).cpu().detach().numpy()

    data_adv=data_adv.cpu().detach().numpy()
    patch_loc=patch_loc.cpu().detach().numpy()

    patch_loc_list.append(patch_loc)
    adv_list.append(data_adv)
    error_list.append(error_adv)
    accuracy_list.append(acc_clean)


adv_list = np.concatenate(adv_list)
patch_loc_list = np.concatenate(patch_loc_list)
joblib.dump(adv_list,os.path.join(DUMP_DIR,'patch_adv_list_{}.z'.format(args.patch_size)))
joblib.dump(patch_loc_list,os.path.join(DUMP_DIR,'patch_loc_list_{}.z'.format(args.patch_size)))
print("Attack success rate:",np.sum(error_list)/len(val_dataset))
print("Clean accuracy:",np.sum(accuracy_list)/len(val_dataset))
    
