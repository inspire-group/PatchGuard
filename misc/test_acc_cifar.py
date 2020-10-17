import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms

import nets.bagnet
import nets.resnet
import PIL

import os 
import joblib
import argparse
from tqdm import tqdm
import numpy as np 

parser = argparse.ArgumentParser()

parser.add_argument("--model_dir",default='checkpoints',type=str,help="path to checkpoints")
parser.add_argument('--data_dir', default='/data/cifar', type=str,help="path to data")
parser.add_argument("--model",default='bagnet17_cifar',type=str,help="model name")
parser.add_argument("--clip",default=-1,type=int,help="clipping value; do clipping when this argument is set to positive")
parser.add_argument("--aggr",default='mean',type=str,help="aggregation methods. one of mean, median, cbn")

args = parser.parse_args()

MODEL_DIR=os.path.join('.',args.model_dir)
DATA_DIR=os.path.join(args.data_dir)

#prepare data
transform_test = transforms.Compose([
    transforms.Resize(192, interpolation=PIL.Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
testset = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform_test)

val_loader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False)

#build and initialize model
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if args.clip > 0:
	clip_range = [0,args.clip]
else:
	clip_range = None

if 'bagnet17' in args.model:
    model = nets.bagnet.bagnet17(pretrained=True,clip_range=clip_range,aggregation=args.aggr)
elif 'bagnet33' in args.model:
    model = nets.bagnet.bagnet33(pretrained=True,clip_range=clip_range,aggregation=args.aggr)
elif 'bagnet9' in args.model:
    model = nets.bagnet.bagnet9(pretrained=True,clip_range=clip_range,aggregation=args.aggr)
elif 'resnet50' in args.model:
    model = nets.resnet.resnet50(pretrained=True,clip_range=clip_range,aggregation=args.aggr)


num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)
model = torch.nn.DataParallel(model)
cudnn.benchmark = True
print('restoring model from checkpoint...')
checkpoint = torch.load(os.path.join(MODEL_DIR,args.model+'.pth'))
model.load_state_dict(checkpoint['net'])
model = model.to(device)
model.eval()


accuracy_list=[]

for data,labels in tqdm(val_loader):
	
	data,labels=data.to(device),labels.to(device)
	output_clean = model(data)
	acc_clean=torch.sum(torch.argmax(output_clean, dim=1) == labels).cpu().detach().numpy()
	accuracy_list.append(acc_clean)
	
print("Test accuracy:",np.sum(accuracy_list)/len(testset))

