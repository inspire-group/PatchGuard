import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from  torchvision import datasets, transforms

import nets.bagnet
import nets.resnet

import os 
import joblib
import argparse
from tqdm import tqdm
import numpy as np 


parser = argparse.ArgumentParser()

parser.add_argument("--model_dir",default='checkpoints',type=str,help="path to checkpoints")
parser.add_argument('--data_dir', default='/data/imagenette', type=str,help="path to data")
#parser.add_argument('--data_dir', default='/data/imagenet',type=str)
parser.add_argument("--model",default='bagnet17',type=str,help="model name")
parser.add_argument("--clip",default=-1,type=int,help="clipping value; do clipping when this argument is set to positive")
parser.add_argument("--aggr",default='mean',type=str,help="aggregation methods. one of mean, median, cbn")
parser.add_argument("--skip",default=1,type=int,help="number of example to skip")

args = parser.parse_args()

MODEL_DIR=os.path.join('.',args.model_dir)
DATA_DIR=os.path.join(args.data_dir)

#prepare data
val_dir=os.path.join(DATA_DIR,'val')
val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

val_dataset_ = datasets.ImageFolder(val_dir,val_transforms)
class_names = val_dataset_.classes
skips = list(range(0, len(val_dataset_), args.skip))
val_dataset = torch.utils.data.Subset(val_dataset_, skips)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16,shuffle=False)

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

if 'imagenette' in args.data_dir:
	num_ftrs = model.fc.in_features
	model.fc = torch.nn.Linear(num_ftrs, len(class_names))
	model = torch.nn.DataParallel(model)
	print('restoring model from checkpoint...')
	checkpoint = torch.load(os.path.join(MODEL_DIR,args.model+'.pth'))
	model.load_state_dict(checkpoint['model_state_dict'])

model = model.to(device)
cudnn.benchmark = True
model.eval()

accuracy_list=[]

for data,labels in tqdm(val_loader):
	data,labels=data.to(device),labels.to(device)
	output_clean = model(data)
	acc_clean=torch.sum(torch.argmax(output_clean, dim=1) == labels).cpu().detach().numpy()
	accuracy_list.append(acc_clean)
	
print("Test accuracy:",np.sum(accuracy_list)/len(val_dataset))


