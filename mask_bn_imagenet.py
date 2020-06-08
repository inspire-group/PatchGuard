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
import joblib
import argparse
from tqdm import tqdm
import numpy as np 
from scipy.special import softmax
from math import ceil

parser = argparse.ArgumentParser()

parser.add_argument("--model_dir",default='checkpoints',type=str,help="path to checkpoints")
parser.add_argument('--data_dir', default='/data/imagenette', type=str,help="path to data")
#parser.add_argument('--data_dir', default='/data/imagenet_data/',type=str)
parser.add_argument("--model",default='bagnet17',type=str,help="model name")
parser.add_argument("--clip",default=-1,type=int,help="clipping value; do clipping when this argument is set to positive")
parser.add_argument("--aggr",default='none',type=str,help="aggregation methods. set to none for local feature")
parser.add_argument("--skip",default=1,type=int,help="number of example to skip")
parser.add_argument("--thres",default=0.0,type=float,help="detection threshold for robust masking")
parser.add_argument("--patch_size",default=31,type=int,help="size of the adversarial patch")
parser.add_argument("--m",action='store_true',help="use robust masking")
parser.add_argument("--cbn",action='store_true',help="use cbn")

args = parser.parse_args()

MODEL_DIR=os.path.join('.',args.model_dir)
DATA_DIR=os.path.join(args.data_dir)

#build and initialize model
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
    rf_size=17
elif 'bagnet33' in args.model:
    model = nets.bagnet.bagnet33(pretrained=True,clip_range=clip_range,aggregation=args.aggr)
    rf_size=33
elif 'bagnet9' in args.model:
    model = nets.bagnet.bagnet9(pretrained=True,clip_range=clip_range,aggregation=args.aggr)
    rf_size=9

rf_stride=8
window_size = ceil((args.patch_size + rf_size -1) / rf_stride)
print("window_size",window_size)
model = model.to(device)
if 'imagenette' in args.data_dir:
	num_ftrs = model.fc.in_features
	model.fc = nn.Linear(num_ftrs, len(class_names))
	model = torch.nn.DataParallel(model)
	print('restoring model from checkpoint...')
	checkpoint = torch.load(os.path.join(MODEL_DIR,args.model+'.pth'))
	model.load_state_dict(checkpoint['model_state_dict'])
	num_cls=10
else:
	model = torch.nn.DataParallel(model)
	num_cls=1000

model = model.to(device)
model.eval()
cudnn.benchmark = True

accuracy_list=[]
result_list=[]
clean_corr=0
clean_fp=0

for data,labels in tqdm(val_loader):
	
	data=data.to(device)
	labels = labels.numpy()
	output_clean = model(data).detach().cpu().numpy() # logits
	#output_clean = softmax(output_clean,axis=-1) # confidence
	#output_clean = (output_clean > 0.2).astype(float) # predictions with confidence threshold

	#note: the provable analysis of robust masking is cpu-intensive and can take some time to finish
	#you can dump the local feature and do the provable analysis with another script so that GPU mempry is not always occupied	
	for i in range(len(labels)):
		if args.m:#robust masking
			result = provable_masking(output_clean[i],labels[i],thres=args.thres,window_shape=[window_size,window_size])
			result_list.append(result)
			cnt,clean_pred = masking_defense(output_clean[i],thres=args.thres,window_shape=[window_size,window_size])
			clean_corr += clean_pred == labels[i]
			clean_fp+=cnt
		elif args.cbn:#cbn
			result = provable_clipping(output_clean[i],labels[i],window_shape=[window_size,window_size])
			result_list.append(result)
			clean_pred = clipping_defense(output_clean[i])
			clean_corr += clean_pred == labels[i]	
	acc_clean = np.mean(np.argmax(np.mean(output_clean,axis=(1,2)),axis=1) == labels)
	accuracy_list.append(acc_clean)


cases,cnt=np.unique(result_list,return_counts=True)
print("Provable analysis cases:",cases)
print("Provable analysis breakdown",cnt/len(result_list))
print("Clean accuracy with defense:",clean_corr/len(result_list))
print("Clean accuracy without defense:",np.mean(accuracy_list))
if args.m:
	print("Detection FP:",clean_fp/len(result_list))
