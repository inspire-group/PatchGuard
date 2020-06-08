################################################################################################################
# provable analysis and defense implementaion of robust masking and clipping-based defense
#
# INPUT
# local feature 	numpy.ndarray feature tensor in the shape of [feature_size_x,feature_size_y,num_cls]
# label				true label
# target_cls 		target class for the provable analysis. If None, take the second prediction as target_cls
# thres				detection threshold T for robust masking
# window_shape 		the shape of window, in the shape of [window_size_x,window_size_y]
#
# OUTPUT
# failure case number (provable analysis) / final prediction
################################################################################################################


import numpy as np 
import torch
from scipy.special import softmax
def provable_masking(local_feature,label,target_cls=None,thres=0.,window_shape=[6,6]):
	#provable analysis for mask-bn
	feature_size_x,feature_size_y,num_cls = local_feature.shape
	window_size_x,window_size_y = window_shape
	num_window_x = feature_size_x - window_size_x + 1
	num_window_y = feature_size_y - window_size_y + 1

	local_feature = np.clip(local_feature,0,10000000)
	global_feature = np.mean(local_feature,axis=(0,1))
	pred_list = np.argsort(global_feature)
	global_pred = pred_list[-1]

	if global_pred != label: #clean prediction is incorrect
		return 0
	local_feature_pred = local_feature[:,:,global_pred]

	if target_cls is None:
		target_cls = pred_list[-2] #second prediction
	# Note: The second prediction is the easist target class; iterating all possible classes gives (almost) identical results
	#for target_cls in range(num_cls):
	#	if target_cls == label:
	#		continue
	local_feature_target = local_feature[:,:,target_cls]
	diff_feature = local_feature_pred - local_feature_target

	#preparation for more efficient computation
	diff_matrix = np.zeros([num_window_x,num_window_y])
	target_window_sum_matrix_base = np.zeros([num_window_x,num_window_y])
	pred_window_sum_matrix_base = np.zeros([num_window_x,num_window_y])
	for x in range(0,num_window_x):
		for y in range(0,num_window_y):
			diff_feature_masked = diff_feature.copy()
			diff_feature_masked[x:x+window_size_x,y:y+window_size_y]=0
			diff_matrix[x,y] = diff_feature_masked.sum() #\delta
			target_window_sum_matrix_base[x,y] =  local_feature_target[x:x+window_size_x,y:y+window_size_y].sum()
			pred_window_sum_matrix_base[x,y] =  local_feature_pred[x:x+window_size_x,y:y+window_size_y].sum()

	#main provable analysis
	for x in range(0,num_window_x):
		for y in range(0,num_window_y):

			#zero out class evidence in this window
			local_feature_target_masked = local_feature_target.copy()
			local_feature_target_masked[x:x+window_size_x,y:y+window_size_y]=0
			local_feature_pred_masked = local_feature_pred.copy()
			local_feature_pred_masked[x:x+window_size_x,y:y+window_size_y]=0
			min_required_diff = diff_matrix[x,y]
			# Case I: not windown detected + Case VI: perfectly detected malicious window
			if min_required_diff / (min_required_diff + local_feature_target_masked.sum()) < thres:
				return 1

			# Case II: a benign window detected	
			target_cls_window_sum_matrix = target_window_sum_matrix_base.copy()
			for xx in range(max(0,x - window_size_x + 1),min(x + window_size_x,num_window_x)):
				for yy in range(max(0,y - window_size_y + 1),min(y + window_size_y,num_window_y)):
					target_cls_window_sum_matrix[xx,yy] = local_feature_target_masked[xx:xx+window_size_x,yy:yy+window_size_y].sum()
			max_target_cls_window_sum = np.max(target_cls_window_sum_matrix)
			idx = np.argmax(target_cls_window_sum_matrix)
			xx = idx // num_window_y
			yy = idx % num_window_y
			if local_feature_pred_masked[xx:xx+window_size_x,yy:yy+window_size_y].sum() > min_required_diff:
				return 2

			# Case III: a partially detected maclious window
			pred_cls_window_sum_matrix = np.zeros([feature_size_x,feature_size_y])
			for xx in range(x - window_size_x + 1,x + window_size_x):
				for yy in range(y - window_size_y + 1,y + window_size_y):
					pred_cls_window_sum_matrix[xx,yy] = local_feature_pred_masked[xx:xx+window_size_x,yy:yy+window_size_y].sum()
			max_pred_cls_window_sum = np.max(pred_cls_window_sum_matrix)
			if max_pred_cls_window_sum > min_required_diff:
				return 3

	return 4 #provable robustness

def masking_defense(local_feature,thres=0.,window_shape=[6,6]):
	#mask-bn defense
	feature_size_x,feature_size_y,num_cls = local_feature.shape
	window_size_x,window_size_y = window_shape
	num_window_x = feature_size_x - window_size_x + 1
	num_window_y = feature_size_y - window_size_y + 1
	
	local_feature = np.clip(local_feature,0,10000000)
	global_feature = np.mean(local_feature,axis=(0,1))
	#note: when feature type is prediction. it is likely that there are more than one max global feature values
	#1) we might get different `global_pred` if the sort algorithm is not stable
	#2) np.argsort(global_feature)[-1] might give different `global_pred` with np.argmax(global_feature)
	#this will not affect the results of provable analysis, since it is trivial for a successful attack
	#when the feature type is logits and confidence, since the values are float number, it is unlikely to have more than one max global feature values
	pred_list = np.argsort(global_feature)
	global_pred = pred_list[-1]
	local_feature_pred = local_feature[:,:,global_pred]


	logits_pred_window_sum_matrix=np.zeros([num_window_x,num_window_y])
	for x in range(0,num_window_x):
		for y in range(0,num_window_y):
			logits_pred_window_sum_matrix[x,y] = local_feature_pred[x:x+window_size_x,y:y+window_size_y].sum()
	max_window_sum = np.max(logits_pred_window_sum_matrix)
	if max_window_sum / local_feature_pred.sum() < thres:
		return 0,global_pred
	else:
		tmp = np.argmax(logits_pred_window_sum_matrix)
		xx = tmp // num_window_y
		yy = tmp % num_window_y
		local_feature[xx:xx+window_size_x,yy:yy+window_size_y,:] = 0
		global_feature = np.mean(local_feature,axis=(0,1))
		global_pred = np.argmax(global_feature)
		return 1,global_pred


def clipping_defense(local_feature,clipping=None):
	#clipping defense
	if clipping is not None:
		local_feature = np.clip(local_feature,0,clipping) #clipped with [0,clipping]
	else:
		local_feature = np.tanh(local_feature*0.05-1) # clipped with tanh (CBN)
	global_feature = np.mean(local_feature,axis=(0,1))
	global_pred = np.argmax(global_feature)
	return global_pred

def provable_clipping(local_feature,label,target_cls=None,clipping=None,window_shape=[6,6]):
	#provable analysis of clipping defense, including clipping with cbn and clipping the [0,clipping]
	feature_size_x,feature_size_y,num_cls = local_feature.shape
	window_size_x,window_size_y = window_shape
	num_window_x = feature_size_x - window_size_x + 1
	num_window_y = feature_size_y - window_size_y + 1

	if clipping is not None: 
		local_feature = np.clip(local_feature,0,clipping) #clipped with [0,clipping]
		max_increase = window_size_x * window_size_y * clipping
	else:
		local_feature = np.tanh(local_feature*0.05-1) # clipped with tanh (CBN)
		max_increase = window_size_x * window_size_y * 2

	local_pred = np.argmax(local_feature,axis=-1)
	global_feature = np.mean(local_feature,axis=(0,1))
	pred_list = np.argsort(global_feature)
	global_pred = pred_list[-1]
	if global_pred != label: #clean prediction is incorrect
		return 0
	local_feature_pred = local_feature[:,:,global_pred]

	if target_cls is None:
		target_cls = pred_list[-2] #second prediction

	local_feature_target = local_feature[:,:,target_cls]
	diff_feature = local_feature_pred - local_feature_target

	for x in range(0,num_window_x):
		for y in range(0,num_window_y):
			diff_feature_masked = diff_feature.copy()
			diff_feature_masked[x:x+window_size_x,y:y+window_size_y]=0
			diff = diff_feature_masked.sum()
			if diff < max_increase:
				return 1
	return 2 # provable robustness



##########################################################################################
# Adapted from https://github.com/alevine0/patchSmoothing/blob/master/utils_band.py
# for the original ds defense and mask-ds
# each function returns the provable ananlysis results as well as the defense prediction
##########################################################################################

def ds(inpt, net,block_size, size_to_certify, num_classes, threshold=0.2):
	# de-randomized smoothing
	# inpt: inpt image BCWH
	# net: pytorch model
	# block_size: pixel band size
	# size_to_certify: patch size
	# num_classes: number of classes
	# threshold: prediction threshold
	predictions = torch.zeros(inpt.size(0), num_classes).type(torch.int).cuda()
	batch = inpt.permute(0,2,3,1) #color channel last
	for pos in range(batch.shape[2]):

		out_c1 = torch.zeros(batch.shape).cuda()
		out_c2 = torch.zeros(batch.shape).cuda()
		if  (pos+block_size > batch.shape[2]):
			out_c1[:,:,pos:] = batch[:,:,pos:]
			out_c2[:,:,pos:] = 1. - batch[:,:,pos:]

			out_c1[:,:,:pos+block_size-batch.shape[2]] = batch[:,:,:pos+block_size-batch.shape[2]]
			out_c2[:,:,:pos+block_size-batch.shape[2]] = 1. - batch[:,:,:pos+block_size-batch.shape[2]]
		else:
			out_c1[:,:,pos:pos+block_size] = batch[:,:,pos:pos+block_size]
			out_c2[:,:,pos:pos+block_size] = 1. - batch[:,:,pos:pos+block_size]

		out_c1 = out_c1.permute(0,3,1,2)
		out_c2 = out_c2.permute(0,3,1,2)
		out = torch.cat((out_c1,out_c2), 1)
		softmx = torch.nn.functional.softmax(net(out),dim=1)
		predictions += (softmx >= threshold).type(torch.int).cuda()

	predinctionsnp = predictions.cpu().numpy()
	idxsort = np.argsort(-predinctionsnp,axis=1,kind='stable')
	valsort = -np.sort(-predinctionsnp,axis=1,kind='stable')
	val =  valsort[:,0]
	idx = idxsort[:,0]
	valsecond =  valsort[:,1]
	idxsecond =  idxsort[:,1] 
	num_affected_classifications=(size_to_certify + block_size -1)
	cert = torch.tensor(((val - valsecond >2*num_affected_classifications) | ((val - valsecond ==2*num_affected_classifications)&(idx < idxsecond)))).cuda()
	return torch.tensor(idx).cuda(), cert


def masking_ds(inpt, labels,net,block_size, size_to_certify, thres=0.0):
	# mask-ds
	# inpt: inpt image BCWH
	# labels: true label
	# net: pytorch model
	# block_size: pixel band size
	# size_to_certify: patch size
	# thres: detection threshold for robust masking (different from threshold in ds())
	logits_list=[]
	cnf_list=[]
	pred_list=[]
	batch = inpt.permute(0,2,3,1) #color channel last
	for pos in range(batch.shape[2]):
		out_c1 = torch.zeros(batch.shape).cuda()
		out_c2 = torch.zeros(batch.shape).cuda()
		if  (pos+block_size > batch.shape[2]):
			out_c1[:,:,pos:] = batch[:,:,pos:]
			out_c2[:,:,pos:] = 1. - batch[:,:,pos:]

			out_c1[:,:,:pos+block_size-batch.shape[2]] = batch[:,:,:pos+block_size-batch.shape[2]]
			out_c2[:,:,:pos+block_size-batch.shape[2]] = 1. - batch[:,:,:pos+block_size-batch.shape[2]]
		else:
			out_c1[:,:,pos:pos+block_size] = batch[:,:,pos:pos+block_size]
			out_c2[:,:,pos:pos+block_size] = 1. - batch[:,:,pos:pos+block_size]

		out_c1 = out_c1.permute(0,3,1,2)
		out_c2 = out_c2.permute(0,3,1,2)
		out = torch.cat((out_c1,out_c2), 1)
		logits_tmp = net(out).detach().cpu().numpy()
		cnf_tmp = softmax(logits_tmp,axis=-1)
		pred_tmp = (cnf_tmp > 0.2).astype(float)
		logits_list.append(logits_tmp)
		cnf_list.append(cnf_tmp)
		pred_list.append(pred_tmp)

	#output_list = np.stack(logits_list,axis=1)
	output_list = np.stack(cnf_list,axis=1)
	#output_list = np.stack(pred_list,axis=1)

	B,W,C=output_list.shape
	result_list=[]
	clean_corr_list=[]
	clean_fp_list=[]
	window_size = block_size + size_to_certify -1

	for i in range(len(labels)):
		local_feature = output_list[i].reshape([W,1,C])
		result=provable_masking(local_feature,labels[i],window_shape=[window_size,1],thres=thres)
		cnt,clean_pred=masking_defense(local_feature,window_shape=[window_size,1],thres=thres)
		result_list.append(result)
		clean_corr_list.append(clean_pred == labels[i])
		clean_fp_list.append(cnt)
	return result_list,clean_corr_list,clean_fp_list
