import numpy as np 
import torch
from scipy.special import softmax

# robust masking defense (Algorithm 1 in the paper)
def masking_defense(local_feature,clipping=-1,thres=0.,window_shape=[6,6]):
	'''
	local_feature	numpy.ndarray, feature tensor in the shape of [feature_size_x,feature_size_y,num_cls]
	clipping 		int/float, the positive clipping value ($c_h$ in the paper). If clipping < 0, treat clipping as np.inf
	thres 			float in [0,1], detection threshold. ($T$ in the paper)
	window_shape	list [int,int], the shape of sliding window

	Return 			int, robust prediction
	'''

	feature_size_x,feature_size_y,num_cls = local_feature.shape
	window_size_x,window_size_y = window_shape
	num_window_x = feature_size_x - window_size_x + 1
	num_window_y = feature_size_y - window_size_y + 1

	# clipping
	if clipping >0:
		local_feature = np.clip(local_feature,0,clipping)
	else:
		local_feature = np.clip(local_feature,0,np.inf)


	global_feature = np.sum(local_feature,axis=(0,1))

	# the sum of class evidence within each window
	in_window_sum_tensor=np.zeros([num_window_x,num_window_y,num_cls])
	for x in range(0,num_window_x):
		for y in range(0,num_window_y):
			in_window_sum_tensor[x,y,:] = np.sum(local_feature[x:x+window_size_x,y:y+window_size_y,:],axis=(0,1))

	# calculate clipped and masked class evidence for each class
	for c in range(num_cls):
		max_window_sum = np.max(in_window_sum_tensor[:,:,c])
		if global_feature[c] > 0 and max_window_sum / global_feature[c] > thres:
			global_feature[c]-=max_window_sum

	pred_list = np.argsort(global_feature,kind='stable')#"stable" is necessary when the feature type is prediction
	return pred_list[-1]


# provable analysis of robust masking defense (Algorithm 2 in the paper)
def provable_masking(local_feature,label,clipping=-1,thres=0.,window_shape=[6,6]):
	'''
	local_feature	numpy.ndarray, feature tensor in the shape of [feature_size_x,feature_size_y,num_cls]
	label 			int, true label
	clipping 		int/float, the positive clipping value ($c_h$ in the paper). If clipping < 0, treat clipping as np.inf
	thres 			float in [0,1], detection threshold. ($T$ in the paper)
	window_shape	list [int,int], the shape of sliding window

	Return 		int, provable analysis results (0: incorrect clean prediction; 1: possible attack found; 2: certified robustness )
	'''

	feature_size_x,feature_size_y,num_cls = local_feature.shape
	window_size_x,window_size_y = window_shape
	num_window_x = feature_size_x - window_size_x + 1
	num_window_y = feature_size_y - window_size_y + 1

	if clipping > 0:
		local_feature = np.clip(local_feature,0,clipping)
	else:
		local_feature = np.clip(local_feature,0,np.inf)

	global_feature = np.sum(local_feature,axis=(0,1))

	pred_list = np.argsort(global_feature,kind='stable')
	global_pred = pred_list[-1]

	if global_pred != label: # clean prediction is incorrect
		return 0

	local_feature_pred = local_feature[:,:,global_pred]

	# the sum of class evidence within each window
	in_window_sum_tensor = np.zeros([num_window_x,num_window_y,num_cls])

	for x in range(0,num_window_x):
		for y in range(0,num_window_y):
			in_window_sum_tensor[x,y,:] = np.sum(local_feature[x:x+window_size_x,y:y+window_size_y,:],axis=(0,1))


	idx = np.ones([num_cls],dtype=bool)
	idx[global_pred]=False
	for x in range(0,num_window_x):
		for y in range(0,num_window_y):

			# determine the upper bound of wrong class evidence
			global_feature_masked = global_feature - in_window_sum_tensor[x,y,:] # $t$ in the proof of Lemma 1
			global_feature_masked[idx]/=(1 - thres) # $t/(1-T)$, the upper bound of wrong class evidence 

			# determine the lower bound of true class evidence
			local_feature_pred_masked = local_feature_pred.copy()
			local_feature_pred_masked[x:x+window_size_x,y:y+window_size_y]=0 # operation $u\odot(1-w)$
			in_window_sum_pred_masked = in_window_sum_tensor[:,:,global_pred].copy()
			# only need to recalculate the windows the are partially masked
			for xx in range(max(0,x - window_size_x + 1),min(x + window_size_x,num_window_x)):
				for yy in range(max(0,y - window_size_y + 1),min(y + window_size_y,num_window_y)):
					in_window_sum_pred_masked[xx,yy]=local_feature_pred_masked[xx:xx+window_size_x,yy:yy+window_size_y].sum()

			max_window_sum_pred = np.max(in_window_sum_pred_masked) # find the window with the largest sum
			if max_window_sum_pred / local_feature_pred_masked.sum() > thres: 
				global_feature_masked[global_pred]-=max_window_sum_pred

			# determine if an attack is possible
			if np.argsort(global_feature_masked,kind='stable')[-1]!=label: 
				return 1

	return 2 #provable robustness




# De-randomized Smoothing 
# Adapted from https://github.com/alevine0/patchSmoothing/blob/master/utils_band.py
def ds(inpt,net,block_size, size_to_certify, num_classes, threshold=0.2):
	'''
	inpt 				torch.tensor, the input images in CWH format
	net 				torch.nn.module, the based model whose input is small pixel bands
	block_size 			int, the width of pixel bands
	size_to_certify     int, the patch size to be certified
	num_classes         int, number of classes
	threshold			float, the threshold for prediction, see their original paper for details

	Return 				[torch.tensor,torch.tensor], the clean prediction, certificate
	'''

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


# mask-ds
def masking_ds(inpt,labels,net,block_size,size_to_certify,thres=0.0):
	'''
	inpt 				torch.tensor, the input images in CWH format
	labels 				numpy.ndarray, the list of label 
	net 				torch.nn.module, the based model whose input is small pixel bands
	block_size 			int, the width of pixel bands
	size_to_certify     int, the patch size to be certified
	thres				float, the detection theshold ($T$). Note it is not `threshold` in ds()

	Return: 			[list,list], a list of provable analysis results and a list of clean prediction correctneses
	'''
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
	window_size = block_size + size_to_certify -1

	for i in range(len(labels)):
		local_feature = output_list[i].reshape([W,1,C])
		result=provable_masking(local_feature,labels[i],window_shape=[window_size,1],thres=thres)
		clean_pred=masking_defense(local_feature,window_shape=[window_size,1],thres=thres)
		result_list.append(result)
		clean_corr_list.append(clean_pred == labels[i])

	return result_list,clean_corr_list



# a extended version of provable_masking()
def provable_masking_large_mask(local_feature,label,clipping=-1,thres=0.,window_shape=[6,6],mask_shape=None):
	'''
	local_feature	numpy.ndarray, feature tensor in the shape of [feature_size_x,feature_size_y,num_cls]
	label 			int, true label
	clipping 		int/float, the positive clipping value ($c_h$ in the paper). If clipping < 0, treat clipping as np.inf
	thres 			float in [0,1], detection threshold. ($T$ in the paper)
	window_shape	list [int,int], the shape of malicious window
	mask_shape	list [int,int], the shape of mask window. If set to None, take the same value of window_shape

	Return 		int, provable analysis results (0: incorrect clean prediction; 1: possible attack found; 2: certified robustness )
	'''
	feature_size_x,feature_size_y,num_cls = local_feature.shape

	patch_size_x,patch_size_y = window_shape
	num_patch_x = feature_size_x - patch_size_x + 1
	num_patch_y = feature_size_y - patch_size_y + 1
	
	if mask_shape is None:
		mask_shape = window_shape
	mask_size_x,mask_size_y = mask_shape
	num_mask_x = feature_size_x - mask_size_x + 1
	num_mask_y = feature_size_y - mask_size_y + 1

	if clipping > 0:
		local_feature = np.clip(local_feature,0,clipping)
	else:
		local_feature = np.clip(local_feature,0,np.inf)

	global_feature = np.sum(local_feature,axis=(0,1))

	pred_list = np.argsort(global_feature,kind='stable')
	global_pred = pred_list[-1]

	if global_pred != label: #clean prediction is incorrect
		return 0

	# the sum of class evidence within mask window
	in_mask_sum_tensor = np.zeros([num_mask_x,num_mask_y,num_cls])
	for x in range(0,num_mask_x):
		for y in range(0,num_mask_y):
			in_mask_sum_tensor[x,y] = np.sum(local_feature[x:x+mask_size_x,y:y+mask_size_y,:],axis=(0,1))


	# the sum of class evidence within each possible malicious window
	in_patch_sum_tensor = np.zeros([num_patch_x,num_patch_y,num_cls])
	for x in range(0,num_patch_x):
		for y in range(0,num_patch_y):
			in_patch_sum_tensor[x,y,:] = np.sum(local_feature[x:x+patch_size_x,y:y+patch_size_y,:],axis=(0,1))

	#out_patch_sum_tensor = global_feature.reshape([1,1,num_cls]) - in_patch_sum_tensor

	idx = np.ones([num_cls],dtype=bool)
	idx[global_pred]=False

	for x in range(0,num_patch_x):
		for y in range(0,num_patch_y):

			# determine the upper bound of wrong class evidence
			cover_patch_mask_sum_tensor = in_mask_sum_tensor[max(0,x + patch_size_x - mask_size_x):min(x+1,num_mask_x),max(0,y + patch_size_y - mask_size_y):min(y+1,num_mask_y)]
			max_cover_patch_mask_sum = np.max(cover_patch_mask_sum_tensor,axis=(0,1))
			global_feature_patched = global_feature - max_cover_patch_mask_sum # $t-k$ in the proof of Lemma 2
			global_feature_patched[idx]/=(1 - thres) # $(t-k)/(1-T)$ in the proof of Lemma 2

			# determine the lower bound of true class evidence
			local_feature_pred_masked = local_feature[:,:,global_pred].copy()
			local_feature_pred_masked[x:x+patch_size_x,y:y+patch_size_y]=0
			in_mask_sum_pred_masked = in_mask_sum_tensor[:,:,global_pred].copy()
			# only need to recalculate the windows the are partially masked
			for xx in range(max(0,x - mask_size_x + 1),min(x + patch_size_x,num_mask_x)):
				for yy in range(max(0,y - mask_size_y + 1),min(y + patch_size_y,num_mask_y)):
					in_mask_sum_pred_masked[xx,yy]=local_feature_pred_masked[xx:xx+mask_size_x,yy:yy+mask_size_y].sum()
			max_mask_sum_pred = np.max(in_mask_sum_pred_masked)

			global_feature_patched[global_pred]= global_feature[global_pred] - in_patch_sum_tensor[x,y,global_pred]
			if max_mask_sum_pred / local_feature_pred_masked.sum() > thres: 
				global_feature_patched[global_pred]-=max_mask_sum_pred
			
			# determine if an attack is possible
			if np.argsort(global_feature_patched,kind='stable')[-1]!=label:
				return 1
	return 2 #provable robustness


# clipping based defense
def clipping_defense(local_feature,clipping=-1):
	'''
	local_feature	numpy.ndarray, feature tensor in the shape of [feature_size_x,feature_size_y,num_cls]
	clipping 		int/float, clipping value. If clipping < 0, use cbn clipping 

	Return 		 	int, provable analysis results (0: incorrect clean prediction; 1: possible attack found; 2: certified robustness )
	'''
	if clipping > 0:
		local_feature = np.clip(local_feature,0,clipping) #clipped with [0,clipping]
	else:
		local_feature = np.tanh(local_feature*0.05-1) # clipped with tanh (CBN)
	global_feature = np.mean(local_feature,axis=(0,1))
	global_pred = np.argmax(global_feature)

	return global_pred

# provable analysis for clipping based defense
def provable_clipping(local_feature,label,clipping=-1,window_shape=[6,6]):

	'''
	local_feature	numpy.ndarray, feature tensor in the shape of [feature_size_x,feature_size_y,num_cls]
	label 			int, true label
	clipping 		int/float, clipping value. If clipping < 0, use cbn clipping 

	window_shape	list [int,int], the shape of sliding window

	Return 		int, provable analysis results (0: incorrect clean prediction; 1: possible attack found; 2: certified robustness )
	'''
	feature_size_x,feature_size_y,num_cls = local_feature.shape
	window_size_x,window_size_y = window_shape
	num_window_x = feature_size_x - window_size_x + 1
	num_window_y = feature_size_y - window_size_y + 1

	if clipping > 0: 
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
