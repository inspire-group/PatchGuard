# PatchGuard: A Provably Robust Defense against Adversarial Patches via Small Receptive Fields and Masking
By [Chong Xiang](http://xiangchong.xyz/), [Arjun Nitin Bhagoji](http://www.princeton.edu/~abhagoji/), [Vikash Sehwag](https://vsehwag.github.io/), [Prattek Mittal](https://www.princeton.edu/~pmittal/)

Code for paper "PatchGuard: A Provably Robust Defense against Adversarial Patches via Small Receptive Fields and Masking" [arXiv](https://arxiv.org/abs/2005.10884)

<img src="./doc/overview.png" width="100%" alt="defense overview pipeline" align=center>

## Requirements
The code is tested with Python 3.8 and PyTorch 1.7.0. The complete list of required packages are available in `requirement.txt`, and can be installed with `pip install -r requirement.txt`. The code should be compatible with other versions of packages.

## Files
```shell
├── README.md                        #this file 
├── requirement.txt                  #required package
├── example_cmd.sh                   #example command to run the code
├── mask_bn.py             	 		 #mask-bn for imagenet/imagenet/cifar
├── mask_ds.py              		 #mask-ds/ds for imagenet/imagenet/cifar
├── nets
|   ├── bagnet.py                    #modified bagnet model for mask-bn
|   ├── resnet.py                    #modified resnet model 
|   ├── dsresnet_imgnt.py            #ds-resnet-50 for imagenet(te)
|   └── dsresnet_cifar.py            #ds-resnet-18 for cifar
├── utils
|   ├── defense_utils.py             #utils for different defenses
|   ├── normalize_utils.py           #utils for nomrlize images stored in numpy array (unused in the paper)
|   ├── cutout.py                    #utils for CUTOUT training (unused)
|   └── progress_bar.py              #progress bar (used in train_cifar.py)
| 
├── misc                             #useful scripts; move them to the main directory for execution
|   ├── test_acc.py         		 #test clean accuracy of resnet/bagnet on imagenet/imagenette/cifar; support clipping, median operations
|   ├── train_imagenet.py            #train resnet/bagnet for imagenet
|   ├── train_imagenette.py          #train resnet/bagnet for imagenette
|   ├── train_cifar.py               #train resnet/bagnet for cifar
|   ├── patch_attack_bagnet.py       #empirically (untargeted) attack resnet/bagnet trained on imagenet/imagenette/cifar
|   ├── PatchAttacker.py             #untargeted adversarial patch attack 
|
├── data   
|   ├── imagenet             		 #data directory for imagenet
|   ├── imagenette           		 #data directory for imagenette
|   └── cifar						 #data directory for cifar
|
└── checkpoints                      #directory for checkpoints
    ├── README.md                    #details of each checkpoint
    └── ...                          #model checkpoints
```
## Datasets
- [ImageNet](http://www.image-net.org/) (ILSVRC2012)
- [ImageNette](https://github.com/fastai/imagenette) ([Full size](https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz))
- [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)

## Usage
- See **Files** for details of each file. 

- Download data in **Datasets** to `data/`.

- (optional) Download checkpoints from Google Drive [link](https://drive.google.com/drive/folders/1u5RsCuZNf7ddWW0utI4OrgWGmJCUDCuT?usp=sharing) and move them to `checkpoints`.

- See `example_cmd.sh` for example commands for running the code.

If anything is unclear, please open an issue or contact Chong Xiang (cxiang@princeton.edu).

## Notes
- 02/11/2020 - Merged a few scripts; fixed a few minor bugs
- 10/17/2020 - Updated old checkpoints. Please download the new checkpoints from Google Drive [link](https://drive.google.com/drive/folders/1u5RsCuZNf7ddWW0utI4OrgWGmJCUDCuT?usp=sharing) for better model performance. Note that checkpoints for 1000-class ImageNet are also available now. Also a few minor updates to the source code. 
- 08/01/2020 - A major update to `defense_utils.py`. Please check the latest version of paper on [arXiv](https://arxiv.org/abs/2005.10884) and use the new provable analysis in `defense_utils.py`.

## Related Repositories
- [certifiedpatchdefense](https://github.com/Ping-C/certifiedpatchdefense)
- [patchSmoothing](https://github.com/alevine0/patchSmoothing)
- [bag-of-local-features-models](https://github.com/wielandbrendel/bag-of-local-features-models)
