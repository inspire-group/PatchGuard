# PatchGuard: Provable Defense against Adversarial Patches Using Masks on Small Receptive Fields
By [Chong Xiang](http://xiangchong.xyz/), [Arjun Nitin Bhagoji](http://www.princeton.edu/~abhagoji/), [Vikash Sehwag](https://scholar.princeton.edu/vvikash/home), [Prattek Mittal](https://www.princeton.edu/~pmittal/)

Code for paper "PatchGuard: Provable Defense against Adversarial Patches Using Masks on Small Receptive Fields" [arXiv](https://arxiv.org/abs/2005.10884)

<img src="./doc/overview.png" width="100%" alt="defense overview pipeline" align=center>

## Requirements
The code is tested with Python 3.6 and PyTorch 1.3.0. The complete list of required packages are available in `requirement.txt`, and can be installed with `pip install -r requirement.txt`. The code should be compatible with other versions of packages.

## Files
```shell
├── README.md                        #this file 
├── requirement.txt                  #required package
├── example_cmd.sh                   #example command to run the code
├── mask_bn_imagenet.py              #mask-bn for imagenet(te)
├── mask_bn_cifar.py                 #mask-bn for cifar
├── mask_ds_imagenet.py              #mask-ds for imagenet(te)
├── mask_ds_cifar.py                 #mask-ds for cifar
├── nets
|   ├── bagnet.py                    #modified bagnet model for mask-bn
|   ├── resnet.py                    #modified resnet model for mask-bn
|   ├── dsresnet_imgnt.py            #ds-resnet-50 for imagenet(te)
|   └── dsresnet_cifar.py            #ds-resnet-18 for cifar
├── utils
|   ├── defense_utils.py             #utils for different defenses
|   ├── normalize_utils.py           #utlis for nomrlize images stored in numpy array (unused in the paper)
|   ├── cutout.py                    #utlis for CUTOUT training (unused)
|   └── progress_bar.py              #progress bar (used in train_cifar.py)
| 
├── misc                             #useful scripts; move them to the main directory for execution
|   ├── test_acc_imagenet.py         #test clean accuracy of resnet/bagnet on imagenet(te); support clipping, median operations
|   ├── test_acc_cifar.py            #test clean accuracy of resnet/bagnet on cifar; support clipping, median operations
|   ├── train_imagenet.py            #train resnet/bagnet for imagenet(te)
|   ├── train_cifar.py               #train resnet/bagnet for cifar
|   ├── patch_attack_imagenet.py     #empirically attack resnet/bagnet trained on imagenet(te)
|   ├── patch_attack_cifar.py        #empirically attack resnet/bagnet trained on cifar
|   ├── PatchAttacker.py             #untargeted adversarial patch attack 
|   ├── ds_imagenet.py               #ds for imagenet(te)
|   └── ds_cifar.py                  #ds for imagenet(te)
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

- Download data in **Datasets** and specify the data directory to the code.

- (optional) Download checkpoints from Google Drive [link](https://drive.google.com/drive/folders/1u5RsCuZNf7ddWW0utI4OrgWGmJCUDCuT?usp=sharing) and move them to `checkpoints`.

- See `example_cmd.sh` for example commands for running the code.

If anything is unclear, please open an issue or contact Chong Xiang (cxiang@princeton.edu).

## Related Repositories
- [certifiedpatchdefense](https://github.com/Ping-C/certifiedpatchdefense)
- [patchSmoothing](https://github.com/alevine0/patchSmoothing)
- [bag-of-local-features-models](https://github.com/wielandbrendel/bag-of-local-features-models)
