#install packages
pip install -r requirement.txt
#test model accuracy
python test_acc_imagenet.py --model resnet50_nette #test accuracy of resnet50 on imagenette (similar for ImageNet)
python test_acc_imagenet.py --model bagnet17_nette #test accuracy of bagnet17 on imagenette (similar for ImageNet)
python test_acc_imagenet.py --model bagnet33_nette #test accuracy of bagnet33 on imagenette (similar for ImageNet)
python test_acc_imagenet.py --model bagnet9_nette #test accuracy of bagnet9 on imagenet (similar for ImageNet)
python test_acc_imagenet.py --model bagnet17_nette --clip 15 #test accuracy of bagnet17 (clipped with [0,15]) on imagenette (similar for ImageNet)
python test_acc_imagenet.py --model bagnet17_nette --aggr median #test accuracy of bagnet17 with median aggregation on imagenette (similar for ImageNet)
python test_acc_imagenet.py --model bagnet17_nette --aggr cbn #test accuracy of bagnet17 with cbn clipping on imagenette (similar for ImageNet)
#provable analysis with CBN and robust masking
python mask_bn_imagenet.py --model bagnet17_nette --patch_size 31 --cbn #cbn with bagnet17 on imagenette (similar for ImageNet)
python mask_bn_imagenet.py --model bagnet17_nette --patch_size 31 --m #mask-bn with bagnet17 on imagenette (similar for ImageNet)
#empirical untargeted attack
python patch_attack_imagenet.py --model bagnet17_nette --patch_size 31 #untargeted attack against bagnet17
python patch_attack_imagenet.py --model bagnet17_nette --patch_size 31 --aggr cbn #untargeted attack against bagnet17 with cbn clipping
#train model
python train_imagenette.py --model_name bagnet17_nette.pth  --epoch 20 #train model on imagenette
python train_imagenette.py --model_name bagnet17_nette.pth --aggr adv --epoch 20 #train model on imagenette with provable adversarial training

#similar usage for cifar
python test_acc_cifar.py --model resnet50_192_cifar
python test_acc_cifar.py --model bagnet17_192_cifar --clip 15
python mask_bn_cifar.py --model bagnet17_192_cifar --patch_size 30 --cbn
python mask_bn_cifar.py --model bagnet17_192_cifar --patch_size 30 --m
python patch_attack_cifar.py --model resnet50_192_cifar --patch_size 30
python train_cifar.py --lr 0.01 #train cifar model
python train_cifar.py --resume --lr 0.001 #resume cifar model training with a different learning rate

#mask-ds and ds
python mask_ds_imagenet.py --patch_size 42 --ds #ds for imagenet(te)
python mask_ds_imagenet.py --patch_size 42 --m #mask-ds for imagenet(te)
python mask_ds_cifar.py --patch_size 5 --ds #ds for cifar
python mask_ds_cifar.py --patch_size 5 --m #mask-ds for cifar
