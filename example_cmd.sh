#install packages
pip install -r requirement.txt
#test model accuracy
python test_acc_imagenet.py --model resnet50 #test accuracy of resnet50 on imagenet(te)
python test_acc_imagenet.py --model bagnet17 #test accuracy of bagnet17 on imagenet(te)
python test_acc_imagenet.py --model bagnet33 #test accuracy of bagnet33 on imagenet(te)
python test_acc_imagenet.py --model bagnet9 #test accuracy of bagnet9 on imagenet(te)
python test_acc_imagenet.py --model bagnet17 --clip 15 #test accuracy of bagnet17 (clipped with [0,15]) on imagenet(te)
python test_acc_imagenet.py --model bagnet17 --aggr median #test accuracy of bagnet17 with median aggregation on imagenet(te)
python test_acc_imagenet.py --model bagnet17 --aggr cbn #test accuracy of bagnet17 with cbn clipping on imagenet(te)
#provable analysis with CBN and robust masking
python mask_bn_imagenet.py --model bagnet17 --patch_size 31 --cbn #cbn with bagnet17 on imagenet(te)
python mask_bn_imagenet.py --model bagnet17 --patch_size 31 --m #mask-bn with bagnet17 on imagenet(te)
#empirical untargeted attack
python patch_attack_imagenet.py --model bagnet17 --patch_size 31 #untargeted attack against bagnet17
python patch_attack_imagenet.py --model bagnet17 --patch_size 31 --aggr cbn #untargeted attack against bagnet17 with cbn clipping
#train model
python train_imagenet.py --model_name bagnet17.pth  --epoch 20 #train model on imagenette

#similar usage for cifar
python test_acc_cifar.py --model resnet50_192
python test_acc_cifar.py --model bagnet17_192 --clip 15
python mask_bn_cifar.py --model bagnet17_192 --patch_size 30 --cbn
python mask_bn_cifar.py --model bagnet17_192 --patch_size 30 --m
python patch_attack_cifar.py --model resnet50_192 --patch_size 30
python train_cifar.py --lr 0.01 #train cifar model
python train_cifar.py --resume --lr 0.001 #resume cifar model training with a different learning rate

#mask-ds and ds
python mask_ds_imagenet.py --patch_size 42 --ds #ds for imagenet(te)
python mask_ds_imagenet.py --patch_size 42 --m #mask-ds for imagenet(te)
python mask_ds_cifar.py --patch_size 5 --ds #ds for cifar
python mask_ds_cifar.py --patch_size 5 --m #mask-ds for cifar
