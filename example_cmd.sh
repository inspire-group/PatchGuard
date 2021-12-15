#install packages
pip install -r requirement.txt
#provable analysis with CBN and robust masking
python mask_bn.py --model bagnet17 --dataset imagenette --patch_size 32 --cbn #cbn with bagnet17 on imagenette
python mask_bn.py --model bagnet17 --dataset imagenette --patch_size 32 --m #mask-bn with bagnet17 on imagenette 
python mask_bn.py --model bagnet17 --dataset imagenet --patch_size 32 --cbn #cbn with bagnet17 on imagenet 
python mask_bn.py --model bagnet17 --dataset imagenet --patch_size 32 --m #mask-bn with bagnet17 on imagenet
python mask_bn.py --model bagnet17 --dataset cifar --patch_size 30 --cbn #cbn with bagnet17 on cifar 
python mask_bn.py --model bagnet17 --dataset cifar --patch_size 30 --m #mask-bn with bagnet17 on cifar
#mask-ds and ds
python mask_ds.py --dataset imagenette --patch_size 42 --ds #ds for imagenette
python mask_ds.py --dataset imagenette --patch_size 42 --m #mask-ds for imagenette
python mask_ds.py --dataset imagenet --patch_size 42 --ds #ds for imagenet
python mask_ds.py --dataset imagenet --patch_size 42 --m #mask-ds for imagenet
python mask_ds.py --dataset cifar --patch_size 5 --ds #ds for cifar
python mask_ds.py --dataset cifar --patch_size 5 --m #mask-ds for cifar

# patchguard++
python det_bn.py --det --model bagnet33 --tau 0.5 --patch_sie 32 --dataset imagenette # an example. the usage is similar to mask_bn.py and mask_ds.py
python det_bn.py --det --model bagnet33 --tau 0.7 --patch_sie 32 --dataset imagenette # you can try different threshold tau

#test model accuracy
python test_acc.py --model resnet50 --dataset imagenette #test accuracy of resnet50 on imagenette 
python test_acc.py --model resnet50 --dataset imagenet #test accuracy of resnet50 on imagenet 
python test_acc.py --model resnet50 --dataset cifar #test accuracy of resnet50 on cifar
python test_acc.py --model bagnet17 --dataset imagenette #test accuracy of bagnet17 on imagenette (similar for imagenet,cifar)
python test_acc.py --model bagnet33 --dataset imagenette #test accuracy of bagnet33 on imagenette (similar for imagenet,cifar)
python test_acc.py --model bagnet9 --dataset imagenette #test accuracy of bagnet9 on imagenet (similar for imagenet)
python test_acc.py --model bagnet17 --dataset imagenette --clip 15 #test accuracy of bagnet17 (clipped with [0,15]) on imagenette (similar for imagenet,cifar)
python test_acc.py --model bagnet17 --dataset imagenette --aggr median #test accuracy of bagnet17 with median aggregation on imagenette (similar for imagenet,cifar)
python test_acc.py --model bagnet17 --dataset imagenette --aggr cbn #test accuracy of bagnet17 with cbn clipping on imagenette (similar for imagenet,cifar)
#empirical untargeted attack
python patch_attack.py --model bagnet17 --dataset imagenette --patch_size 31 #untargeted attack against bagnet17
python patch_attack.py --model bagnet17 --dataset imagenette --patch_size 31 --aggr cbn #untargeted attack against bagnet17 with cbn clipping
#train model
python train_imagenette.py --model_name bagnet17_nette.pth  --epoch 20 #train model on imagenette
python train_imagenette.py --model_name bagnet17_nette.pth --aggr adv --epoch 20 #train model on imagenette with provable adversarial training
python train_cifar.py --lr 0.01 #train cifar model
python train_cifar.py --resume --lr 0.001 #resume cifar model training with a different learning rate


