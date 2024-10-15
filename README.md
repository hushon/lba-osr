# lba-osr

## Setup

You need Dassl to run CoOp. Follow the instructions at <https://github.com/KaiyangZhou/Dassl.pytorch#installation>. 

## Running code

```base
python osr.py --dataset <DATASET> --loss <LOSS>
```

Option --loss can be one of ARPLoss/RPLoss/GCPLoss/Softmax/SoftmaxPlus. --dataset is one of mnist/svhn/cifar10/cifar100/tiny_imagenet. 

