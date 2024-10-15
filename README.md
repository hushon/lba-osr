# LBA-OSR

## Setup

You need Dassl to run CoOp. Follow the instructions at <https://github.com/KaiyangZhou/Dassl.pytorch#installation>. 

## Download pretrained weights

Download the pretrained weights at <https://github.com/hushon/lba-osr/releases/download/v0.0.0/lba-osr-pretrained-coop.zip>. 
Download and unzip at the root directory: `lba-osr/output/`.

## Running code

```base
python osr.py --dataset <DATASET> --loss <LOSS>
```

Option --loss can be one of ARPLoss/RPLoss/GCPLoss/Softmax/SoftmaxPlus. --dataset is one of mnist/svhn/cifar10/cifar100/tiny_imagenet. 

