# LBA-OSR

## Setup

1. Install CLIP  
```bash
pip install git+https://github.com/openai/CLIP.git
```

2. Install Dassl  
You need Dassl to run CoOp. Follow the instructions at <https://github.com/KaiyangZhou/Dassl.pytorch#installation>. 

## Download pretrained weights

Download the pretrained weights at <https://github.com/hushon/lba-osr/releases/download/v0.0.0/lba-osr-pretrained-coop.zip>. 
Download and unzip at the root directory: `lba-osr/output/`.

## Running code

Zero-shot OSR classification

```base
python osr.py --dataset <DATASET> --loss <LOSS> --eval
```

Option --loss can be one of ARPLoss/RPLoss/GCPLoss/Softmax/SoftmaxPlus. --dataset is one of mnist/svhn/cifar10/cifar100/tiny_imagenet. 

