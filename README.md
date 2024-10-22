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
python osr.py --dataset <DATASET> --loss <LOSS> --eval --clip-model ViT-B/32 --coop coop
```

Option `--loss` can be one of `ARPLoss/RPLoss/GCPLoss/Softmax/SoftmaxPlus`. --dataset is one of `mnist/svhn/cifar10/cifar100/tiny_imagenet`.  
`--clip-model` can be one of `["RN50", "ViT-B/32", "ViT-B/16"]`.  
` --coop` can be one of `['vanilla', 'coop', 'cocoop', 'cocoop2']`.  


## References

This codebase is based on following codes: 
- <https://github.com/iCGY96/ARPL>
- <https://github.com/KaiyangZhou/CoOp>