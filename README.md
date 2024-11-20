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

Option `--loss` can be one of `ARPLoss/RPLoss/GCPLoss/Softmax/SoftmaxPlus`. `--dataset` is one of `mnist/svhn/cifar10/cifar100/tiny_imagenet`.  
`--clip-model` can be one of `["RN50", "ViT-B/32", "ViT-B/16"]`.  
`--coop` can be one of `['vanilla', 'coop_c16', 'cocoop_c4', 'cocoop_c16']`.  
When using `SoftmaxPlus` loss, specify `--oe-mode` in one of `[None, 'random', 'wordnet', 'coreset']`.  

*Example command*
```bash
python osr.py --dataset cifar10 --loss SoftmaxPlus --eval --clip-model ViT-B/32 --coop coop_c16 --oe-mode random
```


## Open-vocabulary exposure

Open-vocabulary exposure (OE) has following modes. 

| OE mode | description |
|---|---|
| `random` | randomly sample 1000 OE classes from IM21k classes |
| `wordnet` | sample 1000 OE classes from IM21k classes using WordNet hierarchy |
| `coreset` | sample 1000 OE classes from IM21k classes using text embedding and greedy coreset selection algorithm |


## LBA keyframe selection

*Example input*: `./lba_sample_input/input.json`

Run keyframe selection 
```bash
python lba-keyframe-selection-demo.py --dataset lba --loss SoftmaxPlus --eval --clip-model ViT-B/32 --coop coop_c16 --oe-mode random
```

This will select the best-matching keyframes up to 10 frames per QA pair. 

The output is saved at `./lba_sample_input/output.json`.


## References

This codebase is based on following codes: 
- <https://github.com/iCGY96/ARPL>
- <https://github.com/KaiyangZhou/CoOp>