import os
import argparse
import datetime
import time
import csv
import pandas as pd
import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn

from models import gan
from models.models import classifier32, classifier32ABN
from datasets.osr_dataloader import MNIST_OSR, CIFAR10_OSR, CIFAR100_OSR, SVHN_OSR, Tiny_ImageNet_OSR
from utils import Logger, save_networks, load_networks
from core import train, train_cs, test
from nltk.corpus import wordnet as wn
import numpy as np


# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True


parser = argparse.ArgumentParser("Training")

# Dataset
parser.add_argument('--dataset', type=str, default='mnist', help="mnist | svhn | cifar10 | cifar100 | tiny_imagenet")
parser.add_argument('--dataroot', type=str, default='/home/hyounguk.shon/data')
parser.add_argument('--outf', type=str, default='./log')
parser.add_argument('--out-num', type=int, default=50, help='For CIFAR100')

# optimization
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.1, help="learning rate for model")
parser.add_argument('--gan_lr', type=float, default=0.0002, help="learning rate for gan")
parser.add_argument('--max-epoch', type=int, default=100)
parser.add_argument('--stepsize', type=int, default=30)
parser.add_argument('--temp', type=float, default=1.0, help="temp")
parser.add_argument('--num-centers', type=int, default=1)

# model
parser.add_argument('--weight-pl', type=float, default=0.1, help="weight for center loss")
parser.add_argument('--beta', type=float, default=0.1, help="weight for entropy loss")
parser.add_argument('--model', type=str, default='classifier32')
parser.add_argument('--clip-model', type=str, default='ViT-B/32', choices=["RN50", "ViT-B/32", "ViT-B/16"])
parser.add_argument('--coop', type=str, default='coop', choices=['vanilla', 'coop_c16', 'cocoop_c4', 'cocoop_c16'])
parser.add_argument('--oe-mode', type=str, default=None, choices=[None, 'random', 'wordnet', 'coreset'])

# misc
parser.add_argument('--nz', type=int, default=100)
parser.add_argument('--ns', type=int, default=1)
parser.add_argument('--eval-freq', type=int, default=1)
parser.add_argument('--print-freq', type=int, default=100)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--use-cpu', action='store_true')
parser.add_argument('--save-dir', type=str, default='../log')
parser.add_argument('--loss', type=str, default='ARPLoss')
parser.add_argument('--eval', action='store_true', help="Eval", default=False)
parser.add_argument('--cs', action='store_true', help="Confusing Sample", default=False)


import json
import torch
from torch.utils.data import Dataset
from pathlib import Path
from torchvision.datasets.folder import default_loader


class LbaDataset(Dataset):
    def __init__(
            self,
            json_path="./lba_sample_input/input.json",
            data_root="/mnt/data/DramaQA/AnotherMissOh_images/dramaqa_frames/",
            image_transform = None,
            question_transform = None
        ):
        with open(json_path, "r") as fp:
            self.json_data = json.load(fp)
        self.data_root = Path(data_root)
        self.image_transform = image_transform
        self.question_transform = question_transform

        """
        json_data = [
            {'videoType': 'scene',
            'answers': ['Since Dokyung wanted to kick the old man.',
            'Just because Dokyung wanted to buy some snacks from the old man.',
            "Because Dokyung tried to give Dokyung's umbrella to the old man.",
            'As Dokyung wanted to take the clothes away from the old man.',
            'It was because Dokyung wanted to run away from Haeyoung1.'],
            'vid': 'AnotherMissOh17_001_0000',
            'qid': 13041,
            'shot_contained': [25, 49],
            'q_level_mem': 3,
            'que': 'Why did Dokyung go to the old man?',
            'q_level_logic': 4},
            ...
        ]
        """

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, idx):
        json_sample = self.json_data[idx]
        data_root = self.data_root

        # get the image_paths
        if json_sample['videoType'] == "shot":  # sample consists of a single shot
            episodeName, sceneNum, shotNum = json_sample['vid'].split("_")
            assert shotNum == f"{json_sample['shot_contained'][0]:04d}"
            image_paths = list(data_root.glob(f"{episodeName}/{sceneNum}/{shotNum}/*.jpg"))
        
        elif json_sample['videoType'] == "scene":  # sample consists of multiple shots
            episodeName, sceneNum, shotNum = json_sample['vid'].split("_")
            firstShot, LastShot = json_sample['shot_contained']
            assert shotNum == "0000"
            image_paths = []
            for shot in range(firstShot, LastShot+1):
                image_paths.extend(data_root.glob(f"{episodeName}/{sceneNum}/{shot:04d}/*.jpg"))
        
        else:
            raise ValueError

        image_paths.sort()
        images = [default_loader(path) for path in image_paths]

        question_text = json_sample['que']

        if self.image_transform is not None:
            images = [self.image_transform(img) for img in images]
        
        if self.question_transform is not None:
            question_text = self.question_transform(json_sample['que'])

        return {
            "images": images,
            "question_text": question_text,
            "image_paths": image_paths,
        }




def main_worker(options):
    torch.manual_seed(options['seed'])
    os.environ['CUDA_VISIBLE_DEVICES'] = options['gpu']
    use_gpu = torch.cuda.is_available()
    if options['use_cpu']: use_gpu = False

    if use_gpu:
        print("Currently using GPU: {}".format(options['gpu']))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(options['seed'])
    else:
        print("Currently using CPU")

    # Dataset
    print("{} Preparation".format(options['dataset']))
    if 'mnist' in options['dataset']:
        Data = MNIST_OSR(known=options['known'], dataroot=options['dataroot'], batch_size=options['batch_size'], img_size=options['img_size'])
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
    elif 'cifar10' == options['dataset']:
        Data = CIFAR10_OSR(known=options['known'], dataroot=options['dataroot'], batch_size=options['batch_size'], img_size=options['img_size'])
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
    elif 'svhn' in options['dataset']:
        Data = SVHN_OSR(known=options['known'], dataroot=options['dataroot'], batch_size=options['batch_size'], img_size=options['img_size'])
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
    elif 'cifar100' in options['dataset']:
        Data = CIFAR10_OSR(known=options['known'], dataroot=options['dataroot'], batch_size=options['batch_size'], img_size=options['img_size'])
        trainloader, testloader = Data.train_loader, Data.test_loader
        out_Data = CIFAR100_OSR(known=options['unknown'], dataroot=options['dataroot'], batch_size=options['batch_size'], img_size=options['img_size'])
        outloader = out_Data.test_loader
    elif 'tiny_imagenet' in options['dataset']:
        Data = Tiny_ImageNet_OSR(known=options['known'], dataroot=options['dataroot'], batch_size=options['batch_size'], img_size=options['img_size'])
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
    elif 'lba' in options['dataset']:
        from torch.utils.data import DataLoader
        from torchvision import transforms
        image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

        import spacy
        nlp = spacy.load("en_core_web_sm")  # spaCy 모델 로드

        def question_transform(sentence):
            """
            문장에서 명사, 동사 추출

            Example
                Input: "Why did Sungjin tell Haeyoung1 not to talk with flowers?"
                Returns: ["tell talk flowers"]
            """
            doc = nlp(sentence)  # 텍스트 분석
            keywords = [token.text for token in doc if token.pos_ in ["NOUN", "VERB"]]  # 명사, 동사 추출
            keywords = " ".join(keywords)  # string 으로 변환
            return keywords
        Data = LbaDataset(image_transform=image_transform, question_transform=question_transform)
        trainloader, testloader, outloader = None, DataLoader(Data), None
    
    # options['num_classes'] = Data.num_classes

    # # Model
    # print("Creating model: {}".format(options['model']))
    # if options['cs']:
    #     net = classifier32ABN(num_classes=options['num_classes'])
    # else:
    #     net = classifier32(num_classes=options['num_classes'])
    # feat_dim = 128

    # if options['cs']:
    #     print("Creating GAN")
    #     nz, ns = options['nz'], 1
    #     if 'tiny_imagenet' in options['dataset']:
    #         netG = gan.Generator(1, nz, 64, 3)
    #         netD = gan.Discriminator(1, 3, 64)
    #     else:
    #         netG = gan.Generator32(1, nz, 64, 3)
    #         netD = gan.Discriminator32(1, 3, 64)
    #     fixed_noise = torch.FloatTensor(64, nz, 1, 1).normal_(0, 1)
    #     criterionD = nn.BCELoss()

    # # Loss
    # options.update(
    #     {
    #         'feat_dim': feat_dim,
    #         'use_gpu':  use_gpu
    #     }
    # )

    Loss = importlib.import_module('loss.'+options['loss'])
    criterion = getattr(Loss, options['loss'])(**options)

    # if use_gpu:
    #     net = nn.DataParallel(net).cuda()
    #     criterion = criterion.cuda()
    #     if options['cs']:
    #         netG = nn.DataParallel(netG, device_ids=[i for i in range(len(options['gpu'].split(',')))]).cuda()
    #         netD = nn.DataParallel(netD, device_ids=[i for i in range(len(options['gpu'].split(',')))]).cuda()
    #         fixed_noise.cuda()

    # model_path = os.path.join(options['outf'], 'models', options['dataset'])
    # if not os.path.exists(model_path):
    #     os.makedirs(model_path)
    
    # if options['dataset'] == 'cifar100':
    #     model_path += '_50'
    #     file_name = '{}_{}_{}_{}_{}'.format(options['model'], options['loss'], 50, options['item'], options['cs'])
    # else:
    #     file_name = '{}_{}_{}_{}'.format(options['model'], options['loss'], options['item'], options['cs'])

    # # net, criterion = load_networks(net, model_path, file_name, criterion=criterion)

    if options['dataset'] == 'tiny_imagenet':
        with open('/home/hyounguk.shon/data/tiny_imagenet/tiny-imagenet-200/class_names.txt', 'r') as f:
            classes = [line.strip() for line in f]
        classnames = [classes[i] for i in Data.known] # Tiny ImageNet
    elif 'cifar' in options['dataset']:
        classnames = [testloader.dataset.classes[i] for i in Data.known] # CIFAR
    elif options['dataset'] == 'lba':
        classnames = ['placeholder']
    else:
        raise NotImplementedError


    def get_open_classnames_im21k():
        import random
        # from coop_clip.imagenet_classnames import classnames as open_classnames
        # open_classnames = list(open_classnames.values())
        with open('./coop_clip/imagenet21k_wordnet_lemmas.txt', 'r') as file:
            open_classnames = [line.strip() for line in file.readlines()]
        random.Random(42).shuffle(open_classnames)
        open_classnames = open_classnames[:1000]
        return open_classnames

    # Generate open classnames using WordNet (Ontology-based approach)
    def get_hypernyms(synset, depth=3):
        hypernyms = set()
        queue = [(synset, 0)]
        while queue:
            current_synset, current_depth = queue.pop(0)
            if current_depth >= depth:
                continue
            for hypernym in current_synset.hypernyms():
                if hypernym not in hypernyms:
                    hypernyms.add(hypernym)
                    queue.append((hypernym, current_depth + 1))
        return list(hypernyms)

    def get_open_classnames_ontology(known_classnames, num_classes=1000):
        open_classnames = set()
        for classname in known_classnames:
            synsets = wn.synsets(classname)
            if synsets:
                synset = synsets[0]
                hypernyms = get_hypernyms(synset)
                for hypernym in hypernyms:
                    open_classnames.add(hypernym.lemmas()[0].name())
            if len(open_classnames) >= num_classes:
                break
        return list(open_classnames)[:num_classes]

    # Generate open classnames using Diversity Maximization approach with CLIP embeddings
    def get_open_classnames_diversity_maximization(known_classnames, clip_model, num_classes=1000):
        with open('./coop_clip/imagenet21k_wordnet_lemmas.txt', 'r') as file:
            open_classnames = [line.strip() for line in file.readlines()]

        all_classnames = known_classnames + open_classnames

        model = get_coop_model(all_classnames, clip_model)
        with torch.no_grad():
            text_features = model.text_encoder(
                model.prompt_learner(),
                model.tokenized_prompts
            )
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        '''
        checkpoint load 는 어떻게?
        '''

        from coreset import greedy_coreset_sampling
        initial_indices = list(range(len(known_classnames)))
        coreset_indices = greedy_coreset_sampling(text_features, len(known_classnames) + num_classes, initial_indices, metric='dot_product')
        coreset_indices = coreset_indices[len(known_classnames):]

        selected_open_classnames = [all_classnames[i] for i in coreset_indices]

        return selected_open_classnames

    '''CustomCLIP 에서 text feature 받아오는법:
        model = CustomCLIP(classnames, clip_model)
        text_features = model.text_encoder(
            model.prompt_learner(),
            model.tokenized_prompts
        )
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    '''

    def get_clip_model():
        import coop
        if options['clip_model'] == "RN50":
            clip_model = coop.load_clip_to_cpu("RN50").float()
        elif options['clip_model'] == "ViT-B/32":
            clip_model = coop.load_clip_to_cpu("ViT-B/32").float()
        elif options['clip_model'] == "ViT-B/16":
            clip_model = coop.load_clip_to_cpu("ViT-B/16").float()
        else:
            raise ValueError("Unsupported clip model: {}".format(options['clip_model']))
        return clip_model

    def get_coop_model(classnames, clip_model):
        import coop
        if options['coop'] == 'vanilla':
            model = coop.VanillaCLIP(classnames, options['clip_model'])
        elif options['coop'] == 'coop_c16':
            model = coop.CustomCLIP(classnames, clip_model)
        elif options['coop'] in ['cocoop_c4', 'cocoop_c16']:
            import cocoop
            cfg = None
            model = cocoop.CustomCLIP(cfg, classnames, clip_model)
        else:
            raise ValueError()

        if options['coop'] == 'coop_c16':
            if options['clip_model'] == "RN50":
                checkpoint = coop.load_checkpoint('output/imagenet/CoOp/rn50_ep50_16shots/nctx16_cscFalse_ctpend/seed1/prompt_learner/model.pth.tar-50')
            elif options['clip_model'] == "ViT-B/32":
                checkpoint = coop.load_checkpoint('output/imagenet/CoOp/vit_b32_ep50_16shots/nctx16_cscFalse_ctpend/seed1/prompt_learner/model.pth.tar-50')
            else:
                raise ValueError()
        elif options['coop'] == 'cocoop_c4':
            if options['clip_model'] == "ViT-B/16":
                checkpoint = coop.load_checkpoint('output/base2new/train_base/imagenet/shots_16/CoCoOp/vit_b16_c4_ep10_batch1_ctxv1/seed1/prompt_learner/model.pth.tar-10')
            else:
                raise ValueError()
        elif options['coop'] == 'cocoop_c16':
            if options['clip_model'] == "ViT-B/16":
                checkpoint = coop.load_checkpoint('output/base2new/train_base/imagenet/shots_16/CoCoOp/vit_b16_c16_ep10_batch1/seed1/prompt_learner/model.pth.tar-10')
            else:
                raise ValueError()
        elif options['coop'] == 'vanilla':
            pass
        else:
            raise ValueError()
        
        if 'coop' in options['coop']:
            state_dict = checkpoint['state_dict']
            '''CoCoOp 에서 meta_net 제대로 가져오는지 확인 필요'''

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]
            print(model.prompt_learner.load_state_dict(state_dict, strict=False))
            model.prompt_learner.float()

        return model

    clip_model = get_clip_model()


    if options['loss'] == "SoftmaxPlus":  ## use open_classnames
        if options['oe_mode'] == 'random':
            open_classnames = get_open_classnames_im21k()
        elif options['oe_mode'] == 'wordnet':
            open_classnames = get_open_classnames_ontology(classnames)
        elif options['oe_mode'] == 'coreset':
            open_classnames = get_open_classnames_diversity_maximization(classnames, clip_model, num_classes=1000)
        else:
            raise ValueError()
        model = get_coop_model(classnames+open_classnames, clip_model)
        criterion.num_classes = len(classnames)
    else:
        model = get_coop_model(classnames, clip_model)

    net = model.eval().cuda()


    import tqdm.auto as tqdm
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=False, dtype=torch.bfloat16):
            results = []
            for sample in tqdm.tqdm(Data):
                images = torch.stack(sample["images"]).cuda()
                text = sample["question_text"]
                image_paths = sample["image_paths"]

                net.initialize_classnames([text]+open_classnames)
                net.cuda()

                _, logits = net(images, True)
                score = logits.div(1.0).softmax(1)[:, 0].cpu()

                # get top-matching images
                num_k = len(images) // 2  # select top 50%
                num_k = min(num_k, 10)  # truncate if over 10 images
                top_index = torch.topk(score, num_k, sorted=False).indices.sort().values
                best_image_paths = [str(image_paths[i]) for i in top_index]

                results.append({
                    "keyframes": best_image_paths,
                    "scores": score[top_index].tolist(),
                })
            
            # write best_image_paths to a json file
            with open("./lba_sample_input/output.json", "w") as fp:
                json.dump(results, fp, indent=4)
            breakpoint()


        #     results = test(net, criterion, testloader, outloader, epoch=0, **options)
        # print("Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(results['ACC'], results['AUROC'], results['OSCR']))

    return results


if __name__ == '__main__':
    args = parser.parse_args()
    options = vars(args)
    options['dataroot'] = os.path.join(options['dataroot'], options['dataset'])
    img_size = 32
    results = dict()
    
    # from split import splits_2020 as splits
    
    # for i in range(len(splits[options['dataset']])):
    #     known = splits[options['dataset']][len(splits[options['dataset']])-i-1]
    #     if options['dataset'] == 'cifar100':
    #         unknown = splits[options['dataset']+'-'+str(options['out_num'])][len(splits[options['dataset']])-i-1]
    #     elif options['dataset'] == 'tiny_imagenet':
    #         img_size = 64
    #         options['lr'] = 0.001
    #         unknown = list(set(list(range(0, 200))) - set(known))
    #     else:
    #         unknown = list(set(list(range(0, 10))) - set(known))

    #     options.update(
    #         {
    #             'item':     i,
    #             'known':    known,
    #             'unknown':  unknown,
    #             'img_size': img_size
    #         }
    #     )

    #     dir_name = '{}_{}'.format(options['model'], options['loss'])
    #     dir_path = os.path.join(options['outf'], 'results', dir_name)
    #     if not os.path.exists(dir_path):
    #         os.makedirs(dir_path)

    #     if options['dataset'] == 'cifar100':
    #         file_name = '{}_{}.csv'.format(options['dataset'], options['out_num'])
    #     else:
    #         file_name = options['dataset'] + '.csv'

    #     res = main_worker(options)
    #     res['unknown'] = unknown
    #     res['known'] = known
    #     results[str(i)] = res
    #     df = pd.DataFrame(results)
    #     df.to_csv(os.path.join(dir_path, file_name))

    main_worker(options)