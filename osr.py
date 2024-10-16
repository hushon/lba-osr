import os
import argparse
import datetime
import time
import pandas as pd
import importlib

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

from models import gan
from models.models import classifier32, classifier32ABN
from datasets.osr_dataloader import MNIST_OSR, CIFAR10_OSR, CIFAR100_OSR, SVHN_OSR, Tiny_ImageNet_OSR
from utils import save_networks, load_networks
from core import train, train_cs, test

# Allow TF32 precision for performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def parse_arguments():
    parser = argparse.ArgumentParser("Training")

    # Dataset
    parser.add_argument('--dataset', type=str, default='mnist', help="mnist | svhn | cifar10 | cifar100 | tiny_imagenet")
    parser.add_argument('--dataroot', type=str, default='/home/hyounguk.shon/data')
    parser.add_argument('--outf', type=str, default='./log')
    parser.add_argument('--out-num', type=int, default=50, help='For CIFAR100')

    # Optimization
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.1, help="learning rate for model")
    parser.add_argument('--gan_lr', type=float, default=0.0002, help="learning rate for gan")
    parser.add_argument('--max-epoch', type=int, default=100)
    parser.add_argument('--stepsize', type=int, default=30)
    parser.add_argument('--num-centers', type=int, default=1)

    # Model
    parser.add_argument('--model', type=str, default='classifier32')

    # Misc
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--use-cpu', action='store_true')
    parser.add_argument('--eval', action='store_true', help="Eval", default=False)
    parser.add_argument('--cs', action='store_true', help="Confusing Sample", default=False)
    parser.add_argument('--use-open-classnames', action='store_true', help="Use open classnames for evaluation", default=False)

    return parser.parse_args()


def prepare_environment(options):
    torch.manual_seed(options['seed'])
    os.environ['CUDA_VISIBLE_DEVICES'] = options['gpu']
    use_gpu = torch.cuda.is_available() and not options['use_cpu']

    if use_gpu:
        print(f"Currently using GPU: {options['gpu']}")
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(options['seed'])
    else:
        print("Currently using CPU")

    return use_gpu


def get_dataloader(options):
    dataset_mapping = {
        'mnist': MNIST_OSR,
        'cifar10': CIFAR10_OSR,
        'svhn': SVHN_OSR,
        'cifar100': CIFAR100_OSR,
        'tiny_imagenet': Tiny_ImageNet_OSR
    }
    Data = dataset_mapping[options['dataset']](
        known=options['known'],
        dataroot=options['dataroot'],
        batch_size=options['batch_size'],
        img_size=options['img_size']
    )
    return Data.train_loader, Data.test_loader, Data.out_loader if options['dataset'] != 'cifar100' else None


def create_model(options):
    print(f"Creating model: {options['model']}")
    net = classifier32ABN(num_classes=options['num_classes']) if options['cs'] else classifier32(num_classes=options['num_classes'])
    return net


def setup_loss(options):
    Loss = importlib.import_module('loss.' + options['loss'])
    return getattr(Loss, options['loss'])(**options)


def setup_optimizer(options, net, criterion):
    params_list = [{'params': net.parameters()}, {'params': criterion.parameters()}]
    if options['dataset'] == 'tiny_imagenet':
        return torch.optim.Adam(params_list, lr=options['lr'])
    else:
        return torch.optim.SGD(params_list, lr=options['lr'], momentum=0.9, weight_decay=1e-4)


def main_worker(options):
    use_gpu = prepare_environment(options)
    trainloader, testloader, outloader = get_dataloader(options)
    options['num_classes'] = trainloader.dataset.num_classes

    net = create_model(options)
    criterion = setup_loss(options)

    if use_gpu:
        net = nn.DataParallel(net).cuda()
        criterion = criterion.cuda()

    optimizer = setup_optimizer(options, net, criterion)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90, 120]) if options['stepsize'] > 0 else None

    start_time = time.time()
    for epoch in range(options['max_epoch']):
        print(f"==> Epoch {epoch + 1}/{options['max_epoch']}")
        train(net, criterion, optimizer, trainloader, epoch=epoch, **options)

        if (epoch + 1) % options['eval_freq'] == 0 or (epoch + 1) == options['max_epoch']:
            print(f"==> Test {options['loss']}")
            if options['use_open_classnames']:
                print("Using open classnames for evaluation")
            results = test(net, criterion, testloader, outloader, epoch=epoch, **options)
            print("Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(
                results['ACC'], results['AUROC'], results['OSCR']))
            save_networks(net, os.path.join(options['outf'], 'models', options['dataset']),
                          f"{options['model']}_{options['loss']}_{options['item']}", criterion=criterion)
        if scheduler:
            scheduler.step()

    elapsed = str(datetime.timedelta(seconds=round(time.time() - start_time)))
    print(f"Finished. Total elapsed time (h:m:s): {elapsed}")
    return results


if __name__ == '__main__':
    args = parse_arguments()
    options = vars(args)
    options['dataroot'] = os.path.join(options['dataroot'], options['dataset'])
    img_size = 32 if options['dataset'] != 'tiny_imagenet' else 64
    options['img_size'] = img_size

    from split import splits_2020 as splits
    results = {}

    for i, known in enumerate(reversed(splits[options['dataset']])):
        unknown = list(set(range(200 if options['dataset'] == 'tiny_imagenet' else 10)) - set(known))
        options.update({'item': i, 'known': known, 'unknown': unknown})
        res = main_worker(options)
        results[str(i)] = {**res, 'unknown': unknown, 'known': known}
        pd.DataFrame(results).to_csv(os.path.join(options['outf'], 'results', f"{options['model']}_{options['loss']}.csv"))