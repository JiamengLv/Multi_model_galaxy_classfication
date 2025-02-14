import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
import torch.utils.data.distributed
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.utils import ModelEmaV2
from utils import Save_Checkpoint, param_groups_weight_decay, WarmupLR

from pathlib import Path
from collections import OrderedDict
import os
import time
import argparse
from torch.utils.tensorboard import SummaryWriter
from Models.model import *

import torchvision
import torchvision.transforms as T
import torch
from torch.utils.data import Dataset
from astropy.io import fits
import os
import torchvision.transforms as T
from torchvision.transforms import ToTensor
import glob
import warnings

warnings.filterwarnings('ignore')


# def galaxy(args, scale_size=256, target_size=224):
#     train_data = torchvision.datasets.ImageFolder(root="/data/docker_containers/ljm/GalaxyClassification/GalaxyClassification/Data/point-spirals/jpg/train/")
#     test_data = torchvision.datasets.ImageFolder(root="/data/docker_containers/ljm/GalaxyClassification/GalaxyClassification/Data/point-spirals/jpg/test/")
#     num_class = 2

#     # base augmentation
#     train_data.transform = T.Compose([
#         T.Resize(scale_size, interpolation=3),
#         T.RandomResizedCrop(target_size),
#         T.RandomHorizontalFlip(0.5),
#         T.ToTensor(),
#         T.Normalize(mean=[0.17425595636692184, 0.17917468698416852, 0.1824044169230193], std=[0.10131195658084038, 0.10921934051679688, 0.11584932454748567])
#     ])

#     test_data.transform = T.Compose([
#         T.Resize(scale_size, interpolation=3),
#         T.CenterCrop(target_size),
#         T.ToTensor(),
#         T.Normalize(mean=[0.17425595636692184, 0.17917468698416852, 0.1824044169230193], std=[0.10131195658084038, 0.10921934051679688, 0.11584932454748567])
#     ])

#     return train_data, test_data, num_class




class FitsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_list = glob.glob(root_dir + '/*/*.fits')

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root_dir, self.file_list[idx])

        with fits.open(img_path) as hdul:

            img = hdul[0].data.byteswap().newbyteorder()
            label = int(self.file_list[idx].split('/')[-2]) 
            img = torch.tensor(img).float()

        if self.transform:
            img = self.transform(img)


        return img, label, img_path

def galaxy(args, scale_size=256, target_size=224):
    train_transform = T.Compose([
        # T.Resize(92, interpolation=T.InterpolationMode.BICUBIC),
        T.Resize(scale_size, interpolation=T.InterpolationMode.BICUBIC),
        T.RandomResizedCrop(target_size),
        T.RandomHorizontalFlip(0.5),
        # T.Normalize(mean=[0.00332886, 0.00607329, 0.01019075], std= [0.05832583, 0.0869752,  0.13537677])
    ])

    test_transform = T.Compose([
        # T.Resize(92, interpolation=T.InterpolationMode.BICUBIC),
        T.Resize(scale_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(target_size),
        # T.Normalize(mean=[0.00332886, 0.00607329, 0.01019075], std= [0.05832583, 0.0869752,  0.13537677])
    ])

    train_data = FitsDataset(root_dir="/home/dell461/lys/ring_galaxy/dataset/v1/train/", transform=train_transform)
    test_data = FitsDataset(root_dir="/home/dell461/lys/ring_galaxy/dataset/v1/test/", transform=test_transform)

    print("train_data:", len(train_data))
    print("test_data:", len(test_data))

    num_class = 2

    return train_data, test_data, num_class



def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
        print("Use GPU: {} for training".format(args.gpu))
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)

    dist.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )
    dist.barrier()
    setup_for_distributed(args.rank == 0)


def create_dataset(args):
    train_dataset, val_dataset, num_class = galaxy(args)
    
    args.batch_size = int(args.batch_size / args.world_size)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=False)
    else:
        raise ValueError("Distributed init error.")
    
    train_loader = DataLoader(train_dataset, 
                              batch_size=args.batch_size,
                              num_workers=args.workers,
                              pin_memory=True,
                              sampler=train_sampler,
                              drop_last=True)
    
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            num_workers=args.workers,
                            pin_memory=True,
                            sampler=val_sampler,
                            drop_last=False)
    
    return train_loader, val_loader, num_class, train_sampler


def create_model(args, num_class):

    model = args.model_name
    
    if model == 'vit_base_patch16_224':
        model = vit_base_patch16_224(output_size=num_class)
    elif model == 'vit_small_patch16_224':
        model = vit_small_patch16_224(output_size=num_class)
    elif model == 'vit_tiny_patch16_224':
        model = vit_tiny_patch16_224(output_size=num_class)

    elif model == 'swin_small_patch4_window7_224':
        model = swin_small_patch4_window7_224(output_size=num_class)
    elif model == 'swin_base_patch4_window7_224':
        model = swin_base_patch4_window7_224(output_size=num_class)
    elif model == 'swin_tiny_patch4_window7_224':
        model = swin_tiny_patch4_window7_224(output_size=num_class)

    elif model == 'resnet18':
        model = resnet18(output_size=num_class)
    elif model == 'resnet50':
        model = resnet50(output_size=num_class)
    elif model == 'resnet101':
        model = resnet101(output_size=num_class)

    elif model == 'efficientnet_b0':
        model = Efficientnet_b0(output_size=num_class)

    else:
        raise ValueError(f"Unsupported model: {model}. Supported models are 'vit_base_patch16_224', 'vit_small_patch16_224', 'vit_tiny_patch16_224', 'resnet18', 'resnet50', and 'resnet101'.")


    return model


def main(args):

    init_distributed_mode(args)

    cudnn.benchmark = True

    device = torch.device(args.device)

    # data loaders
    train_loader, val_loader, num_class, train_sampler = create_dataset(args=args)

    # create model
    model = create_model(args=args, num_class=num_class)
    model.to(device)

    model = model


    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before DDP wrapper
        model_ema = ModelEmaV2(model, decay=args.model_ema_decay)
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params (M): %.4f' % (n_parameters / 1.e6))      

    # 模型权重衰减
    param_groups = param_groups_weight_decay(model=model_without_ddp, weight_decay=args.weight_decay, weight_decay_head=args.wd_head)

    # # 优化函数
    # optimizer = torch.optim.SGD(
    #     params=param_groups,
    #     lr=args.lr,
    #     momentum=args.momentum,
    #     nesterov=True,
    # )

    ##################################################################################
    # 由于在优化器构造函数中设置了 weight_decay=0，优化器不会自动应用权重衰减，而是通过 param_groups 中指定的不同权重衰减策略来控制

    if args.model_name.startswith('vit') or args.model_name.startswith('swin'):
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        optimizer = torch.optim.AdamW(params=param_groups, lr=args.lr, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args.epochs-args.warmup_epochs)            # TODO: 
        scheduler_warmupLr  = WarmupLR(optimizer,args.warmup_epochs)
    
    else:
        optimizer = torch.optim.SGD(params=param_groups,lr=args.lr, momentum=args.momentum, nesterov=True,weight_decay=0)
        scheduler = None
        scheduler_warmupLr = None

    # loss
    criterion = nn.CrossEntropyLoss().to(device)

    # file path
    if dist.get_rank() == 0:
        # weights
        save_dir = Path(args.save_dir)
        weights = save_dir / 'weights'
        weights.mkdir(parents=True, exist_ok=True)
        last = weights / 'last'
        best = weights / 'best'

        # tensorboard
        logdir = save_dir / 'logs'
        logdir.mkdir(parents=True, exist_ok=True)
        summary_writer = SummaryWriter(logdir, flush_secs=120)

        # result
        model_file = str(save_dir / 'model.txt')
        with open(model_file, "a") as f:
            print(model_without_ddp, file=f)
            print(args, file=f)
    
    if args.resume:
        if args.gpu is None:
            checkpoint = torch.load(args.resume)
        elif torch.cuda.is_available():
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(args.resume, map_location=loc)
           
        args.start_epoch = checkpoint['epoch']
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_acc = torch.tensor(checkpoint['best_acc'])
        if args.gpu is not None:
            # best_acc may be from a checkpoint from a different GPU
            best_acc = best_acc.to(args.gpu)

        if model_ema:
            model_ema.load_state_dict(checkpoint["model_ema"])

        print('Resuming training from {} epoch'.format(args.start_epoch))
    else:
        best_acc = 0
    
    print("Start training")

    for epoch in range(args.start_epoch, args.epochs):

        print("Epoch {}/{}".format(epoch + 1, args.epochs))
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        train_epoch_loss, train_acc1 = train(model=model,
                                             train_loader=train_loader,
                                             optimizer=optimizer,
                                             scheduler = scheduler,
                                             scheduler_warmupLr = scheduler_warmupLr,
                                             criterion=criterion,
                                             args=args,
                                             epoch=epoch,
                                             model_ema=model_ema)
        
        val_epoch_loss, val_acc = validate(model=model,
                                           val_loader=val_loader,
                                           criterion=criterion,
                                           args=args)
        
        s = " \n Train Loss: {:.3f}, Train Acc: {:.3f}, Test Loss: {:.3f}, Test Acc: {:.3f}, lr: {:.1e}".format(
            train_epoch_loss, train_acc1, val_epoch_loss, val_acc, optimizer.param_groups[-1]['lr'])
        print(s)

        if dist.get_rank() == 0:
            # save model
            is_best = val_acc > best_acc
            best_acc = max(best_acc, val_acc)
            state = {
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_acc': best_acc,
            }
            if model_ema:
                state["model_ema"] = model_ema.state_dict()
            
            last_path = last / 'epoch_{}_loss_{:.4f}_acc_{:.3f}'.format(
                epoch + 1, val_epoch_loss, val_acc)
            best_path = best / 'epoch_{}_acc_{:.4f}'.format(
                epoch + 1, best_acc)
            Save_Checkpoint(state, last, last_path, best, best_path, is_best)
            
            summary_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
            summary_writer.add_scalar('train_loss', train_epoch_loss, epoch)
            summary_writer.add_scalar('train_acc', train_acc1, epoch)
            summary_writer.add_scalar('val_loss', val_epoch_loss, epoch)
            summary_writer.add_scalar('val_acc', val_acc, epoch)
    
    if dist.get_rank() == 0:
        summary_writer.close()


def train(model, train_loader, optimizer,scheduler, scheduler_warmupLr, criterion, args, epoch, model_ema):
    train_loss = AverageMeter()
    train_acc = AverageMeter()
    
    all_preds = []
    all_labels = []


    # Model on train mode
    model.train()
    step_per_epoch = len(train_loader)
    for step, (images, labels,_) in enumerate(train_loader):
        torch.cuda.synchronize()
        start = time.time()

        if args.gpu is not None and torch.cuda.is_available():
            # images = images.cuda(args.gpu, non_blocking=True).double()
            images = images.cuda(args.gpu, non_blocking=True)

            labels = labels.cuda(args.gpu, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        # measure accuracy and record loss
        acc1 = accuracy(logits, labels, topk=(1, ))

        train_loss.update(loss.item(), images.size(0))
        train_acc.update(acc1[0].item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)
        
        t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        s1 = '\r{} [{}/{}]'.format(t, step+1, step_per_epoch)
        s2 = ' - {:.2f}ms/step - train_loss: {:.3f} - train_acc: {:.3f}'.format(
             1000 * (time.time()-start), train_loss.val, train_acc.val)
        print(s1+s2, end='', flush=True)

        preds = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 7))
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm, annot=True, cmap='Blues')
    plt.ylabel('True label') 
    plt.xlabel('Predicted label')
    plt.savefig(args.save_dir + 'train_confusion_matrix.png')


    #############################################

    if args.model_name.startswith('vit') or args.model_name.startswith('swin'):
        if epoch < args.warmup_epochs:
            scheduler_warmupLr.step()
        else:
            scheduler.step()

    else:
        if epoch in args.lr_steps:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1

    #############################################

    return train_loss.avg, train_acc.avg

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def validate(model, val_loader, criterion, args):
    val_loss = AverageMeter()
    val_acc = AverageMeter()

    all_preds = []
    all_labels = []

    # model to evaluate mode
    model.eval()

    with torch.no_grad():
        for step, (images, labels, _) in enumerate(val_loader):
            if args.gpu is not None and torch.cuda.is_available():
                #  images = images.cuda(args.gpu, non_blocking=True).double()
                 images = images.cuda(args.gpu, non_blocking=True)
                 labels = labels.cuda(args.gpu, non_blocking=True)

            # compute output

            logits = model(images)
            loss = criterion(logits, labels)

            # measure accuracy and record loss
            acc1 = accuracy(logits, labels, topk=(1, ))

            # TODO: Average loss and accuracy across processes
            if args.distributed:
                loss = reduce_tensor(loss, args)
                acc1 = reduce_tensor(acc1[0], args)
            
            val_loss.update(loss.item(), images.size(0))
            val_acc.update(acc1[0].item(), images.size(0))

            _, preds = logits.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 7))
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm, annot=True, cmap='Blues')
    plt.ylabel('True label') 
    plt.xlabel('Predicted label')
    plt.savefig(args.save_dir + 'confusion_matrix.png')

    
    return val_loss.avg, val_acc.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def reduce_tensor(tensor, args):
    rt = tensor.clone()
    rt = tensor
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            # res.append(correct_k.mul_(100.0 / batch_size))
            res.append(correct_k.mul_(1.0 / batch_size))
        return res



def test_model(model, test_data):
    val_acc = AverageMeter()

    all_preds = []
    all_labels = []
    
    # model to evaluate mode
    model.eval()
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.workers, pin_memory=True)
    with torch.no_grad():
        for step, (images, labels,_) in enumerate(test_dataloader):
            # images, labels = images.cuda().double(), labels.cuda()
            images, labels = images.cuda(), labels.cuda()
            pred = model(images)
            acc1 = accuracy(pred, labels, topk=(1, ))
            val_acc.update(acc1[0].item(), images.size(0))

            _, preds = pred.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png')
    return val_acc.avg

if __name__ == '__main__':  

    parser = argparse.ArgumentParser(description='PyTorch Training for visual tuning.')
    # model parameters
    parser.add_argument("--model_name", type=str, default="vit_tiny_patch16_224", help="architecture", 
                        choices=["resnet18", "resnet50", "vit_tiny_patch16_224", "vit_small_patch16_224","efficientnet_b0", "vit_base_patch16_224" ,"resnet101" ,"swin_small_patch4_window7_224" ,"swin_base_patch4_window7_224" ,"swin_tiny_patch4_window7_224"])
    parser.add_argument("--model_weights", type=str, default="", help="pretrained model weights")
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--model_ema', action='store_true')
    parser.add_argument('--model-ema-decay', type=float, default=0.9999)

    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--start_epoch", default=0, type=int, help="start epoch")
    parser.add_argument("--warmup_epochs",default=10,type=int, help="warmup Vit")
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument('--workers', default=1, type=int, help='number of data loading workers')
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--wd_head", type=float, default=0.5)
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate')
    parser.add_argument("--lr_steps", default=[70,90,95], type=list, help="decrease lr")

    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("--sync_bn", action="store_true", help="Use sync batch norm")
    parser.set_defaults(sync_bn=True)

    # distributed training parameters
    parser.add_argument('--dist_url', default='env://', type=str, help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')

    parser.add_argument("--resume", type=str, help="ckpt's path to resume most recent training")
    parser.add_argument("--save_dir", type=str, default="./run", help="save path, eg, acc_loss, weights, tensorboard, and so on")
    args = parser.parse_args()

    # if args.model_weights:
    #     train_dataset, val_dataset, num_class = Dataset.__dict__[args.dataset](args)

    #     model = resnet18(output_size=num_class)
    #     ckpt = torch.load(args.model_weights, map_location='cpu')

    #     new_dict = OrderedDict()
    #     for k, v in ckpt['model'].items():
    #         if 'module.' in k:
    #             new_dict[k[7:]] = v
        
    #     model.load_state_dict(new_dict)
    #     pdb.set_trace()
    #     model = model.cuda()
    #     acc1 = test_model(model, val_dataset)
    #     print('Acc Top1: {}'.format(acc1))

    main(args=args)