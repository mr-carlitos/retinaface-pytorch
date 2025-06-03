from __future__ import print_function
import os
import torch
import torch.optim as optim
import argparse
import torch.utils.data as data
from data import WiderFaceDetection, detection_collate, preproc, cfg_re50
from layers.modules import MultiTaskLossWithOHEM, MultiTaskLossWithFocalLoss
from layers.functions.prior_box import PriorBox
import time
import datetime
from models.retinaface import RetinaFace
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from dotenv import load_dotenv
from pathlib import Path
import numpy as np
import warnings

load_dotenv()

def get_args():
    parser = argparse.ArgumentParser(description='Retinaface Training')
    parser.add_argument('--training_dataset', default='./data/widerface/train/label.txt',
                        help='Training dataset directory')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
    parser.add_argument('--save_folder', default='./save-checkpoints/2025-06-03_NEIGHBOURHOOD/', help='Location to save checkpoint models')

    parser.add_argument('--batch_size', default=14, type=int, help='Location to save checkpoint models')
    parser.add_argument('--epoch', default=80, type=int, help='Location to save checkpoint models')
    parser.add_argument('--decay1', default=55, type=int, help='Location to save checkpoint models')
    parser.add_argument('--decay2', default=68, type=int, help='Location to save checkpoint models')
    parser.add_argument('--image_size', default=640, type=int, help='Location to save checkpoint models')

    # Distributed training parameters
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    parser.add_argument('--world_size', type=int, default=1, help='Number of processes participating in the job')
    parser.add_argument('--dist_backend', default='nccl', help='Distributed backend')
    parser.add_argument('--gpu', type=int, default=None, help='GPU id to use, this parameter is used when you want to train only on a single GPU')

    # Checkpointing
    parser.add_argument('--resume', default='', help='Resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, help='Start epoch for resuming training')
    parser.add_argument('--save_freq', default=5, type=int, help='Save checkpoint frequency (epochs)')

    # Other parameters
    parser.add_argument('--seed', default=42, type=int, help='Random seed for reproducibility')
    parser.add_argument('--eval_freq', default=1, type=int, help='Evaluation frequency (epochs)')

    return parser.parse_args()

def setup_distributed(args):
    """Set up distributed training if needed"""
    if args.gpu is not None:
        # Use a single GPU
        torch.cuda.set_device(args.gpu)
        args.device = torch.device(f'cuda:{args.gpu}')
        args.distributed = False
    elif args.local_rank >= 0:
        # Initialize process group for distributed training
        dist.init_process_group(backend=args.dist_backend)
        args.world_size = dist.get_world_size()
        args.rank = dist.get_rank()
        args.local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
        print(args.local_rank)
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device(f'cuda:{args.local_rank}')
        print(args.device)
        args.distributed = True
        print(f"Distributed training initialized: rank {args.rank}/{args.world_size} on device {args.device}")
    else:
        # Use all available GPUs
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        args.distributed = False
        args.rank = 0

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Add cudnn benchmark
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True

    # Create directories
    if args.rank == 0 or not args.distributed:
        Path(args.save_folder).mkdir(parents=True, exist_ok=True)
    return args


def get_model(cfg, args):
    """Create and initialize the model"""
    net = RetinaFace(cfg=cfg)

    if args.rank == 0 or not args.distributed:
        print("Model architecture:")
        print(net)

    # Load checkpoint if resuming training
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=args.device)

            # Handle model state dict
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            # Remove 'module.' prefix if present
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v

            # Load state dict
            net.load_state_dict(new_state_dict)

            # Load optimizer state and epoch if available
            if 'optimizer' in checkpoint and 'epoch' in checkpoint:
                args.start_epoch = checkpoint['epoch'] + 1
                print(f"Resuming from epoch {args.start_epoch}")
            else:
                print("Loaded only model weights")
        else:
            warnings.warn(f"No checkpoint found at '{args.resume}'")

    # Move model to device and wrap with DDP if distributed
    net = net.to(args.device)
    if args.distributed:
        net = DDP(net, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    elif torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)

    return net

def get_data_loader(dataset, batch_size, args):
    """Create data loader based on distributed setting"""
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    data_loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        #pin_memory=True,
        sampler=sampler,
        collate_fn=detection_collate
    )

    return data_loader, sampler

def log_gpu_memory(device):
    allocated = torch.cuda.memory_allocated(device) / (1024**2)  # in MB
    reserved = torch.cuda.memory_reserved(device) / (1024**2)    # in MB
    summary = torch.cuda.memory_summary(device=device)
    print(f"Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")
    print(summary)


def save_checkpoint(net, optimizer, epoch, loss, args, is_best=False):
    """Save model checkpoint"""
    if args.rank != 0 and args.distributed:
        return

    # Make path is a string for torch.save
    save_path = os.path.join(args.save_folder, f"{cfg['name']}_epoch_{epoch}.pth")

    checkpoint = {
        'epoch': epoch,
        'state_dict': net.state_dict() if not isinstance(net,
                                                         (torch.nn.DataParallel, DDP)) else net.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss,
        'args': args
    }

    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")

    # Save best model if applicable
    if is_best:
        best_path = os.path.join(args.save_folder, f"{cfg['name']}_best.pth")
        torch.save(checkpoint, best_path)
        print(f"Best model saved to {best_path}")


def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size, args):
    """Sets the learning rate with warmup and decay.
    The learning rate warms up from args.lr (e.g., 1e-3) to 10x args.lr (i.e., 1e-2) over 5 epochs."""
    warmup_epoch = 5
    lr_start = args.lr        # Starting learning rate (e.g., 1e-3)
    lr_final = lr_start * 10  # Final learning rate after warmup (e.g., 1e-2)

    if epoch < warmup_epoch:
        # Linear warmup: interpolate between lr_start and lr_final over warmup_epoch epochs
        progress = (epoch * epoch_size + iteration) / (warmup_epoch * epoch_size)
        lr = lr_start + progress * (lr_final - lr_start)
    else:
        # After warmup, start with lr_final and apply step decay
        lr = lr_final * (gamma ** step_index)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

def train(cfg, args):
    """Main training function"""
    # Initialize model
    net = get_model(cfg, args)
    # Set model to training mode
    net.train()

    accumulation_steps = 2

    # Initialize optimizer
    optimizer = optim.SGD(
        net.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    # Load optimizer state if resuming
    if args.resume and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, map_location=args.device)
        if 'optimizer' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("Loaded optimizer state")
            except:
                print("Failed to load optimizer state")

    # Initialize loss function
    criterion = MultiTaskLossWithFocalLoss(
        num_classes=2,
        iou_threshold_background=cfg['iou_threshold_background'],
        variance = cfg['variance']
    )

    # Initialize prior boxes
    priorbox = PriorBox(cfg, image_size=(args.image_size, args.image_size))
    with torch.no_grad():
        priors = priorbox.vectorized_forward()
        priors = priors.to(args.device)

    # Initialize dataset
    if args.rank == 0 or not args.distributed:
        print('Loading dataset...')
    rgb_mean = (104, 117, 124)  # BGR order
    dataset = WiderFaceDetection(args.training_dataset, preproc(args.image_size, rgb_mean))

    # Calculate sizes
    batch_size = args.batch_size
    if args.distributed:
        batch_size = batch_size // args.world_size

    epoch_size = len(dataset) // batch_size
    max_epoch = args.epoch

    # Learning rate decay points
    stepvalues = (args.decay1 * epoch_size, args.decay2 * epoch_size)
    step_index = 0

    # Start from saved epoch if resuming
    start_epoch = args.start_epoch

    # Create data loader
    data_loader, sampler = get_data_loader(dataset, batch_size, args)

    # Main training loop
    best_loss = float('inf')

    for epoch in range(start_epoch, max_epoch):
        # Set epoch for distributed sampler
        if args.distributed:
            sampler.set_epoch(epoch)

        # Initialize metrics
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        loc_losses = AverageMeter('Loc', ':.4e')
        cls_losses = AverageMeter('Cls', ':.4e')

        end = time.time()
        optimizer.zero_grad()

        for iteration, (images, targets) in enumerate(data_loader):

            if iteration == 0 and (args.rank == 0 or not args.distributed):
                print(f"Image dimensions: {images.shape}")  # Should show batch_size x channels x height x width
            # Measure data loading time
            data_time.update(time.time() - end)

            # Calculate global iteration
            global_iter = epoch * epoch_size + iteration

            # Check for learning rate decay
            if global_iter in stepvalues:
                step_index += 1

            # Adjust learning rate
            lr = adjust_learning_rate(optimizer, args.gamma, epoch, step_index, iteration, epoch_size, args)

            # Move data to device
            images = images.to(args.device)
            targets = [anno.to(args.device) for anno in targets]

            # Forward pass
            out = net(images)

            # Calculate loss
            loss_l, loss_c = criterion(out, priors, targets)
            loss = (loss_l + loss_c) / accumulation_steps

            actual_loss = loss.item() * accumulation_steps
            # Update metrics
            losses.update(actual_loss, images.size(0))
            loc_losses.update(loss_l.item(), images.size(0))
            cls_losses.update(loss_c.item(), images.size(0))

            # Backward pass and optimizer
            loss.backward()

            # Update model weights after accumulating gradients for accumulation_steps iterations
            if (iteration + 1) % accumulation_steps == 0 or (iteration + 1 == len(data_loader)):
                optimizer.step()
                optimizer.zero_grad()

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # Log progress
            if (iteration % 10 == 0 or iteration == len(data_loader) - 1) and (args.rank == 0 or not args.distributed):
                eta = batch_time.avg * (len(data_loader) - iteration + (max_epoch - epoch - 1) * len(data_loader))
                print(
                    f'Epoch: [{epoch}/{max_epoch}][{iteration}/{len(data_loader)}] '
                    f'ETA: {str(datetime.timedelta(seconds=int(eta)))} '
                    f'LR: {lr:.8f} '
                    f'Time: {batch_time.val:.4f} ({batch_time.avg:.4f}) '
                    f'Data: {data_time.val:.4f} ({data_time.avg:.4f}) '
                    f'Loss: {losses.val:.4f} ({losses.avg:.4f}) '
                    f'Loc: {loc_losses.val:.4f} ({loc_losses.avg:.4f}) '
                    f'Cls: {cls_losses.val:.4f} ({cls_losses.avg:.4f})'
                )

            if iteration == len(data_loader) - 1:
                log_gpu_memory(args.device)

            del images, targets, out, loss, loss_l, loss_c

        torch.cuda.empty_cache()

        # Save checkpoint
        is_best = losses.avg < best_loss
        if is_best:
            best_loss = losses.avg

        if (epoch % args.save_freq == 0 or epoch == max_epoch - 1 or is_best) and (
                args.rank == 0 or not args.distributed):
            save_checkpoint(net, optimizer, epoch, losses.avg, args, is_best=is_best)

    # Save final model
    if args.rank == 0 or not args.distributed:
        final_path = os.path.join(args.save_folder, f"{cfg['name']}_final.pth")
        torch.save({
            'state_dict': net.state_dict() if not isinstance(net,
                                                             (torch.nn.DataParallel, DDP)) else net.module.state_dict(),
            'args': args
        }, final_path)
        print(f"Final model saved to {final_path}")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


if __name__ == '__main__':
    # Parse arguments
    args = get_args()

    # Set up distributed training
    args = setup_distributed(args)

    # Use ResNet-50 config
    cfg = cfg_re50

    # Run training
    train(cfg, args)