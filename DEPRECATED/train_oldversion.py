from __future__ import print_function
import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import torch.utils.data as data
from data import WiderFaceDetection, detection_collate, preproc, cfg_re50
from layers.modules import MultiTaskLossWithOHEM, MultiTaskLossWithFocalLoss
from layers.functions.prior_box import PriorBox
import time
import datetime
import math
from models.retinaface import RetinaFace

parser = argparse.ArgumentParser(description='Retinaface Training')
parser.add_argument('--training_dataset', default='./data/widerface/train/label.txt', help='Training dataset directory')
parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--resume_net', default=None, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--save_folder', default='./weights/', help='Location to save checkpoint models')

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

cfg = cfg_re50

rgb_mean = (104, 117, 124) # bgr order
num_classes = 2
img_dim = cfg['image_size']
batch_size = cfg['batch_size']
max_epoch = cfg['epoch']
gpu_train = cfg['gpu_train']

num_workers = args.num_workers
momentum = args.momentum
weight_decay = args.weight_decay
initial_lr = args.lr
gamma = args.gamma
training_dataset = args.training_dataset
save_folder = args.save_folder

device_ids = [2, 3]
torch.cuda.set_device(device_ids[0])
net = RetinaFace(cfg=cfg)
print("Printing net...")
print(net)

if args.resume_net is not None:
    print('Loading resume network...')
    state_dict = torch.load(args.resume_net)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)


num_gpu = len(device_ids)
if num_gpu > 1 and gpu_train:
    # Set the primary CUDA device to the first in your list
    net = torch.nn.DataParallel(net, device_ids=device_ids).cuda()
else:
    net = net.cuda()

#cudnn.benchmark = True

iou_threshold_background = cfg['iou_threshold_background']
iou_threshold_foreground = cfg['iou_threshold_foreground']
variances = cfg['variance']
focal_gamma = cfg['focal_gamma']
focal_alpha = cfg['focal_alpha']

optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
criterion = MultiTaskLossWithFocalLoss(num_classes, iou_threshold_background, variances, focal_gamma, focal_alpha)

priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
with torch.no_grad():
    priors = priorbox.vectorized_forward()
    priors = priors.cuda()

def train():
    net.train()
    epoch = 0 + args.resume_epoch
    print('Loading Dataset...')

    #When looking here at the preproc object, we just do the __init__(self, img_dim, rgb_means) part here. Not yet the __call__(self, image, targets).
    # __call__(self, image, targets) of preproc gets triggered, once the __getitem__(self, index) function of WiderFaceDetection gets executed
    dataset = WiderFaceDetection(training_dataset, preproc(img_dim, rgb_mean))

    epoch_size = math.ceil(len(dataset) / batch_size)
    max_iter = max_epoch * epoch_size

    stepvalues = (cfg['decay1'] * epoch_size, cfg['decay2'] * epoch_size)
    step_index = 0

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0

    for iteration in range(start_iter, max_iter):
        #TODO: later, we access batch_iterator. What if this if is == False?
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(data.DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers, collate_fn=detection_collate))
            if (epoch % 10 == 0 and epoch > 0) or (epoch % 5 == 0 and epoch > cfg['decay1']):
                torch.save(net.state_dict(), save_folder + cfg['name']+ '_epoch_' + str(epoch) + '.pth')
            epoch += 1

        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size, args)

        # load train data
        images, targets = next(batch_iterator)
        images = images.cuda()
        targets = [anno.cuda() for anno in targets]

        # forward
        out = net(images)

        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, priors, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        load_t1 = time.time()
        batch_time = load_t1 - load_t0
        eta = int(batch_time * (max_iter - iteration))
        print('Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || Loc: {:.4f} Cla: {:.4f} || LR: {:.8f} || Batchtime: {:.4f} s || ETA: {}'
              .format(epoch, max_epoch, (iteration % epoch_size) + 1,
              epoch_size, iteration + 1, max_iter, loss_l.item(), loss_c.item(), lr, batch_time, str(datetime.timedelta(seconds=eta))))

    torch.save(net.state_dict(), save_folder + cfg['name'] + '_Final_Carlos.pth')
    # torch.save(net.state_dict(), save_folder + 'Final_Retinaface.pth')


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

if __name__ == '__main__':
    train()
