import os
import sys

from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import random
import numpy as np

import torch
import torch.optim as optim

from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from dataloaders.utils import test_all_case_2d_LA, get_dice_avg_2d
from utils.util import probs2one_hot
from dataloaders.la_heart import LAHeart_OneHot
from utils.loss_ema import LabeledPenalty_NoRec, UnlabeledPenalty_NoRec
from utils.losses import DiceLoss
from networks.unet2d import UNet_MTL_2D
from networks.init import weights_init

from dataloaders.la_heart import ToTensor, TwoStreamBatchSampler, RandomGenerator, RandomCrop

device_student = "cuda:0"
device_teacher = "cuda:0"

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/2018LA_Seg_Training Set/', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='LA_ema_2D', help='model_name')
parser.add_argument('--max_iterations', type=int, default=60000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=32, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=16, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=0.1, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--beta', type=float, default=0.3, help='balance factor to control regional and sdm loss')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='balance factor to control supervised and consistency loss')
parser.add_argument('--patch_size', type=list, default=[112, 112],
                    help='patch size of network input')
### costs
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str, default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float, default=0.01, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = ""

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs

if not args.deterministic:
    cudnn.benchmark = True  #
    cudnn.deterministic = False  #
else:
    cudnn.benchmark = False  # True #
    cudnn.deterministic = True  # False #
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

num_classes = 2
dice_metric = DiceLoss(num_classes=num_classes).cuda()


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def create_model(ema=False):
    # Network definition
    net = UNet_MTL_2D(in_channel=1, num_classes=num_classes, base_filter=32)
    model = net.cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model


def log_n(data, base):
    return torch.log(data) / torch.log(torch.FloatTensor([base])).cuda()


if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    loss_labelled = LabeledPenalty_NoRec()
    loss_unlabelled = UnlabeledPenalty_NoRec()
    student_net = create_model().to(device_student)
    teacher_net = create_model(True).to(device_teacher)
    student_net.apply(weights_init)
    teacher_net.apply(weights_init)
    student_net = student_net.cuda()
    teacher_net = teacher_net.cuda()
    label_num = -1 # set yours
    num_unlabelled_train = -1 # set yours
    load_path_labelled = ""
    db_train = LAHeart_OneHot(train_path="",
                              num_classes=num_classes,
                              load_path=os.path.join(load_path_labelled, "train"),
                              transform=transforms.Compose([
                                  RandomGenerator(),
                                  RandomCrop(args.patch_size),
                                  ToTensor(),
                              ]),
                              is_3d=False)

    labeled_idxs = list(range(label_num))
    unlabeled_idxs = list(range(label_num, num_unlabelled_train))
    batch_sampler_train = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - labeled_bs)


    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)


    train_loader = DataLoader(db_train, batch_sampler=batch_sampler_train, num_workers=4,
                              pin_memory=True, worker_init_fn=worker_init_fn)

    optimizer = optim.SGD(student_net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info(f"The labelled_num is {label_num}, the unlabelled_num is {num_unlabelled_train}")
    logging.info("{} itertations per epoch".format(len(train_loader)))

    iter_num = 0
    max_epoch = max_iterations // len(train_loader) + 1
    lr_ = base_lr
    avg_dice = get_dice_avg_2d(num_classes)
    iterator = tqdm(range(max_epoch), ncols=70)
    best_dice_val = 0
    for epoch_num in iterator:
        student_net.train()
        for i_batch, sampled_batch in enumerate(train_loader):
            optimizer.zero_grad()
            volume_batch, label_batch = sampled_batch["image"], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            volume_batch_r = volume_batch.repeat(1, 1, 1, 1)
            actual_batch_size = volume_batch.size()[0]
            volume_batch_in = volume_batch + torch.clamp(torch.randn_like(volume_batch) * 0.02, -0.05, 0.05)

            student_results = student_net(volume_batch_in)

            K = 8
            ema_preds, ema_sdm_probs, = torch.zeros([2, K, actual_batch_size, num_classes, *args.patch_size]).cuda()
            for k in range(K):
                ema_inputs = volume_batch_r + torch.clamp(torch.randn_like(volume_batch_r) * 0.02, -0.05, 0.05)
                with torch.no_grad():
                    ema_results = teacher_net(ema_inputs)
                    ema_preds[k] = ema_results["segmentation"]
                    ema_sdm_probs[k] = ema_results["sdm"]

            # dimension [K, B, C, W, H]
            ema_preds = F.softmax(ema_preds, dim=2)
            uncertainty = -1.0 * torch.sum(ema_preds * log_n(ema_preds + 1e-6, base=num_classes), dim=2,
                                           keepdim=True)
            ema_probs = torch.mean(ema_preds, dim=0)
            ema_mask = probs2one_hot(ema_probs.detach())
            ema_seg_uncertainty = -1.0 * torch.sum(ema_probs * log_n(ema_probs + 1e-6, base=num_classes), dim=1,
                                                   keepdim=True)

            ema_sdm = ema_sdm_probs.mean(dim=0)
            ema_sdm_uncertainty = ema_sdm_probs.var(dim=0)

            loss_val_l = loss_labelled(seg_probs=torch.softmax(student_results["segmentation"][:labeled_bs, ...],
                                                               dim=1),
                                       seg_targets=label_batch[:labeled_bs, ...],
                                       ema_mask=ema_mask[:labeled_bs, ...],
                                       ema_sdm=ema_sdm[:labeled_bs, ...],
                                       ema_sdm_uncertainty=ema_sdm_uncertainty[:labeled_bs, ...],
                                       ema_seg_uncertainty=ema_seg_uncertainty[:labeled_bs, ...],
                                       epoch=epoch_num,
                                       sdm_probs=student_results["sdm"][:labeled_bs, ...],
                                       device=device_student)
            loss_val_u = loss_unlabelled(seg_probs=torch.softmax(student_results["segmentation"][labeled_bs:, ...],
                                                                 dim=1),
                                         ema_mask=ema_mask[labeled_bs:, ...],
                                         ema_sdm=ema_sdm[labeled_bs:, ...],
                                         ema_sdm_uncertainty=ema_sdm_uncertainty[labeled_bs:, ...],
                                         ema_seg_uncertainty=ema_seg_uncertainty[labeled_bs:, ...],
                                         epoch=epoch_num,
                                         sdm_probs=student_results["sdm"][labeled_bs:, ...],
                                         device=device_student)

            loss_val = loss_val_l + loss_val_u
            loss_val.backward()
            optimizer.step()
            update_ema_variables(student_net, teacher_net, alpha=0.99, global_step=iter_num)

            iter_num = iter_num + 1
        loss_dice, loss_sdm, seg_con_l, sdm_con_l, loss_l = loss_labelled.get_and_reset_l()
        loss_sdm_seg, seg_con_u, sdm_con_u, loss_u = loss_unlabelled.get_and_reset_u()
        writer.add_scalar('lr', lr_, epoch_num)
        writer.add_scalar('loss/loss', (loss_l + loss_u), epoch_num)
        writer.add_scalar('loss/loss_seg', loss_dice, epoch_num)
        writer.add_scalar('loss/loss_sdm', loss_sdm, epoch_num)
        writer.add_scalar('loss/seg_consistency_l', seg_con_l, epoch_num)
        writer.add_scalar('loss/sdm_consistency_l', sdm_con_l, epoch_num)
        writer.add_scalar('loss/seg_consistency_u', seg_con_u, epoch_num)
        writer.add_scalar('loss/sdm_consistency_u', sdm_con_u, epoch_num)
        writer.add_scalar('loss/sdm_with_seg_guidance', loss_sdm_seg, epoch_num)

        logging.info(
            'iteration %d : loss : %f, loss_dice: %f, loss_sdm: %f, seg_consistency_l: %f '
            'sdm_consistency_l: %f seg_consistency_u: %f sdm_consistency_u: %f sdm_with_seg_guidance: %f ' %
            (epoch_num, (loss_l + loss_u), loss_dice, loss_sdm, seg_con_l, sdm_con_l, seg_con_u,
             sdm_con_u, loss_sdm_seg))

        # eval
        dice_val_overall = test_all_case_2d_LA(
            num_classes=num_classes,
            base_dir=os.path.join(load_path_labelled, "val"),
            test_list="",
            metric_fn=avg_dice,
            net=student_net.eval(),
            stride_xy=18,
            patch_size=args.patch_size,
            permute_dim=None,
        )

        logging.info('iteration %d : Dice on val data : %f, l' % (epoch_num, dice_val_overall))

        if epoch_num % 200 == 0 or dice_val_overall > best_dice_val:
            save_mode_path = os.path.join(snapshot_path, 'iter_' + str(epoch_num) + '_student.pth')
            torch.save(student_net.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            save_mode_path = os.path.join(snapshot_path, 'iter_' + str(epoch_num) + '_teacher.pth')
            torch.save(teacher_net.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            if dice_val_overall > best_dice_val:
                best_dice_val = dice_val_overall
    iterator.close()
    save_mode_path = os.path.join(snapshot_path, 'final_student.pth')
    torch.save(student_net.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    save_mode_path = os.path.join(snapshot_path, 'final_teacher.pth')
    torch.save(teacher_net.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()
