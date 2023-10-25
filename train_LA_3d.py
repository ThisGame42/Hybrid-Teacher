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

from networks.unet2d import UNet_MTL_2D
from networks.unet3d import UNet_MTL_3D
from networks.utils import results_2d_from_3d
from utils.util import probs2one_hot_general
from utils.loss_ema import LabeledPenalty_NoRec, UnlabeledPenalty_NoRec
from utils.losses import DiceLoss
from networks.init import weights_init

from dataloaders.la_heart import ToTensor3D, TwoStreamBatchSampler, RandomRotFlip, RandomCrop3D, LAHeart_OneHot
from dataloaders.utils import test_all_case_3d, get_dice_avg_3d

device_2d = "cuda:0"
device_3d = "cuda:0"

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/2018LA_Seg_Training Set/', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='LA_ema_3D', help='model_name')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=2, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=1, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=0.1, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--beta', type=float, default=0.3, help='balance factor to control regional and sdm loss')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='balance factor to control supervised and consistency loss')
parser.add_argument('--patch_size', type=list, default=[112, 112, 80],
                    help='patch size of network input')
### costs
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str, default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float, default=0.01, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = ""

batch_size = args.batch_size
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
dice_metric = DiceLoss(num_classes=num_classes).to(device_2d)

student_path_2d = ""
teacher_path_2d = ""


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def create_model2D(ema=False):
    # Network definition
    net = UNet_MTL_2D(in_channel=1, num_classes=num_classes, base_filter=32)
    model = net.cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model


def create_model3D(ema=False):
    # Network definition
    net = UNet_MTL_3D(in_channel=1 + num_classes, num_classes=num_classes, base_filter=32, has_dropout=ema)
    model = net.cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model


def load_models_2D(student_path, teacher_path):
    student_model = create_model2D().to(device_2d)
    teacher_model = create_model2D(True).to(device_2d)

    student_model.load_state_dict(torch.load(student_path))
    teacher_model.load_state_dict(torch.load(teacher_path))
    for p in student_model.parameters():
        p.requires_grad = False
    for p in teacher_model.parameters():
        p.requires_grad = False
    student_model = student_model.eval()
    teacher_model = teacher_model.eval()
    return student_model, teacher_model


def log_n(data, base, device):
    return torch.log(data) / torch.log(torch.FloatTensor([base])).to(device)


if __name__ == "__main__":
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    num_slices = -1  # set yours
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    loss_labelled = LabeledPenalty_NoRec()
    loss_unlabelled = UnlabeledPenalty_NoRec()

    _, teacher_net_2d = load_models_2D(student_path=student_path_2d,
                                       teacher_path=teacher_path_2d)
    student_net_3D = create_model3D().to(device_3d)
    teacher_net_3D = create_model3D(True).to(device_3d)
    student_net_3D.apply(weights_init)
    teacher_net_3D.apply(weights_init)
    student_net_3D = student_net_3D.to(device_3d)
    teacher_net_3D = teacher_net_3D.to(device_3d)
    load_path_labelled = ""
    label_num = -1  # set yours
    num_unlabelled_train = -1  # set yours
    logging.info(f"The labelled_num is {label_num}, the unlabelled_num is {num_unlabelled_train}")
    db_train = LAHeart_OneHot(train_path="",
                              num_classes=num_classes,
                              load_path=os.path.join(load_path_labelled, "train"),
                              transform=transforms.Compose([
                                  RandomRotFlip(),
                                  RandomCrop3D(args.patch_size),
                                  ToTensor3D(),
                              ]),
                              is_3d=True)

    labeled_idxs = list(range(label_num))
    unlabeled_idxs = list(range(label_num, num_unlabelled_train))
    batch_sampler_train = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - labeled_bs)


    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)


    train_loader = DataLoader(db_train, batch_sampler=batch_sampler_train, num_workers=4,
                              pin_memory=True, worker_init_fn=worker_init_fn)

    optimizer = optim.SGD(student_net_3D.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} itertations per epoch".format(len(train_loader)))

    iter_num = 0
    max_epoch = max_iterations // len(train_loader) + 1
    lr_ = base_lr

    teacher_net_2d.toggle_dropout(dropout_on=False)
    iterator = tqdm(range(max_epoch), ncols=70)
    best_dice_val = 0
    dice_avg_fn = get_dice_avg_3d(num_classes)
    for epoch_num in iterator:
        student_net_3D.train()
        for i_batch, sampled_batch in enumerate(train_loader):
            optimizer.zero_grad()
            volume_batch, label_batch = sampled_batch["image"], sampled_batch['label']
            volume_batch_r = volume_batch.repeat(1, 1, 1, 1, 1)
            actual_batch_size = volume_batch.size()[0]
            volume_batch_in = volume_batch + torch.clamp(torch.randn_like(volume_batch) * 0.02, -0.05, 0.05)
            volume_batch_in = volume_batch_in.to(device_2d)

            results_2d_to_3d = results_2d_from_3d(input_3d=volume_batch_in,
                                                  model_2d=teacher_net_2d,
                                                  device_2d=device_2d,
                                                  device_3d=device_3d)

            student_results = student_net_3D(torch.cat([volume_batch_in.to(device_3d),
                                                        # uncertainty_2d_to_3d,
                                                        results_2d_to_3d["output_2d_to_3d"]],
                                                       dim=1),
                                             f_seg_2d=results_2d_to_3d["features_seg_2d_to_3d"],
                                             f_sdm_2d=results_2d_to_3d["features_sdm_2d_to_3d"])

            K = 4
            ema_preds, ema_sdm_probs, = torch.zeros([2, K, actual_batch_size,
                                                     num_classes, num_slices, 112, 112])
            for k in range(K):
                ema_inputs = volume_batch_r + torch.clamp(torch.randn_like(volume_batch_r) * 0.02, -0.05, 0.05)
                results_2d_to_3d = results_2d_from_3d(input_3d=ema_inputs,
                                                      model_2d=teacher_net_2d,
                                                      device_2d=device_2d,
                                                      device_3d=device_3d)
                with torch.no_grad():
                    ema_results = teacher_net_3D(torch.cat([ema_inputs.to(device_3d),
                                                            results_2d_to_3d["output_2d_to_3d"]],
                                                           dim=1),
                                                 f_seg_2d=results_2d_to_3d["features_seg_2d_to_3d"],
                                                 f_sdm_2d=results_2d_to_3d["features_sdm_2d_to_3d"])
                    ema_preds[k] = ema_results["segmentation"]
                    ema_sdm_probs[k] = ema_results["sdm"]
            # dimension [K, B, C, D, W, H]
            ema_preds = F.softmax(ema_preds, dim=2)
            uncertainty = -1.0 * torch.sum(ema_preds * log_n(ema_preds + 1e-6, base=num_classes,
                                                             device="cpu"), dim=2,
                                           keepdim=True)
            weights = F.softmax(1 - uncertainty, dim=0)
            ema_probs = torch.sum(ema_preds * weights, dim=0)
            ema_mask = probs2one_hot_general(ema_probs.detach())
            ema_seg_uncertainty = -1.0 * torch.sum(ema_probs * log_n(ema_probs + 1e-6, base=num_classes,
                                                                     device="cpu"), dim=1,
                                                   keepdim=True)

            ema_sdm = ema_sdm_probs.mean(dim=0)
            ema_sdm_uncertainty = ema_sdm_probs.var(dim=0)

            loss_val_l = loss_labelled(seg_probs=torch.softmax(student_results["segmentation"][:labeled_bs, ...],
                                                               dim=1).to(device_2d),
                                       seg_targets=label_batch[:labeled_bs, ...].to(device_2d),
                                       ema_mask=ema_mask[:labeled_bs, ...].to(device_2d),
                                       ema_sdm=ema_sdm[:labeled_bs, ...].to(device_2d),
                                       ema_sdm_uncertainty=ema_sdm_uncertainty[:labeled_bs, ...].to(device_2d),
                                       ema_seg_uncertainty=ema_seg_uncertainty[:labeled_bs, ...].to(device_2d),
                                       epoch=epoch_num,
                                       sdm_probs=student_results["sdm"][:labeled_bs, ...].to(device_2d),
                                       device=device_2d)

            loss_val_u = loss_unlabelled(seg_probs=torch.softmax(student_results["segmentation"][labeled_bs:, ...],
                                                                 dim=1).to(device_2d),
                                         ema_mask=ema_mask[labeled_bs:, ...].to(device_2d),
                                         ema_sdm=ema_sdm[labeled_bs:, ...].to(device_2d),
                                         ema_sdm_uncertainty=ema_sdm_uncertainty[labeled_bs:, ...].to(device_2d),
                                         ema_seg_uncertainty=ema_seg_uncertainty[labeled_bs:, ...].to(device_2d),
                                         epoch=epoch_num,
                                         sdm_probs=student_results["sdm"][labeled_bs:, ...].to(device_2d),
                                         device=device_2d)

            loss_val = loss_val_l + loss_val_u
            loss_val.backward()
            optimizer.step()
            update_ema_variables(student_net_3D, teacher_net_3D, alpha=0.99, global_step=iter_num)
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


        def predict_3d(x):
            ret_2d_3d = results_2d_from_3d(input_3d=x,
                                           model_2d=teacher_net_2d,
                                           device_2d=device_2d,
                                           device_3d=device_3d)
            ret_3d = student_net_3D(torch.cat([x.to(device_3d),
                                               # uncertainty_2d_to_3d,
                                               ret_2d_3d["output_2d_to_3d"]],
                                              dim=1),
                                    f_seg_2d=ret_2d_3d["features_seg_2d_to_3d"],
                                    f_sdm_2d=ret_2d_3d["features_sdm_2d_to_3d"])
            return ret_3d


        # eval
        student_net_3D.eval()
        dice_val_overall = test_all_case_3d(net=predict_3d,
                                            num_classes=num_classes,
                                            patch_size=args.patch_size,
                                            stride_xy=18, stride_z=4,
                                            metric_fn=dice_avg_fn,
                                            avt_fn=torch.nn.Softmax(dim=1),
                                            permute_dim=[(0, 1, 4, 2, 3),
                                                         (0, 1, 3, 4, 2)],
                                            base_dir=os.path.join(load_path_labelled, "val"),
                                            test_list="")
        logging.info('Epoch %d : Dice on val data : %f.' % (epoch_num, dice_val_overall))
        if iter_num % 3000 == 0:
            lr_ = base_lr * 0.1 ** (iter_num // 3000)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
        if epoch_num % 40 == 0 or dice_val_overall > best_dice_val:
            save_mode_path = os.path.join(snapshot_path, 'iter_' + str(epoch_num) + '_student.pth')
            torch.save(student_net_3D.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            save_mode_path = os.path.join(snapshot_path, 'iter_' + str(epoch_num) + '_teacher.pth')
            torch.save(teacher_net_3D.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            save_mode_path = os.path.join(snapshot_path, 'iter_' + str(epoch_num) + '_optimiser.pth')
            torch.save(optimizer.state_dict(), save_mode_path)
            logging.info("saved optimiser to {}".format(save_mode_path))
            if dice_val_overall > best_dice_val:
                best_dice_val = dice_val_overall
    iterator.close()
    save_mode_path = os.path.join(snapshot_path, 'final_student.pth')
    torch.save(student_net_3D.state_dict(), save_mode_path)
    logging.info("saved model to {}".format(save_mode_path))
    save_mode_path = os.path.join(snapshot_path, 'final_teacher.pth')
    torch.save(teacher_net_3D.state_dict(), save_mode_path)
    logging.info("saved model to {}".format(save_mode_path))
    save_mode_path = os.path.join(snapshot_path, 'final_optimiser.pth')
    torch.save(optimizer.state_dict(), save_mode_path)
    logging.info("saved optimiser to {}".format(save_mode_path))
    writer.close()
