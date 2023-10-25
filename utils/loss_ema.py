import math
import torch
import torch.nn as nn

from torch import Tensor
from utils.util import convert_to_sdm
from utils.losses import DiceLoss, probs_to_onehot

from utils.util import probs2one_hot_general


class LabeledPenalty_NoRec(object):
    def __init__(self, sdm_weight=1, num_classes=2, max_epoch=100):
        self.sdm_criterion = nn.MSELoss()
        self.seg_criterion = DiceLoss(num_classes=num_classes)
        self.max_epoch = max_epoch
        self.sdm_weight = sdm_weight

        self.this_dice = 0
        self.this_sdm = 0
        self.this_sdm_con = 0
        self.this_seg_con = 0
        self.this_loss = 0
        self.counter = 0

    def __call__(self, seg_probs: Tensor, seg_targets: Tensor, sdm_probs: Tensor,
                 ema_mask: Tensor, ema_sdm: Tensor, ema_seg_uncertainty,
                 ema_sdm_uncertainty, epoch: int, device) -> Tensor:
        sdm_probs = sdm_probs * 2
        sdm_target = convert_to_sdm(seg_targets.detach().cpu()).float().to(device) + 1
        sdm_loss = self.sdm_criterion(sdm_probs, sdm_target)
        seg_loss = self.seg_criterion(seg_probs, seg_targets)
        sdf_consistency = torch.mean(torch.pow(sdm_probs - ema_sdm, 2) * torch.exp(-ema_sdm_uncertainty))
        seg_consistency = torch.pow(seg_probs - ema_mask, 2) * (1 - ema_seg_uncertainty)
        seg_consistency = torch.mean(seg_consistency)

        consistency_weight = 0.1 * math.exp(-5 * math.pow((1 - epoch / self.max_epoch), 2))
        loss = seg_loss + self.sdm_weight * sdm_loss + consistency_weight * \
               (sdf_consistency + seg_consistency)

        self.this_dice += seg_loss.item()
        self.this_sdm += sdm_loss.item()
        self.this_sdm_con += sdf_consistency.item()
        self.this_seg_con += seg_consistency.item()
        self.this_loss += loss.item()
        self.counter += 1

        return loss

    def get_and_reset_l(self):
        ret_val = self.this_dice / self.counter, \
                  self.this_sdm / self.counter, \
                  self.this_seg_con / self.counter, \
                  self.this_sdm_con / self.counter, \
                  self.this_loss / self.counter
        self.this_dice, self.this_sdm, self.this_seg_con, self.this_sdm_con, self.this_loss = [0] * 5
        self.counter = 0
        return ret_val


class UnlabeledPenalty_NoRec(object):
    def __init__(self, max_epoch=100):
        self.sdm_criterion = nn.MSELoss()
        self.max_epoch = max_epoch

        self.this_sdm_seg_loss = 0
        self.this_sdm_con = 0
        self.this_seg_con = 0
        self.this_loss = 0
        self.this_rec = 0
        self.counter = 0

    def __call__(self, seg_probs: Tensor, sdm_probs: Tensor, ema_mask: Tensor, ema_sdm: Tensor, ema_seg_uncertainty,
                 ema_sdm_uncertainty, epoch: int, device) -> Tensor:
        predicted_mask = probs2one_hot_general(seg_probs.detach().cpu())
        sdm = convert_to_sdm(predicted_mask.cpu()).float().to(device) + 1
        sdm_consistency = torch.mean(torch.pow(sdm_probs - ema_sdm, 2) * torch.exp(-ema_sdm_uncertainty))
        seg_consistency = torch.pow(seg_probs - ema_mask, 2) * (1 - ema_seg_uncertainty)
        seg_consistency = torch.mean(seg_consistency)

        sdm_probs = sdm_probs * 2
        sdm_seg_loss = self.sdm_criterion(sdm_probs, sdm)
        unlabeled_weight = 0.1 * math.exp(-5 * math.pow((1 - epoch / self.max_epoch), 2))
        loss = unlabeled_weight * (sdm_consistency + seg_consistency) \
               + sdm_seg_loss * 0.1

        self.this_sdm_seg_loss += sdm_seg_loss.item()
        self.this_sdm_con += sdm_consistency.item()
        self.this_seg_con += seg_consistency.item()
        self.this_loss += loss.item()
        self.counter += 1
        return loss

    def get_and_reset_u(self):
        ret_val = self.this_sdm_seg_loss / self.counter, \
                  self.this_seg_con / self.counter, \
                  self.this_sdm_con / self.counter, \
                  self.this_loss / self.counter
        self.this_sdm_seg_loss, self.this_seg_con, self.this_sdm_con, self.this_loss = [0] * 4
        self.counter = 0
        return ret_val