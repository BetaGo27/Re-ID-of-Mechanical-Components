from __future__ import absolute_import

from collections import namedtuple
from torch import nn
from utils.config import opt
import torch.nn.functional as F
from model.utils.creator_tool import AnchorTargetCreator, ProposalFilter

import torch as t
from utils import array_tool as at
from torchnet.meter import ConfusionMeter, AverageValueMeter
from sklearn.metrics import pairwise
from utils.vis_tool import Visualizer
import numpy as np
import math
from torch.autograd import Variable

LossTuple = namedtuple('LossTuple', ['rpn_loc_loss',
                                     'rpn_cls_loss',
                                     'roi_loc_loss',
                                     'roi_cls_loss',
                                     'sm_loss',
                                     'total_loss'])


class Trainer(nn.Module):

    def __init__(self, siam_reid):
        super(Trainer, self).__init__()

        self.siam_reid = siam_reid
        self.rpn_sigma = opt.rpn_sigma
        self.roi_sigma = opt.roi_sigma

        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_filter = ProposalFilter()
        self.loc_normalize_mean = (0., 0., 0., 0.)
        self.loc_normalize_std = (0.1, 0.1, 0.2, 0.2)

        self.optimizer = self.siam_reid.get_optimizer()
        self.vis = Visualizer(env=opt.env)

        # self.rpn_cm = ConfusionMeter(2)
        # self.roi_cm = ConfusionMeter(2)
        self.meters = {k: AverageValueMeter() for k in LossTuple._fields}

    def forward_once(self, img, bbox, scale):
        """

        :param img: size: (1, C, H, W)
        :param bbox: size: (1, R, 4)
        :param target: 1, when positive pair; 0, when negative pair
        :param scale: scaling applied to the raw image during preprocessing
        :return: embeddings (n,1), Losses
        """
        n = bbox.shape[0]
        if n != 1:
            raise ValueError('Currently only batch size 1 is supported.')

        _, _, H, W = img.shape
        img_size = (H, W)

        features = self.siam_reid.extractor(img)
        # fe = at.tonumpy(features)
        # fe = fe[np.where(fe > 0)]
        # print(fe)

        rpn_locs, rpn_scores, rois, roi_indices, anchor = \
            self.siam_reid.rpn(features, img_size, scale)

        # Dimension reduction
        bbox = bbox[0]
        rpn_score = rpn_scores[0]
        rpn_loc = rpn_locs[0]
        roi = rois
        # Create Proposal Target
        # print('roi')
        # print(roi.shape)
        # print('at.tonumpy(bbox)')
        # print(at.tonumpy(bbox).shape)

        sample_roi, gt_roi_loc, gt_roi_label, pos_number = \
            self.proposal_filter(roi, at.tonumpy(bbox), self.loc_normalize_mean, self.loc_normalize_std)

        sample_roi_index = t.zeros(len(sample_roi))

        roi_scores, roi_locs_reg, embeddings = \
            self.siam_reid.head(features, sample_roi, sample_roi_index, pos_number)
        # embeddings.register_hook(lambda grad: print(f'DEBUG: embed grad hook {grad.shape}'))

        # -------------------- RPN Loss ------------------------- #
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(at.tonumpy(bbox), anchor, img_size)
        gt_rpn_label = at.totensor(gt_rpn_label).long()
        gt_rpn_loc = at.totensor(gt_rpn_loc)

        rpn_loc_loss = loc_loss_generator(rpn_loc, gt_rpn_loc, gt_rpn_label.data, self.rpn_sigma)
        rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label.cuda(), ignore_index=-1)
        _gt_rpn_label = gt_rpn_label[gt_rpn_label > -1]
        _rpn_score = at.tonumpy(rpn_score)[at.tonumpy(gt_rpn_label) > -1]

        # self.rpn_cm.add(at.totensor(_rpn_score, False), _gt_rpn_label.data.long())

        # -------------------- ROI Loss ------------------------- #
        # roi_loc: (n, 8) --> (n, 2, 4) --> (n, 4)
        n_sample = roi_locs_reg.shape[0]
        roi_locs_reg = roi_locs_reg.view(n_sample, -1, 4)
        roi_loc = roi_locs_reg[t.arange(0, n_sample).long().cuda(),
                               at.totensor(gt_roi_label).long()]

        gt_roi_label = at.totensor(gt_roi_label).long()
        gt_roi_loc = at.totensor(gt_roi_loc)

        roi_loc_loss = loc_loss_generator(roi_loc.contiguous(), gt_roi_loc, gt_roi_label.data, self.roi_sigma)
        roi_cls_loss = nn.CrossEntropyLoss()(roi_scores, gt_roi_label.cuda())

        # self.roi_cm.add(at.totensor(roi_scores, False), gt_roi_label.data.long())

        losses = [rpn_loc_loss,
                  rpn_cls_loss,
                  roi_loc_loss,
                  roi_cls_loss]

        return embeddings, losses

    def forward(self, img1, img2, bbox1, bbox2, target, scale):
        """
        :param target: 1, when positive pair; 0, when negative pair
        """
        embed1, losses1 = self.forward_once(img1, bbox1, scale)
        embed2, losses2 = self.forward_once(img2, bbox2, scale)
        # embed1.register_hook(lambda grad: print(f'DEBUG: embed1 grad hook {grad}'))

        # -------------------- Similarity Measure Loss ------------------------- #
        # embed1.requires_grad = True
        # embed2.requires_grad = True

        n_embd1 = embed1.shape[0]
        n_embd2 = embed2.shape[0]
        # embd1 = at.totensor(embedding1)
        # embd2 = at.totensor(embedding2)
        if n_embd1 > 6:
            embed1 = embed1[0:6, :]
        if n_embd2 > 6:
            embed2 = embed2[0:6, :]
        # embed1.register_hook(lambda grad: print(f'DEBUG: embed1 grad hook {grad.shape}'))

        # embd1 = Variable(embd1.data, requires_grad = True)
        # embd2 = Variable(embd2.data, requires_grad=True)

        # check = t.eq(embd1[0], embd2[0])
        # check = at.tonumpy(check)
        # print(at.tonumpy(embd1[0])[np.where(check == False)])
        embed1 = l2_norm(embed1)
        embed2 = l2_norm(embed2)

        # cos = pairwise.linear_kernel(embd1.cpu(), embd2.cpu())  # Output size (n_embd1, n_embd2)
        cos = t.einsum('ik,jk->ij', embed1, embed2)
        # cos = t.matmul(embd1, embd2)
        cos = t.clamp(cos, min=-1 + 1e-6, max=1 - 1e-6)
        # cos_dist = 1 - np.arccos(cos) / np.pi   # [0,1]
        # ang_dist = 1 - t.div(t.acos(cos), math.pi)
        ang_dist = t.div(t.acos(cos), math.pi)  # (1,0)

        # ang_dist = t.sort(ang_dist)
        # if len(ang_dist) > 2:
        #     if target.item() == 1.:
        #         # mask = np.where(ang_dist > thres_l)
        #         # cos_dist = ang_dist[mask]
        #         ang_dist =
        #     elif target.item() == 0.:
        #         mask = np.where(ang_dist < thres_h)
        #         cos_dist = ang_dist[mask]

        # if cos_dist.size == 0:
        #     sm_loss = t.tensor([1 - 1e-4]).cuda()
        # else:
        # avg_ang_dist = t.mean(ang_dist)
        # sm_loss = 0.5 * (target.float() * avg_ang_dist +
        #                  (1 + (-1 * target)).float() * F.relu(1 - (avg_ang_dist + 1e-7).sqrt()).pow(2))
        log_ang = t.log(ang_dist)
        neg_log_ang = t.log(1 - ang_dist)
        sm_loss = -t.mean(target * neg_log_ang + (1 - target) * log_ang)

        # Task Importance parameter
        alpha_rpn = 0.5
        alpha_roi = 0.5
        alpha_sm = 1.

        rpn_loc_loss_total = (losses1[0] + losses2[0]) / 2
        rpn_cls_loss_total = (losses1[1] + losses2[1]) / 2
        roi_loc_loss_total = (losses1[2] + losses2[2]) / 2
        roi_cls_loss_total = (losses1[3] + losses2[3]) / 2

        Losses = [rpn_loc_loss_total, rpn_cls_loss_total, roi_loc_loss_total, roi_cls_loss_total, sm_loss]
        # Losses = [rpn_cls_loss_total, roi_cls_loss_total, sm_loss]
        LossTotal = alpha_rpn * (Losses[0] + Losses[1]) + alpha_roi * (Losses[2] + Losses[3]) + alpha_sm * Losses[4]
        # LossTotal = alpha_roi * (Losses[2] + Losses[3]) + alpha_sm * Losses[4]
        Losses = Losses + [LossTotal]

        return LossTuple(*Losses)

    def train_step(self, img1, img2, bbox1, bbox2, target, scale):
        self.optimizer.zero_grad()
        losses = self.forward(img1, img2, bbox1, bbox2, target, scale)
        losses.total_loss.backward()
        self.optimizer.step()
        # self.sm_loss = losses.sm_loss
        self.update_meters(losses)
        return losses

    def update_meters(self, losses):
        # print(losses)
        loss_d = {k: at.scalar(v) for k, v in losses._asdict().items()}
        for key, meter in self.meters.items():
            meter.add(loss_d[key])

    def reset_meters(self):
        for key, meter in self.meters.items():
            meter.reset()
        # self.roi_cm.reset()
        # self.rpn_cm.reset()

    def get_meter_data(self):
        return {k: v.value()[0] for k, v in self.meters.items()}

    def get_sm_loss(self):
        return self.sm_loss


# used for doing box regression
def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()


def loc_loss_generator(pred_loc, gt_loc, gt_label, sigma):
    in_weight = t.zeros(gt_loc.shape).cuda()
    # Localization loss is calculated only for positive rois
    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)

    loc_loss /= ((gt_label >= 0).sum().float())  # ignore gt_label==-1 for rpn_loss
    return loc_loss


def l2_norm(input):
    input_size = input.size()
    buffer = t.pow(input, 2)
    normp = t.sum(buffer, 1).add_(1e-10)
    # normp = t.sum(buffer).add_(1e-10)
    norm = t.sqrt(normp)
    _output = t.div(input, norm.view(-1, 1).expand_as(input))
    # _output = t.div(input, norm.item())
    output = _output.view(input_size)
    return output
