from __future__ import absolute_import
import torch as t
import numpy as np
from torch import nn
import torchvision
from torchvision.models import vgg16
from torchvision.ops import RoIPool
from model.rpn import RegionProposalNetwork
from model.utils.creator_tool import ProposalFilter
from utils import array_tool as at
from utils.config import opt
from torchvision.ops import nms
from torch.nn import functional as F


# def feature_extractor(i):
#     """i: input tensor,size((1,3,'',''))"""
#     model = torchvision.models.resnet34(pretrained=False)
#     newmodel = torch.nn.Sequential(*(list(model.children())[:-2]))
#
#     return newmodel(i)
#
# # dummy = torch.zeros((1,3,612,612)).float()
#
# result = feature_extractor(dummy)
# # print(model)
# # print(newmodel)
# print(result.shape)

###### Featrue Extraction ######

def vgg_16():
    # the 30th layer of features is relu of conv5_3
    # if opt.caffe_pretrain:
    #     model = vgg16(pretrained=False)
    #     if not opt.load_path:
    #         model.load_state_dict(t.load(opt.caffe_pretrain_path))
    # else:
    #     model = vgg16(not opt.load_path)
    """ Use vgg16 to extract feature maps
        Output Vector Size (1, 512,'','')
    """
    model = vgg16(pretrained=True)
    # model.load_state_dict(t.load(opt.caffe_pretrain_path))

    features = list(model.features)[:30]
    classifier = model.classifier

    classifier = list(classifier)
    del classifier[6]
    if not opt.use_drop:
        del classifier[5]
        del classifier[2]
    classifier = nn.Sequential(*classifier)

    # freeze top4 conv
    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad = False

    return nn.Sequential(*features), classifier


class ReID(nn.Module):
    feat_stride = 16

    def __init__(self):
        super(ReID, self).__init__()
        self.extractor, self.cls_layers = vgg_16()
        self.rpn = RegionProposalNetwork(512, 512, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32],
                                         feat_stride=self.feat_stride)
        self.head = HeadNet(roi_size=7, spatial_scale=(1. / self.feat_stride), classifier=self.cls_layers)

    def forward(self, im):
        """
                :param im: 4D image
                :return:
                """
        im_size = im.shape[2:]
        features = self.extractor(im)

        rpn_locs, rpn_scores, rois, roi_indices, anchor = \
            self.rpn(features, im_size, scale=1)

        roi_scores, roi_locs_reg, embeddings = self.head(features, rois, roi_indices, None)

        return rois, embeddings, roi_locs_reg, roi_scores


class SiameseReID(nn.Module):
    feat_stride = 16

    def __init__(self):
        super(SiameseReID, self).__init__()
        self.extractor, self.cls_layers = vgg_16()
        self.rpn = RegionProposalNetwork(512, 512, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32],
                                         feat_stride=self.feat_stride)
        self.head = HeadNet(roi_size=7, spatial_scale=(1. / self.feat_stride), classifier=self.cls_layers)
        # self.proposal_target_layer = ProposalFilter()

    def forward_once(self, im):
        """
        :param im: 4D image
        :return:
        """
        im_size = im.shape[2:]
        features = self.extractor(im)

        rpn_locs, rpn_scores, rois, roi_indices, anchor = \
            self.rpn(features, im_size, scale=1)

        roi_scores, roi_locs_reg, embeddings = self.head(features, rois, roi_indices, None)

        return rois, embeddings, roi_locs_reg, roi_scores

    def forward(self, im1, im2):
        rois1, embd1, roi_locs1, roi_score1 = self.forward_once(im1)
        rois2, embd2, roi_locs2, roi_score2 = self.forward_once(im2)

        return rois1, rois2, embd1, embd2, roi_locs1, roi_locs2, roi_score1, roi_score2

    def get_optimizer(self):
        """
        return optimizer, It could be overwriten if you want to specify
        special optimizer
        """
        lr = opt.lr
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': opt.weight_decay}]
        if opt.use_adam:
            self.optimizer = t.optim.Adam(params)
        else:
            self.optimizer = t.optim.SGD(params, momentum=0.9)

        return self.optimizer

    def scale_lr(self, decay=0.1):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= decay
        return self.optimizer


class HeadNet(nn.Module):
    def __init__(self, roi_size, spatial_scale,
                 classifier):
        super(HeadNet, self).__init__()

        self.classifier = classifier
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.roi_pool = RoIPool((self.roi_size, self.roi_size), self.spatial_scale)

        # self.fc_4096 = nn.Sequential(
        #     nn.Linear(512 * 7 * 7, 4096),
        #     nn.ReLU(),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU()
        # )

        self.fc_embedding = nn.Linear(4096, 128)
        self.fc_cls_loc = nn.Linear(4096, 8)  # For 2 classes, foreground and background
        self.fc_score = nn.Linear(4096, 2)

        normal_init(self.fc_cls_loc, 0, 0.001)
        normal_init(self.fc_score, 0, 0.01)
        normal_init(self.fc_embedding, 0, 0.01)

    def forward(self, x, rois, roi_indices, n_pos):
        """

        :param roi_indices: batch of image
        :param x: 4D image variable.
        :param rois:
        :return: roi_cls_locs, roi_scores, embeddings for filtered positive
                 proposals of each image, format:[N,8]
                 N is the number of pos.rois

        Tips: The pos_rois are only for training. It is different by testing
        """

        # precessing the input data, for all rois
        roi_indices = at.totensor(roi_indices).float()
        rois = at.totensor(rois).float()
        indices_and_rois = t.cat([roi_indices[:, None], rois], dim=1)

        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois = xy_indices_and_rois.contiguous()
        pool = self.roi_pool(x, indices_and_rois)

        if n_pos is not None:
            pool_pos = pool[:n_pos]
            # Get the positive part
        else:
            pool_pos = pool

        pool = pool.view(pool.size(0), -1)
        # fc_4096 = self.fc_4096(pool)
        fc_4096 = self.classifier(pool)
        pool_pos = pool_pos.view(pool_pos.size(0), -1)  # Reformat to [N,25088]
        # fc_4096_pos = self.fc_4096(pool_pos)
        fc_4096_pos = self.classifier(pool_pos)

        roi_cls_locs = self.fc_cls_loc(fc_4096)
        roi_scores = self.fc_score(fc_4096)  # [N,2]
        embeddings = self.fc_embedding(fc_4096_pos)  # Format [n_pos,128]

        return roi_scores, roi_cls_locs, embeddings


def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
