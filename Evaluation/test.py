from model.net import vgg16
from model.rpn import RegionProposalNetwork
from model.net import SiameseReID
from utils import array_tool as at
from model.utils.bbox_tools import loc2bbox
import torch
import matplotlib.patches as patches
import numpy as np
from torch.nn import functional as F
from torchvision.ops import nms
from Train.siamese_dataset import read_image, pytorch_normalze
import matplotlib.pyplot as plt
import matplotlib
import math
import random
import os
import time

# matplotlib.use('TkAgg')


def clip_bboxs_on_image(rois, roi_locs):
    """

    :param rois: Tensor
    :param roi_locs: Tensor
    :return: bbox: Tensor
    """
    loc_normalize_mean = (0., 0., 0., 0.)
    loc_normalize_std = (0.1, 0.1, 0.2, 0.2)
    mean = torch.Tensor(loc_normalize_mean).cuda(). \
        repeat(2)[None]
    std = torch.Tensor(loc_normalize_std).cuda(). \
        repeat(2)[None]

    roi_locs = roi_locs * std + mean
    roi_loc = roi_locs.view(-1, 2, 4)
    rois = at.totensor(rois)
    rois = rois.view(-1, 1, 4).expand_as(roi_loc)
    bbox = loc2bbox(at.tonumpy(rois).reshape((-1, 4)), at.tonumpy(roi_loc).reshape((-1, 4)))
    bbox = at.totensor(bbox)
    box = bbox.view(-1, 8)
    box[:, 0::2] = (box[:, 0::2]).clamp(min=0, max=800)
    box[:, 1::2] = (box[:, 1::2]).clamp(min=0, max=800)
    box = box.reshape((-1, 2, 4))[:, 1, :]

    return box


class img_test(object):
    def __init__(self, query_img, target_img, model_path):
        # when testing
        self.query = torch.from_numpy(np.expand_dims(query_img, axis=0)).cuda()
        self.target = torch.from_numpy(np.expand_dims(target_img, axis=0)).cuda()
        self.model_path = model_path
        # when evaluate
        # self.query = query_img
        # self.target = target_img
        # self.target = target_img
        # self.model_path = model_path

    def __call__(self):
        """ Load Model """
        model = SiameseReID().cuda()
        # loaddir = '/home/betago/Documents/Thesis/Model/Dataset/temp/siam_reid_16.pth'
        checkpoint = torch.load(self.model_path)
        model.load_state_dict(checkpoint['model'])

        rois1, rois2, embd1, embd2, roi_locs1, roi_locs2, roi_score1, roi_score2 = model(
            self.query, self.target
        )
        bboxes = []
        scores = []
        score1 = (F.softmax(at.totensor(roi_score1[:, 1]), dim=0))
        score2 = (F.softmax(at.totensor(roi_score2[:, 1]), dim=0))
        # print(np.sort(score1.cpu().numpy()))
        # print(np.sort(score2.cpu().numpy()))

        """prepare query embeddings"""
        # res, idx = torch.topk(score1, 5)
        bbox1 = clip_bboxs_on_image(rois1, roi_locs1)
        keep1 = score1 > 0.006
        score1 = score1[keep1]
        bbox1 = bbox1[keep1]
        embd1 = embd1[keep1]
        sample1 = nms(bbox1, score1, 0.3)
        # max_idx = np.argmax(score1[sample1].cpu().numpy())
        bbox1 = bbox1[sample1]
        # embd1 = at.tonumpy(embd1[sample1])
        # embd1 = np.expand_dims(embd1[max_idx], axis=0)
        embd1 = embd1[sample1][0]
        embd1 = embd1[None, :]

        """use the embedding to determine target"""
        bbox2 = clip_bboxs_on_image(rois2, roi_locs2)
        keep2 = score2 > 0.006
        score2 = score2[keep2]
        # print(score2)
        bbox2 = bbox2[keep2]
        # print(bbox2)
        sample2 = nms(bbox2, score2, 0.3)
        bbox2 = bbox2[sample2]
        # print(bbox2)
        score2 = score2[sample2]
        bboxes.append(bbox2.cpu().numpy())
        if len(sample1) == 0 or len(sample2) == 0:
            dist = torch.tensor([0.])
        else:
            dist = self.calculate_cosine_dist(embd1, embd2[sample2])
            # max_idx = np.unravel_index(np.argmax(dist, axis=None), dist.shape)
            # max_idx = np.unique(max_idx, axis=0)
            # print(dist)
        # min_idx = torch.argmin(dist, dim=1)
        # dist[0, min_idx] = 0.9
        # min_idx = torch.argmin(dist, dim=1)
        bboxes = np.concatenate(bboxes, axis=0).astype(np.float32)
        # print(dist)
        # print(bboxes)
        # print(min_idx)
        dist = dist[0, 0]
        score2 = score2[0]
        bboxes = bboxes[0]
        # dist = dist[0, min_idx]
        # score2 = score2[min_idx]
        # bboxes = bboxes[min_idx]
        # bboxes.append(bbox2.cpu().numpy())


        return bboxes, dist, score2, bbox1

    def calculate_cosine_dist(self, embed1, embed2):
        embd1 = at.totensor(embed1)
        embd2 = at.totensor(embed2)
        embd1 = self.l2_norm(embd1)
        embd2 = self.l2_norm(embd2)
        # cos = torch.matmul(embd1, embd2)
        cos = torch.einsum('ik,jk->ij', embd1, embd2)
        cos = torch.clamp(cos, min=-1 + 1e-6, max=1 - 1e-6)

        ang_dist = torch.div(torch.acos(cos), math.pi)  # [1,0]
        # avg_cos_dist = torch.mean(ang_dist)
        return ang_dist

    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        # normp = torch.sum(buffer).add_(1e-10)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        # _output = torch.div(input, norm.item())
        output = _output.view(input_size)
        return output


#
# loaddir = '/home/betago/Documents/Thesis/Model/Dataset/filter/siam_reid_14.pth'
# # query_path = '/home/betago/Documents/Thesis/real_img_test/f1/q4.jpg'
# query_path = '/home/betago/Documents/Thesis/Model/Dataset/val_f/2/1.jpg'
# # target_dir = '/home/betago/Documents/Thesis/real_img_test/time/3.jpg'
# target_dir = '/home/betago/Documents/Thesis/Model/Dataset/val_f/0/'
# bboxArr = []
#
# query_img = read_image(query_path)
# query_2_plot = np.array(query_img.transpose((1, 2, 0)), dtype=np.intc)
# query = pytorch_normalze(query_img)
#
# target_name = random.choice(os.listdir(target_dir))
# target_path = target_dir + '/' + target_name
#
# target_img = read_image(target_path)
# target_2_plot = np.array(target_img.transpose((1, 2, 0)), dtype=np.intc)
# target = pytorch_normalze(target_img)
#
# test_tool = img_test(query, target, loaddir)
# # pre_time = time.time()
# bbox, dist, score2, bbox1 = test_tool()
# # end_time = time.time()
# # print('Runtime' + '--- %s seconds ---'%(end_time-pre_time))
#
# bbox1 = bbox1.cpu().numpy()
# bbox1 = bbox1[0]
#
# fig = plt.figure()
# ax_q = fig.add_subplot(1, 2, 1)
# ax_q.title.set_text('Query Image')
# plt.imshow(query_2_plot)
# y_min_q, x_min_q, y_max_q, x_max_q = bbox1[0], bbox1[1], bbox1[2], bbox1[3]
# h_q = y_max_q - y_min_q
# w_q = x_max_q - x_min_q
# ax = plt.gca()
# rect_q = patches.Rectangle((x_min_q, y_min_q), w_q, h_q, linewidth=1, edgecolor='r', facecolor='none')
# ax.add_patch(rect_q)
#
# ax_t = fig.add_subplot(1, 2, 2)
# ax_t.title.set_text('Target Image')
# plt.imshow(target_2_plot)
# #
# y_min, x_min, y_max, x_max = bbox[0], bbox[1], bbox[2], bbox[3]
# h = y_max - y_min
# w = x_max - x_min
# ax = plt.gca()
# rect = patches.Rectangle((x_min, y_min), w, h, linewidth=1, edgecolor='r', facecolor='none')
# ax.add_patch(rect)
# label = "similarity: " + ("%.2f" % (1 - dist))
# plt.text(x_min, y_min, label, color='red')
#
# # dist_ind = 0
# # for i in bbox:
# #     print(dist_ind)
# #     y_min, x_min, y_max, x_max = i[0], i[1], i[2], i[3]
# #     h = y_max - y_min
# #     w = x_max - x_min
# #     ax = plt.gca()
# #     rect = patches.Rectangle((x_min, y_min), w, h, linewidth=1, edgecolor='r', facecolor='none')
# #     ax.add_patch(rect)
# #     label = "similarity: " + ("%.2f" % (1 - dist[0][dist_ind]))
# #     plt.text(x_min, y_min, label, color='red')
# #     dist_ind += 1
#
#
# plt.show()

# # bboxArr.append(bbox)
# #
# # print(bboxArr)
