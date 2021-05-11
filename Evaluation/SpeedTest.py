from model.net import ReID
from utils import array_tool as at
from model.utils.bbox_tools import loc2bbox
import torch
import numpy as np
from torch.nn import functional as F
from torchvision.ops import nms
from Train.siamese_dataset import read_image, pytorch_normalze
import math
import os
import time


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


def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-10)
    # normp = torch.sum(buffer).add_(1e-10)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    # _output = torch.div(input, norm.item())
    output = _output.view(input_size)
    return output


def calculate_dist(embed1, embed2):
    embd1 = at.totensor(embed1)
    embd2 = at.totensor(embed2)
    embd1 = l2_norm(embd1)
    embd2 = l2_norm(embd2)
    # cos = torch.matmul(embd1, embd2)
    cos = torch.einsum('ik,jk->ij', embd1, embd2)
    cos = torch.clamp(cos, min=-1 + 1e-6, max=1 - 1e-6)

    ang_dist = torch.div(torch.acos(cos), math.pi)  # [1,0]
    avg_dist = torch.mean(ang_dist)
    return avg_dist


class speed_test(object):
    def __init__(self, model_path):
        self.model_path = model_path

    def __call__(self, img):
        """ Load Model """
        img = torch.from_numpy(np.expand_dims(img, axis=0)).cuda()
        model = ReID().cuda()
        # loaddir = '/home/betago/Documents/Thesis/Model/Dataset/temp/siam_reid_16.pth'
        checkpoint = torch.load(self.model_path)
        model.load_state_dict(checkpoint['model'])

        rois, embd, roi_locs, roi_score = model(img)
        score = (F.softmax(at.totensor(roi_score[:, 1]), dim=0))
        # print(np.sort(score1.cpu().numpy()))

        """prepare embeddings"""
        # res, idx = torch.topk(score1, 5)
        bbox = clip_bboxs_on_image(rois, roi_locs)
        keep = score > 0.006
        score = score[keep]
        bbox = bbox[keep]
        embd = embd[keep]
        sample = nms(bbox, score, 0.3)
        # max_idx = np.argmax(score1[sample1].cpu().numpy())
        embd = embd[sample][0]
        embd = embd[None, :]

        return embd


loaddir = '/home/betago/Documents/Thesis/Model/Dataset/filter/siam_reid_14.pth'
query_path = '/home/betago/Documents/Thesis/real_img_test/time/400/8.jpg'
target_path = '/home/betago/Documents/Thesis/real_img_test/time/400/'

query_img = read_image(query_path)
query = pytorch_normalze(query_img)
test_tool = speed_test(loaddir)
embed1 = test_tool(query)

runTime = []
files = os.listdir(target_path)
for im in files:
    target_dir = target_path + im
    img = read_image(target_dir)
    img = pytorch_normalze(img)
    pre_time = time.time()
    embed2 = test_tool(img)
    dist = calculate_dist(embed1, embed2)
    end_time = time.time()
    print('Runtime' + '--- %s seconds ---' % (end_time - pre_time))
    runTime.append((end_time-pre_time))

meanTime = sum(runTime) / len(runTime)
print(meanTime)





