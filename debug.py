from model.net import vgg_16
from model.rpn import RegionProposalNetwork
from model.utils.creator_tool import AnchorTargetCreator, ProposalFilter
from torchvision.models import vgg16
import torch
from torch import nn
import numpy as np
import utils.array_tool as at
from Train.siamese_dataset import read_image, pytorch_normalze

Img = read_image('/home/betago/Documents/Thesis/Model/Dataset/train_2/1/1248.jpg')
Img1 = read_image('/home/betago/Documents/Thesis/Model/Dataset/train_2/0/11.jpg')
bbox_path = '/home/betago/Documents/Thesis/Model/Dataset/anno_train_2/1248.txt'
img = pytorch_normalze(Img)
img1 = pytorch_normalze(Img1)

bbox = list()
anno = open(bbox_path, 'r')
for item in anno.readline().split():
    bbox.append(float(item))
bbox.pop(0)
bbox = np.asarray(bbox)
bbox = np.expand_dims(bbox,axis=0)
img = torch.from_numpy(np.expand_dims(img,axis=0))
img1 = torch.from_numpy(np.expand_dims(img1,axis=0))

extractor = vgg_16()
#
# feature = extractor(img)

# extractor = nn.Sequential(*features)
feature = extractor(img)
feature1 = extractor(img1)
check = torch.eq(feature, feature1)
check = at.tonumpy(check)
print(check[np.where(check == False)])




img_size = img.shape[2:]
rpn = RegionProposalNetwork(512, 512, [0.5,1,2], [8,16,32], 16)

rpn_locs, rpn_scores, rois, roi_indices, anchor = rpn(feature, img_size, scale=1)

print(rois.shape)

proposal_target = ProposalFilter(128,0.5,0.25,0.5,0.1)

sample_roi, gt_roi_loc, gt_roi_label,pos_roi_per_this_image = proposal_target(rois,bbox)

