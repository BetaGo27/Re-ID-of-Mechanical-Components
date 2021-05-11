from torch.nn import functional as F
import torch
import os
from Train.siamese_dataset import SiameseDataset
from Train.train import eval
from utils.config import opt
from torch.utils import data as data_
import numpy as np
import re

root_dir = '/home/betago/Documents/Thesis/Model/Dataset/cfg2/siam_reid_16.pth'
# files = os.listdir(root_dir)
# files.sort(key=lambda f: int(re.sub('\D', '', f)))
# print(files)
# epoch = 0
eval_set = SiameseDataset(opt=opt, split='eval')
#
# for file in files:
eval_dataloader = data_.DataLoader(eval_set, batch_size=1, shuffle=False, num_workers=opt.num_workers)
# model_path = root_dir + file
model_path = root_dir
# print('epoch:')
# print(epoch)
best_distance = eval(val_dataloader=eval_dataloader, path=model_path, epoch=10, epochs=opt.epoch)

print(best_distance)

# epoch += 1


