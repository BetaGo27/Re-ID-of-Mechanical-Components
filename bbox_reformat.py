import fnmatch
import os
from os import listdir, getcwd
from os.path import join
import xml.etree.ElementTree as ET
import pickle


def convert(size, box):
    H, W = size[0], size[1]

    x = box[0] * W
    y = box[1] * H
    w = box[2] * W
    h = box[3] * H

    x_min = x - (w / 2)
    x_min = int(x_min)
    y_min = y - (h / 2)
    y_min = int(y_min)
    x_max = x + (w / 2)
    x_max = int(x_max)
    y_max = y + (h / 2)
    y_max = int(y_max)

    return y_min, x_min, y_max, x_max


def convert_annotation(filename):
    in_file = open('C:/Users/Xue Gao/Documents/Masterarbeit/Dataset/anno_train/%s.txt' % (filename))
    out_file = open('C:/Users/Xue Gao/Documents/Masterarbeit/Dataset/anno_train_new/%s.txt' % (filename), 'w')
    size = (800, 800)
    line = in_file.readline()
    data = line.split()

    cls_id = data[0]
    box = (float(data[1]), float(data[2]), float(data[3]), float(data[4]))
    bbox = convert(size, box)
    out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bbox]) + '\n')


anno_path = 'C:/Users/Xue Gao/Documents/Masterarbeit/Dataset/anno_train/'
img_path = 'C:/Users/Xue Gao/Documents/Masterarbeit/Dataset/train_resize/'
anno_list = os.listdir(anno_path)
img_list = os.listdir(img_path)

for file in anno_list:
    file_path = anno_path + file
    filename = file[:-4]
    for f in img_list:
        f = f[:-4]
        if filename == f:
            convert_annotation(filename)

