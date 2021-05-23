from __future__ import absolute_import
from __future__ import division

from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import random
import os
import torch as t
from torchvision import transforms as tvtsf


class SiameseDataset(Dataset):

    def __init__(self, opt, split, transform=None):
        """

        :param root_dir: the root directory of dataset files
         e.g.'C:/Users/Xue Gao/Documents/Masterarbeit/Dataset/train_new/'
        :param transform:
        :param anno_dir: the root directory of annotation files
         e.g.'C:/Users/Xue Gao/Documents/Masterarbeit/Dataset/anno_train_new/'
        :param cls_number: number of object classes
        :param setsize: lenth of dataset
        """
        self.transform = transform

        if split == 'train':
            self.setsize = opt.set_size
            self.root_dir = opt.img_data_dir
            self.anno_dir = opt.label_data_dir
            self.cls_number = opt.cls_number
        if split == 'eval':
            self.root_dir = opt.val_img_dir
            self.setsize = opt.val_set_size
            self.cls_number = opt.cls_number_val

    def __len__(self):
        return self.setsize

    def __getitem__(self, idx):
        """
        :return: img: numpy.ndarray, (C,H,W), RGB(-1, 1)
                 label: 1 --> positive pair
                        0 --> negative pair
                 bbox: numpy.ndarray with format (R,4)
        """
        img1 = None
        img2 = None
        bbox1 = []
        bbox2 = []
        label = None
        #
        # print('getitem')

        # randomly get the image pair and change the image format
        # idx = random.randint(0, 1)
        # should_same = random.randint(0, 1)
        if idx % 2:
            category = random.randint(0, self.cls_number - 1)
            img_dir = self.root_dir + str(category)
            img1_name = random.choice(os.listdir(img_dir))
            img2_name = random.choice(os.listdir(img_dir))
            img1_path = img_dir + '/' + img1_name
            img2_path = img_dir + '/' + img2_name
            anno1_path = self.anno_dir + img1_name[:-4] + '.txt'
            anno2_path = self.anno_dir + img2_name[:-4] + '.txt'

            img1 = read_image(img1_path)
            img1 = pytorch_normalze(img1)
            img2 = read_image(img2_path)
            img2 = pytorch_normalze(img2)
            label = 1.

            # Get the Bbox data from annotation file
            anno1 = open(anno1_path, 'r')
            anno2 = open(anno2_path, 'r')

            for item in anno1.readline().split():
                bbox1.append(float(item))
            bbox1.pop(0)
            bbox1 = np.asarray(bbox1)
            bbox1 = np.expand_dims(bbox1, axis=0)

            for itm in anno2.readline().split():
                bbox2.append(float(itm))
            bbox2.pop(0)
            bbox2 = np.asarray(bbox2)
            bbox2 = np.expand_dims(bbox2, axis=0)

            anno1.close()
            anno2.close()

        else:
            category1, category2 = random.randint(0, self.cls_number - 1), random.randint(0, self.cls_number - 1)
            while category1 == category2:
                category2 = random.randint(0, self.cls_number - 1)

            img_dir1, img_dir2 = self.root_dir + str(category1), self.root_dir + str(category2)
            img1_name = random.choice(os.listdir(img_dir1))
            img2_name = random.choice(os.listdir(img_dir2))
            img1_path = img_dir1 + '/' + img1_name
            img2_path = img_dir2 + '/' + img2_name
            anno1_path = self.anno_dir + img1_name[:-4] + '.txt'
            anno2_path = self.anno_dir + img2_name[:-4] + '.txt'

            img1 = read_image(img1_path)
            img1 = pytorch_normalze(img1)
            img2 = read_image(img2_path)
            img2 = pytorch_normalze(img2)
            label = 0.0

            anno1 = open(anno1_path, 'r')
            anno2 = open(anno2_path, 'r')

            for item in anno1.readline().split():
                bbox1.append(float(item))
            bbox1.pop(0)
            bbox1 = np.asarray(bbox1)
            bbox1 = np.expand_dims(bbox1, axis=0)

            for itm in anno2.readline().split():
                bbox2.append(float(itm))
            bbox2.pop(0)
            bbox2 = np.asarray(bbox2)
            bbox2 = np.expand_dims(bbox2, axis=0)

            anno1.close()
            anno2.close()

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        # print('img1_path')
        # print(img1_path)
        # print('img2_path')
        # print(img2_path)
        # print(label)

        return (img1, img2, bbox1, bbox2), t.from_numpy(np.array([label], dtype=np.float32))


def read_image(path, dtype=np.float32, color=True):
    """Read an image from a file.
    This function reads an image from given file. The image is CHW format and
    the range of its value is :math:`[0, 255]`. If :obj:`color = True`, the
    order of the channels is RGB.
    Args:
        path (str): A path of image file.
        dtype: The type of array. The default value is :obj:`~numpy.float32`.
        color (bool): This option determines the number of channels.
            If :obj:`True`, the number of channels is three. In this case,
            the order of the channels is RGB. This is the default behaviour.
            If :obj:`False`, this function returns a grayscale image.
    Returns:
        ~numpy.ndarray: An image.
    """

    f = Image.open(path)
    try:
        if color:
            img = f.convert('RGB')
        else:
            img = f.convert('P')
        img = np.asarray(img, dtype=dtype)
    finally:
        if hasattr(f, 'close'):
            f.close()

    if img.ndim == 2:
        # reshape (H, W) -> (1, H, W)
        return img[np.newaxis]
    else:
        # transpose (H, W, C) -> (C, H, W)
        return img.transpose((2, 0, 1))


def pytorch_normalze(img):
    """
    https://github.com/pytorch/vision/issues/223
    return appr -1~1 RGB
    """
    img = img / 255.
    normalize = tvtsf.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    img = normalize(t.from_numpy(img))
    return img.numpy()
