import os.path
import random


# for i in range(8,11):
#     rootdir = '/home/betago/Documents/Thesis/Model/Dataset/train_2/{}/'.format(i)

rootdir = '/home/betago/Documents/Thesis/Model/Dataset/anno/'
file_list = os.listdir(rootdir)

for file in file_list:
    src_file = ''
    src_file = rootdir + file
    new_flie = ''
    new_flie = rootdir + "a" + file

    os.rename(src_file, new_flie)






