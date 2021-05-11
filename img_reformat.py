from PIL import Image
import os
import os.path


# for i in range(2,16):
#     rootdir = '/home/betago/Documents/Thesis/Model/Dataset/train_lego/{}/'.format(i)
#
#     file_list = os.listdir(rootdir)
#     print(len(file_list))
#
#     for file in file_list:
#         path = ''
#         path = rootdir + file
#         # print(path)
#         im = Image.open(path)
#         out = im.resize((800, 800), Image.ANTIALIAS)
#         save_path = ''
#         save_path = '/home/betago/Documents/Thesis/Model/Dataset/train_lego/{}/'.format(i) + file
#         out.save(save_path)
root = '/home/betago/Documents/Thesis/Model/Dataset/val/0/'
file = os.listdir(root)
for f in file:
    dir = root + f
    im = Image.open(dir)
    out = im.resize((400, 400), Image.ANTIALIAS)
    new = '/home/betago/Documents/Thesis/real_img_test/time/400/' + f
    out.save(new)
#
# root = '/home/betago/Documents/Thesis/real_img_test/1/4.jpg'
# im = Image.open(root)
# out = im.resize((800, 800), Image.ANTIALIAS)
# out.save(root)
