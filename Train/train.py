from __future__ import absolute_import
import os

import matplotlib
from tqdm import tqdm
from utils.config import opt
from Train.siamese_dataset import SiameseDataset
from model.net import SiameseReID
from torch.utils import data as data_
from Train.trainer import Trainer
from utils import array_tool as at
from utils.eval_tool import evaluate
from utils.plot_tool import plot_roc, plot_accuracy
import torch
import numpy as np
from Evaluation.test import img_test
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')


# def eval(val_dataloader, path, epoch, epochs):
#     with torch.no_grad():
#         distances = []
#         labels = []
#         for ii, (data, target) in tqdm(enumerate(val_dataloader)):
#             img1, img2 = data[0], data[1]
#             img1, img2 = img1.cuda().float(), img2.cuda().float()
#             test_tool = img_test(img1, img2, path)
#             _, distance, _, _ = test_tool()
#
#             if distance.item() != 0.:
#                 distances.append(distance)
#                 labels.append(target)
#
#         labels = np.array([label for label in labels])
#         # print(distances)
#         # print(labels)
#         distances = np.array([dist for dist in distances])
#         true_pos_rate, false_pos_rate, accuracy, roc_auc, best_distances = evaluate(
#             distances=distances, labels=labels
#         )
#         # print the infos
#         print("Accuracy on LFW: {:.4f}+-{:.4f}\t"
#               "ROC Area Under Curve: {:.4f}\tBest distance threshold: {:.2f}+-{:.2f}\t".format(
#             np.mean(accuracy),
#             np.std(accuracy),
#             roc_auc,
#             np.mean(best_distances),
#             np.std(best_distances)
#             ))
#         with open("logs/siameseReID_log.txt", 'a') as f:
#             val_list = [
#                 epoch + 1,
#                 np.mean(accuracy),
#                 np.std(accuracy),
#                 roc_auc,
#                 np.mean(best_distances),
#                 np.std(best_distances),
#             ]
#             log = '\t'.join(str(value) for value in val_list)
#             f.writelines(log + '\n')
#     # try:
#     #     # Plot ROC curve
#     #     plot_roc(
#     #         false_positive_rate=false_pos_rate,
#     #         true_positive_rate=true_pos_rate,
#     #         figure_name="plots/roc_plots/roc_{}_epoch.png".format(epoch + 1)
#     #     )
#         # plot_accuracy(
#         #     log_dir='logs/siameseReID_log.txt',
#         #     epochs=epochs,
#         #     figure_name="plots/accuracies.png"
#         # )
#     # except Exception as e:
#     #     print(e)
#
#     return best_distances


def train(**kwargs):
    opt._parse(kwargs)

    trainset = SiameseDataset(opt=opt, split='train')
    eval_set = SiameseDataset(opt=opt, split='eval')
    print('load data')
    train_dataloader = data_.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=opt.num_workers)
    eval_dataloader = data_.DataLoader(eval_set, batch_size=1, shuffle=False, num_workers=opt.num_workers)

    siam_reid = SiameseReID()
    optimizer = siam_reid.get_optimizer()
    start_epoch = 0

    if opt.resume:
        loaddir = '/home/betago/Documents/Thesis/Model/Dataset/cfg/siam_reid_15.pth'
        checkpoint = torch.load(loaddir)
        siam_reid.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('model loaded')

    print('model completed')
    trainer_ = Trainer(siam_reid).cuda()
    lr_ = opt.lr
    for epoch in range(start_epoch, opt.epoch):
        trainer_.reset_meters()
        # sm_loss_epoch = []
        for ii, (data, target) in tqdm(enumerate(train_dataloader)):
            scale = at.scalar(1)
            img1, img2, bbox1, bbox2 = data[0], data[1], data[2], data[3]
            img1, img2, bbox1, bbox2, target = img1.cuda().float(), img2.cuda().float(), \
                                               bbox1.cuda(), bbox2.cuda(), target.cuda()
            trainer_.train_step(img1, img2, bbox1, bbox2, target, 1)
            # siam_reid.head.fc_cls_loc.weight.register_hook(
            #     lambda grad: print(f'DEBUG: loc grad hook {grad}'))
            # siam_reid.head.fc_embedding.weight.register_hook(
            #     lambda grad: print(f'DEBUG: embed grad hook {grad}'))
            # if ii + 1 % 2 == 0:
            #     trainer_.optimizer.zero_grad()
            #     losses.total_loss.backward()
            #     trainer_.optimizer.step()

            if (ii + 1) % opt.plot_every == 0:
                # plot loss
                trainer_.vis.plot_many(trainer_.get_meter_data())

        # best_distance = eval(val_dataloader=eval_dataloader, model=siam_reid, epoch=epoch, epochs=opt.epoch)
        optimizer = trainer_.siam_reid.optimizer
        lr_ = optimizer.param_groups[0]['lr']
        log_info = 'lr:{}, loss:{}'.format(str(lr_), str(trainer_.get_meter_data()),
                                           )
        trainer_.vis.log(log_info)
        PATH = os.path.join(opt.out_path, 'siam_reid_{}.pth'.format(str(epoch + 1)))
        print("Epoch number {}\n Current loss {}\n".format(
            epoch + 1, trainer_.get_meter_data(),
        ))
        torch.save({'epoch': epoch + 1,
                    'model': siam_reid.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    # 'best_dist_threshold': np.mean(best_distance)
                    }, PATH)
        # if epoch == 9:
        # trainer_.siam_reid.scale_lr(opt.lr_decay)
        #     lr_ = 1e-4
        if epoch == 12:
            trainer_.siam_reid.scale_lr(opt.lr_decay)
            lr_ = 6e-5
        if epoch == 17:
            break


# def inverse_normalize(img):
#     if opt.caffe_pretrain:
#         img = img + (np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1))
#         return img[::-1, :, :]
#     # approximate un-normalize for visualize
#     return (img * 0.225 + 0.45).clip(min=0, max=1) * 255


if __name__ == '__main__':
    import fire
    fire.Fire(train)
