from pprint import pprint


# Default Configs for training
# NOTE that, config items could be overwriten by passing argument through command line.
# NOTE that, config items could be overwriten by passing argument through command line.
# e.g. --voc-data-dir='./data/'

class Config:
    # data
    img_data_dir = '/home/betago/Documents/Thesis/Model/Dataset/train-ori/'
    label_data_dir = '/home/betago/Documents/Thesis/Model/Dataset/anno-ori/'
    val_img_dir = '/home/betago/Documents/Thesis/Model/Dataset/val_f/'
    # val_label_dir = '/home/betago/Documents/Thesis/Model/Dataset/anno_val/'
    out_path = '/home/betago/Documents/Thesis/Model/Dataset/cfg/'
    set_size = 5000
    val_set_size = 240
    # set_size = 6734
    cls_number = 50
    cls_number_val = 4
    min_size = 600  # image resize
    max_size = 1000  # image resize
    num_workers = 4
    test_num_workers = 4
    resume = True

    # sigma for l1_smooth_loss
    rpn_sigma = 3.
    roi_sigma = 1.

    # param for optimizer
    # 0.0005 in origin paper but 0.0001 in tf-faster-rcnn
    weight_decay = 0.0005
    lr_decay = 0.1  # 1e-3 -> 1e-4
    lr = 6e-4
    # lr = 6e-5

    # visualization
    env = 'siam_reid'  # visdom env
    port = 8097
    plot_every = 80  # vis every N iter

    # preset
    data = 'voc'
    pretrained_model = 'vgg16'

    # training
    epoch = 20

    use_adam = True  # Use Adam optimizer
    use_chainer = False  # try match everything as chainer
    use_drop = False  # use dropout in RoIHead
    # debug
    debug_file = '/tmp/debugf'

    caffe_pretrain = False  # use caffe pretrained model instead of torchvision
    # caffe_pretrain_path = '/home/betago/Documents/Thesis/Model/utils/fasterrcnn_caffe_pretrain.pth'

    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}


opt = Config()
