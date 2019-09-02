import os
import shutil
import argparse
import numpy as np
from tqdm import tqdm
import mxnet as mx
from mxnet import gluon, autograd,nd
from mxnet.gluon.data.vision import transforms
import gluoncv
from PIL import Image
from gluoncv.loss import *
from gluoncv.utils import LRScheduler
from gluoncv.model_zoo.segbase import *
from gluoncv.model_zoo import get_model
from gluoncv.utils.parallel import *
from gluoncv.data.segbase import SegmentationDataset
from nets.seg_zoo import get_model
from mxnet import cpu
import mxnet.ndarray as F
'''
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 deecamp_train.py --model deeplab_v3 --aux --backbone resnet101 --lr 0.01 --syncbn --ngpus 4 --checkname res101 --epochs 50
'''

def parse_args():
    """Training Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(description='MXNet Gluon \
                                     Segmentation')
    # model and dataset
    parser.add_argument('--model', type=str, default='deeplab_v3',
                        help='model name (default: fcn)')
    parser.add_argument('--backbone', type=str, default='resnet101',
                        help='backbone name (default: resnet101)')
    parser.add_argument('--dataset', type=str, default='satellite',
                        help='dataset name (default: pascal)')
    parser.add_argument('--workers', type=int, default=20,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=512,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=512,
                        help='crop image size')
    parser.add_argument('--train-split', type=str, default='train',
                        help='dataset train split (default: train)')
    # training hyper params
    parser.add_argument('--aux', action='store_true', default= False,
                        help='Auxiliary loss')
    parser.add_argument('--aux-weight', type=float, default=0.5,
                        help='auxiliary loss weight')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=16,
                        metavar='N', help='input batch size for \
                        training (default: 16)')
    parser.add_argument('--test-batch-size', type=int, default=16,
                        metavar='N', help='input batch size for \
                        testing (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate (default: 1e-2)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        metavar='M', help='w-decay (default: 1e-4)')
    parser.add_argument('--no-wd', action='store_true',
                        help='whether to remove weight decay on bias, \
                        and beta/gamma for batchnorm layers.')
    # cuda and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--ngpus', type=int,
                        default=len(mx.test_utils.list_gpus()),
                        help='number of GPUs (default: 4)')
    parser.add_argument('--kvstore', type=str, default='device',
                        help='kvstore to use for trainer/module.')
    parser.add_argument('--dtype', type=str, default='float32',
                        help='data type for training. default is float32')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default='default',
                        help='set the checkpoint name')
    # evaluation only
    parser.add_argument('--eval', action='store_true', default= False,
                        help='evaluation only')
    parser.add_argument('--no-val', action='store_true', default= False,
                            help='skip validation during training')
    # synchronized Batch Normalization
    parser.add_argument('--syncbn', action='store_true', default= False,
                        help='using Synchronized Cross-GPU BatchNorm')
    # the parser
    args = parser.parse_args()
    # handle contexts
    if args.no_cuda:
        print('Using CPU')
        args.kvstore = 'local'
        args.ctx = [mx.cpu(0)]
    else:
        print('Number of GPUs:', args.ngpus)
        args.ctx = [mx.gpu(i) for i in range(args.ngpus)]
    # Synchronized BatchNorm
    args.norm_layer = mx.gluon.contrib.nn.SyncBatchNorm if args.syncbn \
        else mx.gluon.nn.BatchNorm
    args.norm_kwargs = {'num_devices': args.ngpus} if args.syncbn else {}
    print(args)
    return args

class Deecamp_Dataset(SegmentationDataset):
    NUM_CLASS = 2
    def __init__(self, root='./satellite/',split='train', mode=None, transform=None, **kwargs):
        super(Deecamp_Dataset, self).__init__(root, split, mode, transform, **kwargs)
        _mask_dir = os.path.join(root,split ,'labels')
        _image_dir = os.path.join(root, split,'imgs')
        self.images = []
        self.masks = []
        for file_name in os.listdir(_image_dir):
            if file_name.endswith('.png'):
                img_path=os.path.join(_image_dir,file_name)
                mask_path=os.path.join(_mask_dir,file_name)
                self.images.append(img_path)
                self.masks.append(mask_path)
    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        mask= Image.open(self.masks[index])
        img, mask = self._img_transform(img), self._mask_transform(mask)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        return img, mask
    def __len__(self):
        return len(self.images)
    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32')
        target[target == 255] = 1
        return F.array(target, cpu(0))
    @property
    def classes(self):
        """Category names."""
        return ('land','backgroud')



class Trainer(object):
    def __init__(self, args):
        self.args = args
        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        # dataset and dataloader
        data_kwargs = {'transform': input_transform, 'base_size': args.base_size,
                       'crop_size': args.crop_size}
        trainset = Deecamp_Dataset(root='./data/', split='train',transform=input_transform)
        valset =Deecamp_Dataset(root='./data/', split='val',transform=input_transform)
        self.train_data = gluon.data.DataLoader(
            trainset, args.batch_size, shuffle=True, last_batch='rollover',
            num_workers=args.workers)
        self.eval_data = gluon.data.DataLoader(valset, args.test_batch_size,
            last_batch='rollover', num_workers=args.workers)
        # create network
        model = get_model(args.model,nclass=2,backbone=args.backbone, norm_layer=args.norm_layer,norm_kwargs=args.norm_kwargs,
                          aux=args.aux,crop_size=args.crop_size)
        model.cast(args.dtype)
        print(model)
        self.net = DataParallelModel(model, args.ctx, args.syncbn)
        self.evaluator = DataParallelModel(SegEvalModel(model), args.ctx)
        # resume checkpoint if needed
        if args.resume is not None:
            if os.path.isfile(args.resume):
                model.load_parameters(args.resume, ctx=args.ctx)
            else:
                raise RuntimeError("=> no checkpoint found at '{}'" \
                    .format(args.resume))
        # create criterion
        criterion = MixSoftmaxCrossEntropyLoss(args.aux, aux_weight=args.aux_weight)
        self.criterion = DataParallelCriterion(criterion, args.ctx, args.syncbn)
        # optimizer and lr scheduling
        self.lr_scheduler = LRScheduler(mode='poly', base_lr=args.lr,
                                        nepochs=args.epochs,
                                        iters_per_epoch=len(self.train_data),
                                        power=0.9)
        kv = mx.kv.create(args.kvstore)
        optimizer_params = {'lr_scheduler': self.lr_scheduler,
                            'wd':args.weight_decay,
                            'momentum': args.momentum,
                            'learning_rate': args.lr
                           }
        if args.dtype == 'float16':
            optimizer_params['multi_precision'] = True

        if args.no_wd:
            for k, v in self.net.module.collect_params('.*beta|.*gamma|.*bias').items():
                v.wd_mult = 0.0

        self.optimizer = gluon.Trainer(self.net.module.collect_params(), 'sgd',
                                       optimizer_params, kvstore = kv)
        # evaluation metrics
        self.metric = gluoncv.utils.metrics.SegmentationMetric(trainset.num_class)

    def training(self, epoch):
        tbar = tqdm(self.train_data)
        train_loss = 0.0
        alpha = 0.2
        for i, (data, target) in enumerate(tbar):
            with autograd.record(True):
                outputs = self.net(data.astype(args.dtype, copy=False))
                losses = self.criterion(outputs, target)
                mx.nd.waitall()
                autograd.backward(losses)
            self.optimizer.step(self.args.batch_size)
            for loss in losses:
                train_loss += np.mean(loss.asnumpy()) / len(losses)
            tbar.set_description('Epoch %d, training loss %.3f'%\
                (epoch, train_loss/(i+1)))
            mx.nd.waitall()


    def validation(self, epoch):
        #total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
        self.metric.reset()
        tbar = tqdm(self.eval_data)
        for i, (data, target) in enumerate(tbar):
            outputs = self.evaluator(data.astype(args.dtype, copy=False))
            outputs = [x[0] for x in outputs]
            targets = mx.gluon.utils.split_and_load(target, args.ctx, even_split=False)
            self.metric.update(targets, outputs)
            pixAcc, mIoU = self.metric.get()
            tbar.set_description('Epoch %d, validation pixAcc: %.3f, mIoU: %.3f'%\
                (epoch, pixAcc, mIoU))
            mx.nd.waitall()
        return pixAcc,mIoU
def save_checkpoint(net, args, best_mIoU,is_best=False):
    """Save Checkpoint"""
    directory = "runs/%s/" % (args.model)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename='%s_miou_%f.params'%(args.model,best_mIoU)
    filename = directory + filename
    if is_best:
        net.save_parameters(filename)


if __name__ == "__main__":
    args = parse_args()
    trainer = Trainer(args)
    if args.eval:
        print('Evaluating model: ', args.resume)
        trainer.validation(args.start_epoch)
    else:
        best_mIoU=0
        print('Starting Epoch:', args.start_epoch)
        print('Total Epochs:', args.epochs)
        for epoch in range(args.start_epoch, args.epochs):
            trainer.training(epoch)
            if not trainer.args.no_val:
                pixAcc,mIoU=trainer.validation(epoch)
                if mIoU > best_mIoU:
                    best_mIoU = mIoU
                    is_best = True
                    print("A new best! mIoU = %.4f" % mIoU)
                else:
                    is_best=False
                save_checkpoint(trainer.net.module,trainer.args,best_mIoU,is_best=is_best)
