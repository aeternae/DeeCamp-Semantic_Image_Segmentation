import os
from mxnet.gluon.data import dataset
from gluoncv.data.segbase import SegmentationDataset
import mxnet as mx
import numpy as np
from mxnet import cpu
import mxnet.ndarray as F
from PIL import Image
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
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        return img, mask
    def __len__(self):
        return len(self.images)
    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32')
        target[target == 255] = -1
        return F.array(target, cpu(0))
    @property
    def classes(self):
        """Category names."""
        return ('background','land')