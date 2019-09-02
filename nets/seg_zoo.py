from .deeplabv3 import *
from .pspnet import *
from .deeplabv3_plus import *

_models = {
    'deeplab_v3': deeplab_v3,
    'pspnet':pspnet,
    'deeplab_v3_plus':deeplab_v3_plus
}


def get_model(name, **kwargs):
    name = name.lower()
    if name not in _models:
        err_str = '"%s" is not among the following model list:\n\t' % (name)
        err_str += '%s' % ('\n\t'.join(sorted(_models.keys())))
        raise ValueError(err_str)
    net = _models[name](*kwargs,**kwargs)
    return net


def get_model_list():
    return _models.keys()
