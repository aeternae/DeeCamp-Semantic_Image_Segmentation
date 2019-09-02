from gluoncv.model_zoo import DeepLabV3

def deeplab_v3(*args,**kwargs):
    parameter=kwargs
    model=DeepLabV3(nclass=parameter.get('nclass'),backbone=parameter['backbone'],norm_layer=parameter.get('norm_layer'),
                          norm_kwargs=parameter.get('norm_kwargs'),
                          aux=parameter.get('aux'),crop_size=parameter.get('crop_size'))
    return model