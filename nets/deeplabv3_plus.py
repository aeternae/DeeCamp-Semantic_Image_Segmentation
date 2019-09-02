from gluoncv.model_zoo import DeepLabV3Plus

def deeplab_v3_plus(*args,**kwargs):
    parameter=kwargs
    model=DeepLabV3Plus(nclass=parameter.get('nclass'),backbone='xception',norm_layer=parameter.get('norm_layer'),
                          norm_kwargs=parameter.get('norm_kwargs'),
                          aux=parameter.get('aux'),crop_size=parameter.get('crop_size'))
    return model