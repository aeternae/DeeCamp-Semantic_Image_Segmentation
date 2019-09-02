from gluoncv.model_zoo import PSPNet


def pspnet(*args,**kwargs):
    parameter=kwargs
    model=PSPNet(nclass=parameter.get('nclass'),backbone=parameter['backbone'],norm_layer=parameter.get('norm_layer'),
                          norm_kwargs=parameter.get('norm_kwargs'),
                          aux=parameter.get('aux'),crop_size=parameter.get('crop_size'))
    return model