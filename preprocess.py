import cv2
import tifffile
import numpy as np
import os
import  matplotlib.pyplot as plt
from PIL import Image
def get_max_min(path):
    this_img=tifffile.imread(path)
    this_img_min=np.min(this_img)
    this_img_max=np.max(this_img)
    return this_img_min,this_img_max

def tif_to_img(path):
    min=278
    max=1615
    this_img = tifffile.imread(path)
    this_img=np.array(this_img,dtype='float32')
    this_img=(this_img-min)/max*255.
    this_img[this_img<0]=0
    this_img[this_img>255]=255
    this_img=this_img.astype('uint8')
    return this_img
def mask_to_img(path):
    this_img = tifffile.imread(path)
    this_img=np.array(this_img,dtype='uint8')
    return this_img




def get_all_dataset_min_max(root_path):
    min_list=[]
    max_list=[]
    for root,dirs,files in os.walk(root_path):
        for file in  files:
            if file.endswith('.tif') and 'mask' not in file:
                file_path=os.path.join(root,file)
                print(file_path)
                this_img_min,this_img_max=get_max_min(file_path)
                min_list.append(this_img_min)
                max_list.append(this_img_max)
    print(min(min_list),max(max_list))


def get_all_tiff_to_img(root_path,img_folder_path,mask_folder_path):
    for root,dirs,files in os.walk(root_path):
        for file in  files:
            file_path = os.path.join(root, file)
            if file.endswith('.tif') :
                file=file.replace('.tif','.png')
                if 'mask' not in file:
                    img_path=os.path.join(img_folder_path,file)
                    this_img=tif_to_img(file_path)
                    this_img = Image.fromarray(this_img.astype('uint8')).convert('RGB')
                    this_img.save(img_path)
                else:
                    mask_path = os.path.join(mask_folder_path, file)
                    this_img=mask_to_img(file_path)
                    print(this_img.shape)
                    this_img = Image.fromarray(this_img.astype('uint8'))
                    this_img.save(mask_path)
def get_origin_sets(img_folder_path,mask_folder_path):
    img_list=[]
    mask_list=[]
    for root,dirs,files in os.walk(img_folder_path):
        for file in  files:
            file_path = os.path.join(root, file)
            img_list.append(file_path)
    for root,dirs,files in os.walk(mask_folder_path):
        for file in  files:
            file_path = os.path.join(root, file)
            mask_list.append(file_path)
    return img_list,mask_list
def sequential_cut_single(img_path,mask_path,stride=256,img_size=512,img_dir='D:\\downloads\\satellite\\train\\imgs',mask_dir='D:\\downloads\\satellite\\train\\labels'):
    file_name=os.path.split(img_path)[-1].replace('.png','')
    img=np.array(Image.open(img_path).convert('RGB'))
    mask=np.array(Image.open(mask_path))
    h,w,_=img.shape
    padding_h = (h // stride + 1) * stride
    padding_w = (w // stride + 1) * stride
    padding_img =np.zeros((padding_h, padding_w, 3), dtype=np.uint8)
    padding_img[0:h, 0:w, :] = img[:, :, :]
    padding_mask=np.zeros((padding_h, padding_w), dtype=np.uint8)
    padding_mask[0:h, 0:w] = mask[:, :]
    count=1
    for i in range(w // stride):
        for j in range(h // stride):
            crop = padding_img[i * stride:i * stride + img_size, j * stride:j * stride + img_size,:]
            mask_crop=padding_mask[i * stride:i * stride + img_size, j * stride:j * stride + img_size]
            ch, cw,_ = crop.shape
            if ch == img_size and  cw == img_size:
                crop_file_name=os.path.join(img_dir,file_name+'_'+str(count)+'.png')
                mask_crop_file_name=os.path.join(mask_dir,file_name+'_'+str(count)+'.png')
                crop=Image.fromarray(crop.astype('uint8')).convert('RGB')
                crop.save(crop_file_name)
                mask_crop=Image.fromarray(mask_crop.astype('uint8'))
                mask_crop.save(mask_crop_file_name)
                count+=1
            else:
                continue
def get_all_sequential_cut(img_list,mode):
    for img_path in img_list:
        mask_path=img_path.replace('ImageSets','SegmentationClass').replace('.png', '_mask.png')
        print('正在处理%s,%s' % (img_path,mask_path))
        sequential_cut_single(img_path,mask_path,stride=256,img_size=512,img_dir='D:\\downloads\\satellite\\%s\\imgs'%mode,mask_dir='D:\\downloads\\satellite\\%s\\labels'%mode)




if __name__ == '__main__':
    mode='val'
    # get_all_tiff_to_img(root_path='D:\\downloads\\deecamp_seg',img_folder_path='D:\\downloads\\deeecamp_dataset\\train\\ImageSets',mask_folder_path='D:\\downloads\\deeecamp_dataset\\train\\SegmentationClass')
    img_list,mask_list=get_origin_sets(img_folder_path='D:\\downloads\\deeecamp_dataset\\%s\\ImageSets'%mode,mask_folder_path='D:\\downloads\\deeecamp_dataset\\%s\\SegmentationClass'%mode)
    get_all_sequential_cut(img_list=img_list,mode=mode)
