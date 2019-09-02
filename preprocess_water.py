import os
import shutil
import PIL
import tifffile
import  numpy as np

VAL_DATA=[17, 23, 26, 27, 33, 37, 53, 55, 56, 61, 80, 81, 83, 94, 96, 105, 109, 123, 125, 126, 128, 139, 146, 149, 150, 151, 157, 163, 165, 166, 174, 191, 192, 202, 206, 209, 212, 215, 224, 227, 235, 239, 248, 255, 288, 289, 297, 307, 309, 312, 314, 315, 327, 329, 330, 331, 333, 336, 338, 339, 341, 345, 356]



def rename(origin_dir):
    '''
    重命名文件
    '''
    for file in os.listdir(origin_dir):
        if file.endswith('tif'):
            if 'mask' not in file:
                file_path=os.path.join(origin_dir,file)
                dst_name=str(int(file.split('_')[0]))+'.tif'
                dst_path=os.path.join(origin_dir,dst_name)
                os.rename(file_path,dst_path)
            else:
                file_path=os.path.join(origin_dir,file)
                dst_name=str(int(file.split('_')[0]))+'_mask.tif'
                dst_path = os.path.join(origin_dir, dst_name)
                os.rename(file_path, dst_path)
def make_dataset(src_dir,dst_dir):
    train_dir=os.path.join(dst_dir,'train')
    val_dir=os.path.join(dst_dir,'val')
    train_img_dir=os.path.join(train_dir,'imgs')
    train_label_dir=os.path.join(train_dir,'labels')
    val_img_dir=os.path.join(val_dir,'imgs')
    val_label_dir=os.path.join(val_dir,'labels')

    for file in os.listdir(src_dir):
        file_path=os.path.join(src_dir,file)
        print(file_path)
        if file.endswith('.tif'):
            if 'mask' not in file:
                idx=int(file.replace('.tif',''))
                if idx in VAL_DATA:
                    shutil.copy(file_path,val_img_dir)
                else:
                    shutil.copy(file_path,train_img_dir)
            else:
                idx = int(file.replace('_mask.tif', ''))
                if idx in VAL_DATA:
                    shutil.copy(file_path, val_label_dir)
                else:
                    shutil.copy(file_path, train_label_dir)

def checkBoundry(x,size,right):
    x_left = x
    x_right = x + size
    if x + size > right:
        x_right = right
        x_left = right - size
    return x_left, x_right

def sequential_cut_single_valid(img_path,mask_path,stride=400,img_size=400,img_dir='D:\\downloads\\dataset_water\\data\\train\\imgs',mask_dir='D:\\downloads\\dataset_water\\data\\train\\labels'):
    file_name=os.path.split(img_path)[-1].replace('.tif','')
    img=tifffile.imread(img_path)
    mask=tifffile.imread(mask_path)
    h,w,_=img.shape
    count=1
    for i in range(0,h,img_size):
        for j in range(0,w,img_size):
            x1, x2 = checkBoundry(i,img_size,h)
            y1, y2 = checkBoundry(j,img_size,w)
            crop = img[x1:x2, y1:y2, :]
            mask_crop=mask[x1:x2, y1:y2]
            ch, cw,_ = crop.shape
            if ch == img_size and  cw == img_size:
                crop_file_name=os.path.join(img_dir,file_name+'_'+str(count)+'.tif')
                mask_crop_file_name=os.path.join(mask_dir,file_name+'_'+str(count)+'.tif')
                tifffile.imwrite(crop_file_name,crop,dtype=np.uint16)
                tifffile.imwrite(mask_crop_file_name,mask_crop)
                count+=1
            else:
                continue


def sequential_cut_single(img_path,mask_path,stride=200,img_size=400,img_dir='D:\\downloads\\dataset_water\\data\\train\\imgs',mask_dir='D:\\downloads\\dataset_water\\data\\train\\labels'):
    file_name=os.path.split(img_path)[-1].replace('.tif','')
    img=tifffile.imread(img_path)
    mask=tifffile.imread(mask_path)
    h,w,_=img.shape
    padding_h = (h // stride + 1) * stride
    padding_w = (w // stride + 1) * stride
    padding_img =np.zeros((padding_h, padding_w, 4),dtype=np.uint16)
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
                crop_file_name=os.path.join(img_dir,file_name+'_'+str(count)+'.tif')
                mask_crop_file_name=os.path.join(mask_dir,file_name+'_'+str(count)+'.tif')
                tifffile.imwrite(crop_file_name,crop,dtype=np.uint16)
                tifffile.imwrite(mask_crop_file_name,mask_crop)
                count+=1
            else:
                continue
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

def get_all_sequential_cut(img_list,mode):
    if mode=='train':
        for img_path in img_list:
            mask_path=img_path.replace('imgs','labels').replace('.tif', '_mask.tif')
            print('正在处理%s,%s' % (img_path,mask_path))
            sequential_cut_single(img_path,mask_path,stride=200,img_size=400,   img_dir='D:\\downloads\\dataset_water\\data\\%s\\imgs'%mode,mask_dir='D:\\downloads\\dataset_water\\data\\%s\\labels'%mode)
    elif mode=='val':
        for img_path in img_list:
            mask_path = img_path.replace('imgs', 'labels').replace('.tif', '_mask.tif')
            print('正在处理%s,%s' % (img_path, mask_path))
            sequential_cut_single_valid(img_path, mask_path, stride=400, img_size=400,
                                  img_dir='D:\\downloads\\dataset_water\\data\\%s\\imgs' % mode,
                                  mask_dir='D:\\downloads\\dataset_water\\data\\%s\\labels' % mode)








if __name__ == '__main__':
    # rename(origin_dir='D:\\downloads\\dataset_water\\origin')
    # make_dataset(src_dir='D:\\downloads\\dataset_water\\origin',dst_dir='D:\\downloads\\dataset_water\\dataset')
    # sequential_cut_single(img_path='D:\\downloads\\dataset_water\\dataset\\train\\1.tif',mask_path='D:\\downloads\\dataset_water\\dataset\\train\\1_mask.tif')
    mode='train'
    img_list,mask_list=get_origin_sets(img_folder_path='D:\\downloads\\dataset_water\\dataset\\%s\\imgs'%mode,mask_folder_path='D:\\downloads\\dataset_water\\dataset\\%s\\labels'%mode)
    get_all_sequential_cut(img_list=img_list,mode=mode)