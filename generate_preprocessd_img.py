import random
import os
import cv2
from PIL import Image
import numpy as np
from typing import Optional
import tqdm
import shutil
def adjust_img(img):
    cv_img=cv2.cvtColor(np.array(img),cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    cv_img_res = cv_img[y+20:y+h-20, x+20:x+w-20]
    return Image.fromarray(cv2.cvtColor(cv_img_res, cv2.COLOR_BGR2RGB))
def rotate_img(img,degree:Optional[int]=None):
    if degree is None:
        return img
    cv_img=cv2.cvtColor(np.array(img),cv2.COLOR_RGB2BGR)
    cv_img_res=cv2.rotate(cv_img,degree)
    return Image.fromarray(cv2.cvtColor(cv_img_res, cv2.COLOR_BGR2RGB))
def process_img_file(src_folder:str,dst_folder,img_file:str):
    ext_name=str(img_file.split(".")[-1]).lower()
    if ext_name=="jpg":
        ext_name="jpeg"
    file_name=".".join(img_file.split(".")[:-1])
    path_in=os.path.join(src_folder,img_file)
    img=Image.open(path_in)
    img.save(os.path.join(dst_folder,img_file),ext_name)
    img_adjusted=adjust_img(img)
    img_adjusted.save(os.path.join(dst_folder,file_name+"_修正."+ext_name),ext_name)
    img_adjusted_rotate_90=rotate_img(img_adjusted,cv2.ROTATE_90_CLOCKWISE)
    img_adjusted_rotate_90.save(os.path.join(dst_folder,file_name+"_修正_旋转90度."+ext_name),ext_name)
    img_adjusted_rotate_180=rotate_img(img_adjusted,cv2.ROTATE_180)
    img_adjusted_rotate_180.save(os.path.join(dst_folder,file_name+"_修正_旋转180度."+ext_name),ext_name)
    img_adjusted_rotate_270=rotate_img(img_adjusted,cv2.ROTATE_90_COUNTERCLOCKWISE)
    img_adjusted_rotate_270.save(os.path.join(dst_folder,file_name+"_修正_旋转270度."+ext_name),ext_name)
    
src_base="datasets/medical/imgs"
dst_base="new_imgs"
shutil.rmtree(dst_base)
os.mkdir(dst_base)
folders=os.listdir(src_base)
folders=random.sample(folders,50)
for folder in tqdm.tqdm(folders):
    src_folder=os.path.join(src_base,folder)
    dst_folder=os.path.join(dst_base,folder)
    os.mkdir(dst_folder)
    files=os.listdir(src_folder)
    for file in files:
        process_img_file(src_folder,dst_folder,file)