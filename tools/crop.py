# -*- coding: utf-8 -*-
from PIL import Image
import sys
import os.path
from  datetime import *
import random
import time


#im = Image.open('./images/1.png')  # 打开图片句柄

#box = (IMAGE_X1, IMAGE_Y1, IMAGE_X2, IMAGE_Y2)  # 设定裁剪区域

#region = im.crop(box)  # 裁剪图片，并获取句柄region

#region.save("11111.jpg")  # 保存图片
#region.show()


def ListFile(dir,wildcard,out):
    box = (80,0,180,100)
    exts = wildcard.split(" ")
    for ext in exts:    
        files = os.listdir(dir)
        for name in files:
            fullname=os.path.join(dir,name)
            im=Image.open(fullname)
            if name.endswith(ext):
                region = im.crop(box)
                outname=os.path.join(out,name)
                region.save(outname)
              
in_dir='/home/lixiaojie/torch-srgan/crop/2/'   
out_dir='./crop/1'     
wildcard_= ".jpg .png .bmp"
ListFile(in_dir,wildcard_,out_dir)

