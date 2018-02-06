# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.signal as signal
import glob
import time
from scipy.misc import imread, imresize, imsave,info
import math
import os  
import re  


def edge(img_path):
	# 生成高斯算子的函数
	def func(x,y,sigma=1):
	    return 100*(1/(2*np.pi*sigma))*np.exp(-((x-2)**2+(y-2)**2)/(2.0*sigma**2))

	# 生成标准差为5的5*5高斯算子
	suanzi = np.fromfunction(func,(1,1),sigma=5)

	'''
	# 打开图像并转化成灰度图像
	image = Image.open("img_005_SRF_4_HR.png").convert("L")
	image_array = np.array(image)
	'''
	img = imread(img_path)
	label= Image.fromarray(np.uint8(img))
	label_gray = label.convert("L")
	gray_array = np.array(label_gray)


	# 利用生成的高斯算子与原图像进行卷积对图像进行平滑处理
	image_blur = signal.convolve2d(gray_array, suanzi, mode="same")


	# x方向的Sobel算子
	suanzi_x = np.array([[-1, 0, 1],
	                    [ -2, 0, 2],
	                    [ -1, 0, 1]])

	# y方向的Sobel算子
	suanzi_y = np.array([[-1,-2,-1],
	                     [ 0, 0, 0],
	                     [ 1, 2, 1]])



	# 转化成图像矩阵
	image_array = np.array(gray_array)
	    
	# 得到x方向矩阵
	image_x = signal.convolve2d(image_array,suanzi_x,mode="same")

	# 得到y方向矩阵
	image_y = signal.convolve2d(image_array,suanzi_y,mode="same")

	# 得到梯度矩阵
	image_xy = np.sqrt(image_x**2+image_y**2)
	#print info(image_xy)

	#print image_xy.shape
	h= image_xy.shape[0]
	w= image_xy.shape[1]


	# 梯度矩阵统一到0-255(1~2) test1((0-1)+1),test2((0-3),+1)
	image_xy = (255.0/image_xy.max())*image_xy
	#print image_xy.max()


	for i in range(h):
		for j in range(w):
			if image_xy[i][j]<50:
				image_xy[i][j]=0
	
	return image_xy


'''
	# 绘出图像
	plt.subplot(2,2,1)
	plt.imshow(image_array,cmap=cm.gray)
	plt.axis("off")
	plt.subplot(2,2,2)
	plt.imshow(image_x,cmap=cm.gray)
	plt.axis("off")
	plt.subplot(2,2,3)
	plt.imshow(image_y,cmap=cm.gray)
	plt.axis("off")
	plt.subplot(2,2,4)
	plt.imshow(image_xy,cmap=cm.gray)
	plt.axis("off")
	plt.show()

	imsave('label1.png',image_xy)
	plt.subplot(1,2,1)
	plt.imshow(image_array,cmap=cm.gray)
	plt.axis("off")

	plt.subplot(1,2,2)
	plt.imshow(image_xy,cmap=cm.gray)
	plt.axis("off")
	plt.show()

def HFE(img1,img2):
	H=img1.shape[0]
	W=img1.shape[1]
	loss=0
	for i in range(H):
		for j in range(W):
				loss=loss+(img1[i][j]-img2[i][j])*(img1[i][j]-img2[i][j])
				HFE=20 * math.log10(255/ math.sqrt(loss))
				return HFE

def ListFile(dir,wildcard,ref):
    exts = wildcard.split(" ")
    for ext in exts:    
        files = os.listdir(dir)
        for name in files:
            fullname=os.path.join(dir,name)
            if name.endswith(ext):
                HR_edge=edge(ref)
                SR_edge=edge(fullname)
                name_crop=fullname+'_crop.png'
                imsave(name_crop,SR_edge)
                print name
                print HFE(HR_edge,SR_edge)


dir_='/home/lixiaojie/torch-srgan/crop/baboon/'        
ref_file = '/mnt/lvm/xiaojie/SelfExSR/data/Set14/image_SRF_4/img_001_SRF_4_HR.png'
ref_= imread(ref_file, flatten=True).astype(np.float32)
wildcard = ".png .jpg"
print dir_
ListFile(dir_,wildcard,ref_file)      
'''

path1="/home/lixiaojie/Data/Set14/image_SRF_4/img_001_SRF_4_HR.png"
path2="/home/lixiaojie/DRCN/DRCN_Set14_x4/baboon.png"
path3="/home/lixiaojie/torch-srgan/good/baboon_SRCNN.png"
path4="/home/lixiaojie/torch-srgan/good/densenet_fft_76baboon_24.322102834949_0.48847780556677.png"
path5="/home/lixiaojie/torch-srgan/11/fft.png"
path6="/home/lixiaojie/torch-srgan/11/nofft.png"
path7="/home/lixiaojie/torch-srgan/good/densenet_fft1-8_76baboon_24.322102834949_0.48847780556677.png"

HR_edge=edge(path1)
SR_edge1=edge(path2)
SR_edge2=edge(path3)
SR_edge3=edge(path4)
SR_edge4=edge(path5)
SR_edge5=edge(path6)
SR_edge6=edge(path7)

def HFE(img1,img2):
	H=img1.shape[0]
	W=img1.shape[1]
	loss=0.0
	for i in range(H):
		for j in range(W):
				loss=loss+(float(img1[i][j])-float(img2[i][j]))**2
	#print loss
	HFE=20.0 * math.log10(255.0/ math.sqrt(loss))
	return HFE
	


print HFE(HR_edge,SR_edge1)
print HFE(HR_edge,SR_edge2)
print HFE(HR_edge,SR_edge3)
print HFE(HR_edge,SR_edge4)
print HFE(HR_edge,SR_edge5)
print HFE(HR_edge,SR_edge6)

plt.subplot(1,7,1)
plt.imshow(HR_edge,cmap=cm.gray)
plt.axis("off")

plt.subplot(1,7,2)
plt.imshow(SR_edge1,cmap=cm.gray)
plt.axis("off")

plt.subplot(1,7,3)
plt.imshow(SR_edge2,cmap=cm.gray)
plt.axis("off")

plt.subplot(1,7,4)
plt.imshow(SR_edge3,cmap=cm.gray)
plt.axis("off")

plt.subplot(1,7,5)
plt.imshow(SR_edge4,cmap=cm.gray)
plt.axis("off")

plt.subplot(1,7,6)
plt.imshow(SR_edge5,cmap=cm.gray)
plt.axis("off")

plt.subplot(1,7,7)
plt.imshow(SR_edge6,cmap=cm.gray)
plt.axis("off")


plt.show()


#print HR_edge.shape
