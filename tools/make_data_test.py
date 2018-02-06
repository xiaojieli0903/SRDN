# -*- coding: utf-8 -*-
import os, json, argparse
from threading import Thread
from Queue import Queue

import numpy as np 
from scipy.misc import imread, imresize, imsave
from PIL import Image
from pylab import *
import h5py
import time
#import edge_detaction

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.signal as signal
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir',default='/mnt/lvmhdd1/dataset/ILSVRC/ILSVRC2015/Data/CLS-LOC/val')
parser.add_argument('--val_dir',default='/home/lixiaojie/Data/Set14/image_SRF_4')
parser.add_argument('--output_file',default='/mnt/lvm/xiaojie/imagenet-val-192_multi.h5')
parser.add_argument('--height',type=int,default=192)
parser.add_argument('--width',type=int,default=192)
parser.add_argument('--scale',type=int,default=4)
parser.add_argument('--max_images',type=int,default=-1)
parser.add_argument('--num_workers',type=int,default=1)
parser.add_argument('--include_val',type=int,default=1)
args = parser.parse_args()

def func(x,y,sigma=1):
		    return 100*(1/(2*np.pi*sigma))*np.exp(-((x-2)**2+(y-2)**2)/(2.0*sigma**2))

def imconv(image_array,suanzi):
    '''计算卷积
        参数
        image_array 原灰度图像矩阵
        suanzi      算子
        返回
        原图像与算子卷积后的结果矩阵
    '''
    image = image_array.copy()     # 原图像矩阵的深拷贝
    
    dim1,dim2 = image.shape

    # 对每个元素与算子进行乘积再求和(忽略最外圈边框像素)
    for i in range(1,dim1-1):
        for j in range(1,dim2-1):
            image[i,j] = (image_array[(i-1):(i+2),(j-1):(j+2)]*suanzi).sum()
    
    # 由于卷积后灰度值不一定在0-255之间，统一化成0-255
    image = image*(255.0/image.max())

    # 返回结果矩阵
    return image

# 生成标准差为5的5*5高斯算子,(test1.3,3,5)(test2.1,1,5)
suanzi1 = np.fromfunction(func,(1,1),sigma=5)

# x方向的Sobel算子
suanzi_x = np.array([[-1, 0, 1],
                    [ -2, 0, 2],
                    [ -1, 0, 1]])

# y方向的Sobel算子
suanzi_y = np.array([[-1,-2,-1],
                    [ 0, 0, 0],
                    [ 1, 2, 1]])

                                #train/val
def add_data(h5_file, image_dir, prefix, args):
	#print 'add_data'
    #制作image_list

	image_list = []
	image_extensions = {'.jpg','.jpeg','.JPG','.JPEG','.png','.PNG','.bmp'}
	for filename in os.listdir(image_dir):
		ext = os.path.splitext(filename)[1]
		if ext in image_extensions:
			image_list.append(os.path.join(image_dir,filename))
	if args.max_images > 0:
		num_images = args.max_images
	else:
		num_images = len(image_list)
	#制作数据集，一个数据集里有两个部分，data和label
	dset_data_name = prefix + '_data'
	dset_data_size = (num_images, 3, args.height / args.scale, args.width / args.scale)
	imgs_dset_data = h5_file.create_dataset(dset_data_name,dset_data_size,np.uint8)
                                            #train_data   ,
	dset_label_name = prefix + '_label'
	dset_label_size = (num_images, 3, args.height, args.width)
	imgs_dset_label = h5_file.create_dataset(dset_label_name,dset_label_size,np.uint8)

	dset_edge_name = prefix + '_edge'
	dset_edge_size = (num_images, 1, args.height, args.width)
	imgs_dset_edge = h5_file.create_dataset(dset_edge_name,dset_edge_size,np.uint8)


	input_queue = Queue()
	output_queue = Queue()
    #输入文件夹里的图片，预处理
	def read_worker():
		#print 'read_worker'
		while True:
			#idx,文件编号；filename，文件地址及名称
			idx, filename = input_queue.get()
			img = imread(filename)
			try:
				H, W = img.shape[0], img.shape[1]
				label = Image.fromarray(np.uint8(img))
				# scale the short edge to arg.width or arg.height
				if H <= W:
					if H < args.height:
						label = label.resize((W * args.height / H , args.height), Image.ANTIALIAS)
				else:
					if W < args.width:
						label = label.resize((args.width, H * args.width / W), Image.ANTIALIAS)
				# center crop
				W, H   = label.size
				left   = (W - args.width ) / 2
				top    = (H - args.height) / 2
				right  = (W + args.width ) / 2
				bottom = (H + args.height) / 2
				label = label.crop((left, top, right, bottom))
				#将图片均剪切到192*192
				data = label.resize((args.width / args.scale, args.height / args.scale),Image.ANTIALIAS)
				
				#制作edge权重矩阵 
				#label_new = Image.fromarray(label)
				label_gray = label.convert("L")
				gray_array = np.array(label_gray)

				# 利用生成的高斯算子与原图像进行卷积对图像进行平滑处理
				image_blur= signal.convolve2d(gray_array, suanzi1, mode="same")

				# 转化成图像矩阵
				image_array = np.array(image_blur)

				# 得到x方向矩阵
				image_x = signal.convolve2d(image_array,suanzi_x,mode="same")

				# 得到y方向矩阵
				image_y = signal.convolve2d(image_array,suanzi_y,mode="same")

				# 得到梯度矩阵
				image_xy = np.sqrt(image_x**2+image_y**2)
				# 梯度矩阵统一到0-255(1~2)
				edge = (5.0/image_xy.max())*image_xy+1.0
			
			except (ValueError, IndexError) as e:
				print filename
				print img.shape, img.dtype
				print e
			input_queue.task_done()
			output_queue.put((idx,np.asarray(data),np.asarray(label),np.asarray(edge)))
	#将读入的图片写道
	def write_work():
		#print 'write_work'
		num_written = 0
		tic = time.time()
		while True:
			idx, data, label, edge = output_queue.get()
			if label.ndim == 3:
				if label.shape[2] != 3:
					label = label[:,:,0:3]
					data = data[:,:,0:3]
				# RGB image, transpose from H x W x C to C x H x W
				imgs_dset_label[idx] = label.transpose(2, 0, 1)
				imgs_dset_data[idx] = data.transpose(2, 0, 1)
				imgs_dset_edge[idx] = edge
			elif label.ndim == 2:
				# Grayscale image; it is H x W so broadcasting to C x H x W will just copy
  		  		# grayscale values into all channels.
				imgs_dset_label[idx] = label
				imgs_dset_data[idx] = data
			output_queue.task_done()
			num_written = num_written + 1
			if num_written % 100 == 0:
				print 'Copied %d / %d images, times: %4fsec' % (num_written, num_images, time.time() - tic)
				tic = time.time()
              #两个工作任务

	for i in xrange(args.num_workers):
		t = Thread(target=read_worker)#双线程载入图片
		t.daemon = True
		t.start()

		t = Thread(target=write_work)#双线程载入图片
		t.daemon = True
		t.start()

                        #列举，枚举
	for idx, filename in enumerate(image_list):
		if args.max_images > 0 and idx >= args.max_images: break
		input_queue.put((idx, filename))
	input_queue.join()
	output_queue.join()




if __name__ == '__main__':
	print(args)
	with h5py.File(args.output_file,'w') as f:
		if args.include_val:
			add_data(f, args.val_dir, 'val', args)
			print('val set done!')
		add_data(f, args.train_dir, 'train', args)
		print('train set done!')
