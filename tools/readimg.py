# -*- coding: utf-8 -*-
#!usr/bin/env python
import os, json, argparse
from threading import Thread
from Queue import Queue

import numpy as np 
from scipy.misc import imread, imresize, imsave,imshow
from PIL import Image
import h5py
import time
import re
import sys
import subprocess
import os.path
import os  

height=192
width=192
scale=4
wildcard = ".JPEG"
dir='/mnt/lvmhdd1/dataset/ILSVRC/ILSVRC2015/Data/CLS-LOC/val/'    
print dir
i=0
exts = wildcard.split(" ")
for ext in exts:    
	files = os.listdir(dir)
	for name in files:
		fullname=os.path.join(dir,name)
		if i<=20:
			if name.endswith(ext):
				img = imread(fullname)
				H, W = img.shape[0], img.shape[1]
				label = Image.fromarray(np.uint8(img))
				# scale the short edge to arg.width or arg.height
				if H <= W:
					if H < height:
						label = label.resize((W * height / H , height), Image.ANTIALIAS)
				else:
					if W < width:
						label = label.resize((width, H * width / W), Image.ANTIALIAS)
				# center crop
				W, H   = label.size
				left   = (W - width ) / 2
				top    = (H - height) / 2
				right  = (W + width ) / 2
				bottom = (H + height) / 2
				label = label.crop((left, top, right, bottom))

				data = label.resize((width / scale, height / scale),Image.ANTIALIAS)
				#imshow(label)
				name_label="./imagenet_img/label"
				name_data="./imagenet_img/data"
				name_data_=name_data+str(i)+".png"
				name_label_=name_label+str(i)+".png"
				imsave(name_label_,label)
				imsave(name_data_,data)
			i=i+1











