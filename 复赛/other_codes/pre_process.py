# -*- coding: utf-8 -*-


import os
import datetime
from PIL import Image
import random
import tensorflow as tf
import numpy as np
import pandas as pd
import pandas as pd
import keras
import matplotlib.pyplot as plt
import scipy.misc
import cv2
import xml.etree.cElementTree as et
import shutil

def preprocess_for_train(image, height, width, bbox):
    # 查看是否存在标注框。
    if bbox is None:
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        
    # 随机的截取图片中一个块。
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
        tf.shape(image), bounding_boxes=bbox, min_object_covered=1)
    #bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
        #tf.shape(image), bounding_boxes=bbox, min_object_covered=0.4)
    distorted_image = tf.slice(image, bbox_begin, bbox_size)

    # 将随机截取的图片调整为神经网络输入层的大小。
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    distorted_image = tf.image.random_flip_up_down(distorted_image)
    distorted_image = distort_color(distorted_image, np.random.randint(2))
    scipy.misc.imsave('../data/'+'zengqiang_xc.jpeg', distorted_image.eval())
    img=cv_imread('../data/'+'zengqiang_xc.jpeg')
    res = cv2.resize(img,(height,width), interpolation = cv2.INTER_AREA)
    return res
def crop_img_xc(dir,target_size,num,savepath,i):
    with tf.Session() as sess:
        for dir2 in os.listdir(dir):  
            for file in os.listdir(dir+dir2+"/"):  
                if('xml'in file):
                    s=dir+dir2+"/"+file
                    tree=et.parse(s)
                    root=tree.getroot()
                    filename=root.find('filename').text
                    for Object in root.findall('object'):
                        bndbox=Object.find('bndbox')
                        xmin=bndbox.find('xmin').text
                        ymin=bndbox.find('ymin').text
                        xmax=bndbox.find('xmax').text
                        ymax=bndbox.find('ymax').text
                        image_raw_data = tf.gfile.FastGFile(dir+dir2+"/"+filename, 'rb').read()
                        img_data = tf.image.decode_jpeg(image_raw_data)       
                        img_data.set_shape([1920,2560,3])
                        a=img_data.get_shape()
                        box=tf.constant([[[int(ymin)/int(a[0]),int(xmin)/int(a[1]),int(ymax)/int(a[0]),int(xmax)/int(a[1])]]], dtype=tf.float32)
                        for j in range(num):
                             res = preprocess_for_train(img_data, target_size, target_size, box)
                             cv2.imwrite(savepath+"/zengqiang_xc_"+str(i)+".jpg",res)
                             i=i+1
                        img=cv_imread(dir+dir2+"/"+filename)
                        res2 = cv2.resize(img,(target_size,target_size), interpolation = cv2.INTER_AREA)
                        cv2.imwrite(savepath+"/zengqiang_xc_"+str(i)+".jpg",res2)
                        i=i+1     
    return i
def crop_img_zc(dir,target_size,num,savepath,i):
    with tf.Session() as sess:
        for file in os.listdir(dir):
            filename=file
            image_raw_data = tf.gfile.FastGFile(dir+filename, 'rb').read()
            img_data = tf.image.decode_jpeg(image_raw_data)
            img_data.set_shape([1920,2560,3])
            a=img_data.get_shape()
            box=None
            for j in range(num):
                res = preprocess_for_train(img_data, target_size, target_size, box)
                cv2.imwrite(savepath+"/zengqiang_zc_"+str(i)+".jpg",res)
                i=i+1
            img=cv_imread(dir+filename)
            res2 = cv2.resize(img,(target_size,target_size), interpolation = cv2.INTER_AREA)
            cv2.imwrite(savepath+"/zengqiang_zc_"+str(i)+".jpg",res2)
            i=i+1
    return i
            
def mkdir(path):
 
	folder = os.path.exists(path)
 
	if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
		os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径
		print ("---  new folder...  ---")
		print ("---  OK  ---")
 
	else:
		print ("---  There is this folder!  ---")
        
  
def resize_img_test(dir,savepath,height, width):
    i=0
    for file in os.listdir(dir):  
        img=cv2.imread(dir+"/"+file)
        res = cv2.resize(img,(height,width), interpolation = cv2.INTER_AREA)
        cv2.imwrite(savepath+"/"+file,res)
        i=i+1
def cv_imread(file_path):
    cv_img=cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),-1) 
    return cv_img   
    
def distort_color(image, color_ordering):
    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32./255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    else:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32./255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
    return tf.clip_by_value(image, 0.0, 1.0)
 
def get_files(file_dir):
    zc = []
    label_zc = [] 
    xc = []
    label_xc = []
    
    for file in os.listdir(file_dir+'/label_0'):
        zc.append(file_dir +'/label_0'+'/'+ file)
        label_zc.append(0)
    for file in os.listdir(file_dir+'/label_1'):
        xc.append(file_dir +'/label_1'+'/'+file)
        label_xc.append(1)
#把cat和dog合起来组成一个list（img和lab）
    image_list = np.hstack((zc, xc))
    label_list = np.hstack((label_zc, label_xc))
 
#利用shuffle打乱顺序
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
 
#从打乱的temp中再取出list（img和lab）
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]                                                   
    return image_list,label_list
#返回两个list 分别为图片文件名及其标签  顺序已被打乱
    ### ----讲测试集转为训练集
def copy_dir(dir1,dir2):
    if os.path.exists(dir2):
        try:
            shutil.rmtree(dir2)
        except Exception as ex:
            print("错误信息："+str(ex))#提示：错误信息，目录不是空的
    shutil.copytree(dir1,dir2)
            