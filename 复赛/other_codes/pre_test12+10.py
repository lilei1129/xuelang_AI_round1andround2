# -*- coding: utf-8 -*-




# coding: utf-8

import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets import base
import matplotlib.pyplot as plt # plt 用于显示图片
import datetime
import pandas as pd
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.applications.inception_v3 import preprocess_input
#from keras.models import *
#from keras.preprocessing.image import *
import os
import keras.backend as K
K.set_image_data_format('channels_last')
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def mkdir(path):     
	folder = os.path.exists(path)
 
	if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
		os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径
		print ("---  new folder...  ---")
		print ("---  OK  ---")
 
	else:
		print ("---  There is this folder!  ---")
             
def resize_img(dir,savepath,height, width):
    i=0
    for file in os.listdir(dir):  
        img=cv2.imread(dir+"/"+file)
        res = cv2.resize(img,(height,width), interpolation = cv2.INTER_AREA)
        cv2.imwrite(savepath+"/"+file,res)
        i=i+1

 
#model2 = load_model("./model/model0.hdf5")    
#file_1 = './data/test_new'
#mkdir(file_1)             
#resize_img(dir="./xuelang_round2_test_a_20180809",savepath="./data/test_new",height=499, width=499)

#Y=[]
#image_name=[]
#for file in os.listdir("./data/test_new"):
#    img = np.array(plt.imread('./data/test_new'+'/'+file))
#    x = image.img_to_array(img)  #将img由unit8变成float32
#    x = np.expand_dims(x, axis=0)
#    x/=255.
#    y=model2.predict(x)
#    Y.append(y)
#    image_name.append(file)
#    
#
        
#        


#def new_loss10(y_true,y_pred):
#    
#    y_pred /= tf.reduce_sum(y_pred,-1, True)
#    y_pred=tf.clip_by_value(y_pred,1e-10,1.0-1e-10)
##    print(y_true.shape)
#    #a_0=-(y_true[:,0] * tf.log(y_pred[:,0]) + (1-y_true[:,0]) * tf.log(1-y_pred[:,0]))
#    a_1=-(y_true[:,1] * tf.log(y_pred[:,1]) + (1-y_true[:,1]) * tf.log(1-y_pred[:,1]))
#    a_2=-(y_true[:,2] * tf.log(y_pred[:,2]) + (1-y_true[:,2]) * tf.log(1-y_pred[:,2]))
#    a_3=-(y_true[:,3] * tf.log(y_pred[:,3]) + (1-y_true[:,3]) * tf.log(1-y_pred[:,3]))
#    a_4=-(y_true[:,4] * tf.log(y_pred[:,4]) + (1-y_true[:,4]) * tf.log(1-y_pred[:,4]))
#    a_5=-(y_true[:,5] * tf.log(y_pred[:,5]) + (1-y_true[:,5]) * tf.log(1-y_pred[:,5]))
#    a_6=-(y_true[:,6] * tf.log(y_pred[:,6]) + (1-y_true[:,6]) * tf.log(1-y_pred[:,6]))
#    a_7=-(y_true[:,7] * tf.log(y_pred[:,7]) + (1-y_true[:,7]) * tf.log(1-y_pred[:,7]))
#    a_8=-(y_true[:,8] * tf.log(y_pred[:,8]) + (1-y_true[:,8]) * tf.log(1-y_pred[:,8]))
#    a_9=-(y_true[:,9] * tf.log(y_pred[:,9]) + (1-y_true[:,9]) * tf.log(1-y_pred[:,9]))
#    a_10=-(y_true[:,10] * tf.log(y_pred[:,10]) + (1-y_true[:,10]) * tf.log(1-y_pred[:,10]))
#    
#    haha= 154/3331*a_1 + 142/3331*a_2 + 313/3331*a_3 + 179/3331*a_4 + 195/3331*a_5 +339/3331*a_6 +163/3331*a_7 + 210/3331*a_8 + 141/3331*a_9 + 332/3331*a_10
#    newloss=tf.reduce_mean(haha)
#    return newloss

#model10= load_model("./Xception_models_2/model.hdf5")
##
df = pd.DataFrame(columns=('filename|defect', 'probability'))#生成空的pandas表
LABELS = [
        'norm','defect_1','defect_2','defect_3','defect_4','defect_5','defect_6','defect_7','defect_8','defect_9','defect_10']
number=0


# examples using a.item()
type(np.float32(0).item()) # <type 'float'>
type(np.float64(0).item()) # <type 'float'>
j=-1
Y=[]
name1=[]
for line in open("submit_20180824_143001.csv"):
    if j==-1:#删除文件头
        file,a = line.split(",")
        a = a.strip(' \t\r\n') 
        file = file.strip(' \t\r\n')
        j=0
    else:
        file,a = line.split(",")
        a = float(a.strip(' \t\r\n'))
        file = file.strip(' \t\r\n')
   
        Y.append(a)   
        name1.append(file)
        j=j+1
#df.to_csv(("submit_"+datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".csv"), index=False,sep=',')
j=-1
Y2=[]
name2=[]
for line in open("submit_20180829_115556.csv"):
    if j%11==0:#删除文件头
        file,a = line.split(",")
        a = float(a.strip(' \t\r\n'))
        file = file.strip(' \t\r\n')           
        Y2.append(a)
        name2.append(file)
    else:
        file,a = line.split(",")
        a = a.strip(' \t\r\n')
        file = file.strip(' \t\r\n')   
    j=j+1
    
    
error_file=[]    
count=0
for i in range(len(Y)):    
    if Y[i]<0.5 and Y2[i]>0.5:
            a=0
    elif Y[i]>0.5 and Y2[i]<0.5:
            a=0
    else:
        print(name1[i])
        print(name2[i])
        print(Y[i])
        print(Y2[i])
        error_file.append(name2[i])
        count=count+1
  
