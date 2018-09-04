
import matplotlib.pyplot as plt # plt 用于显示图片
import datetime
import tensorflow as tf
import numpy as np
import pandas as pd
from keras.preprocessing import image
from keras.models import Model, load_model
import numpy as np
#from keras.models import *
#from keras.preprocessing.image import *
import os
import keras.backend as K
K.set_image_data_format('channels_last')
import cv2

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

    
file_1 = "../data/test_new"
mkdir(file_1)             
resize_img(dir="../data/xuelang_round1_test_b",savepath="../data/test_new",height=499, width=499)

model = load_model("./model0.hdf5")

file_dir="../data/test_new"
fn=[]
y_label=[]


for file in os.listdir(file_dir):
    '''获取该路径文件下的所有图片'''    
    #file是文件夹的名字
    src = os.path.join(os.path.abspath(file_dir), file)         
    simg = src
    fn.append(file)

    img = np.array(plt.imread(simg))
    x = image.img_to_array(img)  #将img由unit8变成float32
    x = np.expand_dims(x, axis=0)
    x/=255.
    y=model.predict(x)
    if y == 1:
        y = 0.999999   
    elif y == 0:
        y = 0.000001
        
    y=float('%.6f' % y)
    y_label.append(y)
    print(y)
dataframe = pd.DataFrame({'filename':fn,'probability':y_label})
#dataframe.to_csv("test_VGG13_models.csv",index=False,sep=',')
dataframe.to_csv(("../submit/submit_"+datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".csv"), index=False,sep=',')



