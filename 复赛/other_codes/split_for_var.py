# -*- coding: utf-8 -*-

##### 分出验证集的部分，分为N部分，用于N折交叉验证
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
from  DataSet_list import *

'''
函数功能： 将训练集和验证集分开

参数：  original_dir存放袁术图片的地址
        split_num 如果涉及交叉验证，此处为交叉验证集的数目，默认为5
        if_only_var 如果只是单纯的分出验证集的比例，此处为True，如果是交叉验证方式此时为false
        percent  仅当if_only_var为True时该项有意义，表示分出验证集的比例，默认为0.3，不要小于0.2，避免过拟合·
无返回值
'''
def split_for_var(original_dir,split_num=5,if_only_var=False,percent=0.3):
    if(if_only_var):#仅分出验证集
        for i in LABELS:
            mkdir('../data/split_train_round2/'+str(i))
        for i in LABELS:
            mkdir('../data/split_Verification_round2/'+str(i))
        for dir0 in os.listdir(original_dir):
            image_list=[]
            for file in os.listdir(original_dir+'/'+dir0+"/"):  
                image_list.append(dir0+"/"+file)
            random.shuffle(image_list)
            train_image_list=list(image_list[0:int(len(image_list)*(1-percent))])    
            test_image_list=list(image_list[int(len(image_list)*(1-percent)):len(image_list)])
            for filename in train_image_list:
                shutil.copy(original_dir+'/'+filename,'../data/split_train_round2/'+filename)
            for filename in test_image_list:
                shutil.copy(original_dir+'/'+filename,'../data/split_Verification_round2/'+filename)
    else:#N折验证
        for j in range(split_num):
            for i in LABELS:
                mkdir('../data/split_round2_part'+str(j)+'/'+str(i))
        
        for dir0 in os.listdir(original_dir):   
            image_list=[]
            for file in os.listdir(original_dir+'/'+dir0+"/"):  
                image_list.append(dir0+"/"+file)
            random.shuffle(image_list)
            
            for j in range(split_num):
                if(j==(split_num-1)):
                    part=list(image_list[int(len(image_list)*(j/split_num)):len(image_list)])    
                else:
                    part=list(image_list[int(len(image_list)*(j/split_num)):int(len(image_list)*((j+1)/split_num))])    
                for filename in part:
                    shutil.copy(original_dir+'/'+filename,'../data/split_round2_part'+str(j)+'/'+filename)






if __name__=="__main__":
    split_for_var('../data/train_round2/',split_num=5,if_only_var=True,percent=0.3)