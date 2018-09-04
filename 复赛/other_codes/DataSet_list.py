# -*- coding: utf-8 -*-

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
import xml.etree.cElementTree as et
import shutil
from pre_process import *
##函数功能 读取训练数集，并且生成img 及其label的对应
LABELS = [
        'norm','defect_1','defect_2','defect_3','defect_4','defect_5','defect_6','defect_7','defect_8','defect_9','defect_10']

def get_files_for_defects(root_dir):
    xml_list=[]
    image_list=[]
    label_list=[]
    for dir2 in root_dir:  
        for dir3 in os.listdir(dir2):
            if dir3=='正常':
                for file in os.listdir(dir2+'/'+dir3):
                    if ('jpg' in file):
                        image_list.append(dir2+'/'+dir3+'/'+file)
                        label_list.append(0)
                        xml_list.append(None)
            elif dir3=='扎洞':
                for file in os.listdir(dir2+'/'+dir3):
                    if ('jpg' in file):
                        image_list.append(dir2+'/'+dir3+'/'+file)
                        label_list.append(1)
                        xml_list.append(dir2+'/'+dir3+'/'+file.replace('jpg','xml'))
            elif dir3=='毛斑':
                for file in os.listdir(dir2+'/'+dir3):
                    if ('jpg' in file):
                        image_list.append(dir2+'/'+dir3+'/'+file)
                        label_list.append(2)
                        a=file.split('.')
                        xml_list.append(dir2+'/'+dir3+'/'+file.replace('jpg','xml'))
            elif dir3=='擦洞':
                for file in os.listdir(dir2+'/'+dir3):
                    if ('jpg' in file):
                        image_list.append(dir2+'/'+dir3+'/'+file)
                        label_list.append(3)
                        a=file.split('.')
                        xml_list.append(dir2+'/'+dir3+'/'+file.replace('jpg','xml'))
            elif dir3=='毛洞':
                for file in os.listdir(dir2+'/'+dir3):
                    if ('jpg' in file):
                        image_list.append(dir2+'/'+dir3+'/'+file)
                        label_list.append(4)
                        a=file.split('.')
                        xml_list.append(dir2+'/'+dir3+'/'+file.replace('jpg','xml'))
            elif dir3=='织稀':
                for file in os.listdir(dir2+'/'+dir3):
                    if ('jpg' in file):
                        image_list.append(dir2+'/'+dir3+'/'+file)
                        label_list.append(5)
                        a=file.split('.')
                        xml_list.append(dir2+'/'+dir3+'/'+file.replace('jpg','xml'))
            elif dir3=='吊经':
                for file in os.listdir(dir2+'/'+dir3):
                    if ('jpg' in file):
                        image_list.append(dir2+'/'+dir3+'/'+file)
                        label_list.append(6)
                        a=file.split('.')
                        xml_list.append(dir2+'/'+dir3+'/'+file.replace('jpg','xml'))
            elif dir3=='缺经':
                for file in os.listdir(dir2+'/'+dir3):
                    if ('jpg' in file):
                        image_list.append(dir2+'/'+dir3+'/'+file)
                        label_list.append(7)
                        a=file.split('.')
                        xml_list.append(dir2+'/'+dir3+'/'+file.replace('jpg','xml'))
            elif dir3=='跳花':
                for file in os.listdir(dir2+'/'+dir3):
                    if ('jpg' in file):
                        image_list.append(dir2+'/'+dir3+'/'+file)
                        label_list.append(8)
                        a=file.split('.')
                        xml_list.append(dir2+'/'+dir3+'/'+file.replace('jpg','xml'))
            elif dir3=='油渍' or dir3=='污渍':
                for file in os.listdir(dir2+'/'+dir3):
                    if ('jpg' in file):
                        image_list.append(dir2+'/'+dir3+'/'+file)
                        label_list.append(9)
                        a=file.split('.')
                        xml_list.append(dir2+'/'+dir3+'/'+file.replace('jpg','xml'))
            else:
                for file in os.listdir(dir2+'/'+dir3):
                    if ('jpg' in file):
                        image_list.append(dir2+'/'+dir3+'/'+file)
                        label_list.append(10)
                        a=file.split('.')
                        xml_list.append(dir2+'/'+dir3+'/'+file.replace('jpg','xml'))

    return image_list,label_list,xml_list

def get_files_for_defects_test(answer_dir,img_dir,image_list,label_list,xml_list):
    img_dir2= "../data/img_dir2"
    copy_dir(img_dir,img_dir2)
    for dir2 in os.listdir(answer_dir):  
        if dir2=='扎洞':
            for file in os.listdir(answer_dir+'/'+dir2):
                if ('xml' in file):
                    xml_list.append(answer_dir+'/'+dir2+'/'+file)
                    label_list.append(1)
                    a=file.split('.')
                    image_list.append(img_dir+'/'+file.replace('xml','jpg'))
                    if os.path.exists(img_dir2+'/'+file.replace('xml','jpg')):
                        os.remove(img_dir2+'/'+file.replace('xml','jpg'))
        elif dir2=='毛斑':
            for file in os.listdir(answer_dir+'/'+dir2):
                if ('xml' in file):
                    xml_list.append(answer_dir+'/'+dir2+'/'+file)
                    label_list.append(2)
                    a=file.split('.')
                    image_list.append(img_dir+'/'+file.replace('xml','jpg'))
                    if os.path.exists(img_dir2+'/'+file.replace('xml','jpg')):
                        os.remove(img_dir2+'/'+file.replace('xml','jpg'))

        elif dir2=='擦洞':
            for file in os.listdir(answer_dir+'/'+dir2):
                if ('xml' in file):
                    xml_list.append(answer_dir+'/'+dir2+'/'+file)
                    label_list.append(3)
                    a=file.split('.')
                    image_list.append(img_dir+'/'+file.replace('xml','jpg'))
                    if os.path.exists(img_dir2+'/'+file.replace('xml','jpg')):
                        os.remove(img_dir2+'/'+file.replace('xml','jpg'))
        elif dir2=='毛洞':
            for file in os.listdir(answer_dir+'/'+dir2):
                if ('xml' in file):
                    xml_list.append(answer_dir+'/'+dir2+'/'+file)
                    label_list.append(4)
                    a=file.split('.')
                    image_list.append(img_dir+'/'+file.replace('xml','jpg'))
                    if os.path.exists(img_dir2+'/'+file.replace('xml','jpg')):
                        os.remove(img_dir2+'/'+file.replace('xml','jpg'))
        elif dir2=='织稀':
            for file in os.listdir(answer_dir+'/'+dir2):
                if ('xml' in file):
                    xml_list.append(answer_dir+'/'+dir2+'/'+file)
                    label_list.append(5)
                    a=file.split('.')
                    image_list.append(img_dir+'/'+file.replace('xml','jpg'))
                    if os.path.exists(img_dir2+'/'+file.replace('xml','jpg')):
                        os.remove(img_dir2+'/'+file.replace('xml','jpg'))
        elif dir2=='吊经':
            for file in os.listdir(answer_dir+'/'+dir2):
                if ('xml' in file):
                    xml_list.append(answer_dir+'/'+dir2+'/'+file)
                    label_list.append(6)
                    a=file.split('.')
                    image_list.append(img_dir+'/'+file.replace('xml','jpg'))
                    if os.path.exists(img_dir2+'/'+file.replace('xml','jpg')):
                        os.remove(img_dir2+'/'+file.replace('xml','jpg'))
        elif dir2=='缺经':
            for file in os.listdir(answer_dir+'/'+dir2):
                if ('xml' in file):
                    xml_list.append(answer_dir+'/'+dir2+'/'+file)
                    label_list.append(7)
                    a=file.split('.')
                    image_list.append(img_dir+'/'+file.replace('xml','jpg'))
                    if os.path.exists(img_dir2+'/'+file.replace('xml','jpg')):
                        os.remove(img_dir2+'/'+file.replace('xml','jpg'))
        elif dir2=='跳花':
            for file in os.listdir(answer_dir+'/'+dir2):
                if ('xml' in file):
                    xml_list.append(answer_dir+'/'+dir2+'/'+file)
                    label_list.append(8)
                    a=file.split('.')
                    image_list.append(img_dir+'/'+file.replace('xml','jpg'))
                    if os.path.exists(img_dir2+'/'+file.replace('xml','jpg')):
                        os.remove(img_dir2+'/'+file.replace('xml','jpg'))
        elif dir2=='油渍' or dir2=='污渍':
            for file in os.listdir(answer_dir+'/'+dir2):
                if ('xml' in file):
                    xml_list.append(answer_dir+'/'+dir2+'/'+file)
                    label_list.append(9)
                    a=file.split('.')
                    image_list.append(img_dir+'/'+file.replace('xml','jpg'))
                    if os.path.exists(img_dir2+'/'+file.replace('xml','jpg')):
                        os.remove(img_dir2+'/'+file.replace('xml','jpg'))
        else:
            for file in os.listdir(answer_dir+'/'+dir2):
                #print(dir2)
                if ('xml' in file):
                    xml_list.append(answer_dir+'/'+dir2+'/'+file)
                    label_list.append(10)
                    a=file.split('.')
                    image_list.append(img_dir+'/'+file.replace('xml','jpg'))
                    if os.path.exists(img_dir2+'/'+file.replace('xml','jpg')):
                        os.remove(img_dir2+'/'+file.replace('xml','jpg'))

    for file in os.listdir(img_dir2):
        image_list.append(img_dir+'/'+file)
        label_list.append(0)                
        xml_list.append(None)
        if os.path.exists(img_dir2+'/'+file):
            os.remove(img_dir2+'/'+file)
    if os.path.exists(img_dir2):        
        try:
            shutil.rmtree(img_dir2)
        except Exception as ex:
            print("错误信息："+str(ex))#提示：错误信息，目录不是空的                                                           
    return image_list,label_list,xml_list
    #返回两个list 分别为图片文件名及其标签  顺序已被打乱
if __name__=="__main__":
    img_list = ['../data/xuelang_round1_train_part1_20180628', '../data/xuelang_round1_train_part2_20180705',
                    '../data/xuelang_round1_train_part3_20180709']
    img_dir='../data/xuelang_round1_test_a_20180709'
    answer_dir='../data/xuelang_round1_answer_a_20180808'
    #image_list,label_list,xml_list=get_files_for_defects(img_list)
    xml_list=[]
    image_list=[]
    label_list=[]
    image_list,label_list,xml_list=get_files_for_defects_test(answer_dir,img_dir,image_list,label_list,xml_list)