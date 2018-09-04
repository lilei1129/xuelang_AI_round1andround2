# -*- coding: utf-8 -*-
from DataSet_list import *
from pre_process import *
import cv2
import tensorflow as tf
import os
# 使用第一张与第三张GPU卡
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
'''

函数功能：利用DataSet_list生成的所有原始数据的image_list,label_list,xml_list对应列表
        利用xml信息对图像进行预处理增强工作，处理之后的图像存放在data/train_round2 列表之下      
'''

####  num_list 为各个瑕疵类别的增强程度，为整数  如3指一张瑕疵图增强为原来的3张，，resize原图为默认操作
####  需在程序启动前设置该list
num_list=[0,5,7,1,4,3,1,5,3,9,1]

def zengqiang_round2(image_list,label_list,xml_list,weight,height):
    for i in LABELS:
        mkdir('../data/train_round2/'+i)
    for i in range(len(image_list)):
        if(xml_list[i]):
            #如果是瑕疵图
            tf.reset_default_graph()
            with tf.Session() as sess:
                tree=et.parse(xml_list[i])
                root=tree.getroot()
                filename=root.find('filename').text
                for Object in root.findall('object'):
                    bndbox=Object.find('bndbox')
                    xmin=bndbox.find('xmin').text
                    ymin=bndbox.find('ymin').text
                    xmax=bndbox.find('xmax').text
                    ymax=bndbox.find('ymax').text
                    image_raw_data = tf.gfile.FastGFile(image_list[i], 'rb').read()
                    img_data = tf.image.decode_jpeg(image_raw_data)       
                    img_data.set_shape([1920,2560,3])
                    a=img_data.get_shape()
                    box=tf.constant([[[int(ymin)/int(a[0]),int(xmin)/int(a[1]),int(ymax)/int(a[0]),int(xmax)/int(a[1])]]], dtype=tf.float32)
                    for j in range(num_list[label_list[i]]):
                        res = preprocess_for_train(img_data, weight, height, box)
                        cv2.imwrite('../data/train_round2/'+LABELS[label_list[i]]+'/'+str(j+1)+'_'+filename,res)                                                
                img=cv_imread(image_list[i])
                res2 = cv2.resize(img,(weight,height), interpolation = cv2.INTER_AREA)
                cv2.imwrite('../data/train_round2/'+LABELS[label_list[i]]+'/'+filename,res2)
        else:
            #如果是正常图片
            tf.reset_default_graph()
            with tf.Session() as sess:
                image_raw_data = tf.gfile.FastGFile(image_list[i], 'rb').read()
                img_data = tf.image.decode_jpeg(image_raw_data)       
                img_data.set_shape([1920,2560,3])
                a=img_data.get_shape()
                box=None
                filename=image_list[i].split('/')[3]
                if(('jpg' in filename)!=True):
                    filename=image_list[i].split('/')[4]
                print(filename)
                for j in range(num_list[label_list[i]]):
                    res = preprocess_for_train(img_data, weight, height, box)
                    cv2.imwrite('../data/train_round2/'+LABELS[label_list[i]]+'/'+str(j+1)+'_'+filename,res)                                                
                img=cv_imread(image_list[i])
                res2 = cv2.resize(img,(weight,height), interpolation = cv2.INTER_AREA)
                cv2.imwrite('../data/train_round2/'+LABELS[label_list[i]]+'/'+filename,res2)
    return 0

if __name__=="__main__":     
    img_list = ['../data/xuelang_round1_train_part1_20180628', '../data/xuelang_round1_train_part2_20180705',
                    '../data/xuelang_round1_train_part3_20180709']
    img_dir1='../data/xuelang_round1_test_a_20180709'
    img_dir2='../data/xuelang_round1_test_b'
    answer_dir1='../data/xuelang_round1_answer_a_20180808'
    answer_dir2='../data/xuelang_round1_answer_b_20180808'
    image_list,label_list,xml_list=get_files_for_defects(img_list)
    image_list,label_list,xml_list=get_files_for_defects_test(answer_dir1,img_dir1,image_list,label_list,xml_list)
    image_list,label_list,xml_list=get_files_for_defects_test(answer_dir2,img_dir2,image_list,label_list,xml_list)
    zengqiang_round2(image_list,label_list,xml_list,800,600)