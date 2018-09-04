# -*- coding: utf-8 -*-

"""
Created on Sun Aug  5 09:58:26 2018

@author: 飞的更高队代码

"""

import predict

import os
import datetime
import tensorflow as tf
import numpy as np
import pandas as pd
import keras
#import numpy as np
from keras.models import Sequential
from keras.initializers import he_normal
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
import keras.backend as K
K.set_image_data_format('channels_last')
from keras.preprocessing import image
from keras.models import Model, load_model
import matplotlib.pyplot as plt
import scipy.misc
import cv2
import xml.etree.cElementTree as et
import shutil
#加载keras模块
#from keras import optimizers
from keras.optimizers import SGD, Adam, RMSprop, Adadelta
from keras.preprocessing.image import ImageDataGenerator

 #创建文件夹
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
    tf.reset_default_graph()
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
    tf.reset_default_graph()
    with tf.Session() as sess:
        for file in os.listdir(dir):
            filename=file
            image_raw_data = tf.gfile.FastGFile(dir+filename, 'rb').read()
            img_data = tf.image.decode_jpeg(image_raw_data)
            img_data.set_shape([1920,2560,3])
            #a=img_data.get_shape()
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
#返回两个list 分别为图片文件名及其标签  顺序已被打乱
            
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}
 
    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))
 
    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))
 
    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()
        
        
        
# build model
def VGG_13_model():
    model = Sequential()
    weight_decay = 0.0001
    dropout=0.5
    # Block 1
    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                 kernel_initializer=he_normal(), name='block1_conv1', input_shape=(499,499,3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                 kernel_initializer=he_normal(), name='block1_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))
    
    # Block 2
    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                 kernel_initializer=he_normal(), name='block2_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                 kernel_initializer=he_normal(), name='block2_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))
    
    # Block 3
    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                 kernel_initializer=he_normal(), name='block3_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                 kernel_initializer=he_normal(), name='block3_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    #model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
    #             kernel_initializer=he_normal(), name='block3_conv3'))
    #model.add(BatchNormalization())
    #model.add(Activation('relu'))
    #model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
    #             kernel_initializer=he_normal(), name='block3_conv4'))
    #model.add(BatchNormalization())
    #model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))
    
    # Block 4
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                 kernel_initializer=he_normal(), name='block4_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                 kernel_initializer=he_normal(), name='block4_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    #model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
    #             kernel_initializer=he_normal(), name='block4_conv3'))
    #model.add(BatchNormalization())
    #model.add(Activation('relu'))
    #model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
    #             kernel_initializer=he_normal(), name='block4_conv4'))
    #model.add(BatchNormalization())
    #model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))
    
    # Block 5
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                 kernel_initializer=he_normal(), name='block5_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
                 kernel_initializer=he_normal(), name='block5_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    #model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
    #             kernel_initializer=he_normal(), name='block5_conv3'))
    #model.add(BatchNormalization())
    #model.add(Activation('relu'))
    #model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_decay),
    #             kernel_initializer=he_normal(), name='block5_conv4'))
    #model.add(BatchNormalization())
    #model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))
    
    # model modification for cifar-10
    #model.add(Flatten(name='flatten'))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1080, use_bias=True, kernel_regularizer=keras.regularizers.l2(weight_decay),
                kernel_initializer=he_normal(), name='fc1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1080, kernel_regularizer=keras.regularizers.l2(weight_decay),
                kernel_initializer=he_normal(), name='fc2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1, kernel_regularizer=keras.regularizers.l2(weight_decay),
                kernel_initializer=he_normal(), name='fc3'))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))
    return model
    
    #G=4
    #if G <= 1:
    #    print("[INFO] training with 1 GPU...")
    #    parallel_model = VGG19_model()
    #
    ## otherwise, we are compiling using multiple GPUs
    #else:
    #    print("[INFO] training with {} GPUs...".format(G))
    #    # we'll store a copy of the model on *every* GPU and then combine
    #    # the results from the gradient updates on the CPU
    #    with tf.device("/cpu:0"):
    #        # initialize the model
    #        model = VGG19_model()
    #        # make the model parallel(if you have more than 2 GPU)
    #    parallel_model = multi_gpu_model(model, gpus=G)
    #    parallel_model.__setattr__('callback_model', model)
	

if __name__=="__main__":
    #创建所有所需的文件夹
    file = "../data/train_data"
    mkdir(file)          
    file2= "../data/test_data"
    mkdir(file2)         
    file3= "../data/resize_test"
    mkdir(file3) 
    file4= "../data/all_data/label_0"
    mkdir(file4) 
    file5= "../data/all_data/label_1"
    mkdir(file5) 
    file6= "../data/train_data/label_0"
    mkdir(file6) 
    file7= "../data/train_data/label_1"
    mkdir(file7) 
    file8= "../data/test_data/label_0"
    mkdir(file8) 
    file9= "../data/test_data/label_1"
    mkdir(file9) 
    file10= "../data/VGG13_models"
    mkdir(file10) 
    #对所有图片进行预处理
    i0=0
    i1=crop_img_xc('../data/xuelang_round1_train_part1_20180628/',499,3,'../data/all_data/label_1',i0)
    i2=crop_img_xc('../data/xuelang_round1_train_part2_20180705/',499,3,'../data/all_data/label_1',i1)
    i3=crop_img_xc('../data/xuelang_round1_train_part3_20180709/',499,3,'../data/all_data/label_1',i2)
    
    i4=0
    i5=crop_img_zc('../data/xuelang_round1_train_part1_20180628/正常/',499,1,'../data/all_data/label_0',i4)
    i6=crop_img_zc('../data/xuelang_round1_train_part2_20180705/正常/',499,1,'../data/all_data/label_0',i5)
    i7=crop_img_zc('../data/xuelang_round1_train_part3_20180709/正常/',499,1,'../data/all_data/label_0',i6)
   
    resize_img_test(dir="../data/xuelang_round1_test_b/",savepath="../data/resize_test",height=499, width=499)
            
                
    #分验证集和训练集
    image_list,label_list = get_files('../data/all_data')     
    train_image_list=list(image_list[0:int(len(image_list)*0.8)])    
    test_image_list=list(image_list[int(len(image_list)*0.8):len(image_list)])

    for filename in train_image_list:
        print(filename)
        if('zc'in filename):
            a=filename.split('/')
            b=a[-1]
            shutil.copy(filename,'../data/train_data/label_0/'+b)
        else:
            a=filename.split('/')
            b=a[-1]
            shutil.copy(filename,'../data/train_data/label_1/'+b)
    for filename in test_image_list:
        if('zc'in filename):
            a=filename.split('/')
            b=a[-1]
            shutil.copy(filename,'../data/test_data/label_0/'+b)
        else:
            a=filename.split('/')
            b=a[-1]
            shutil.copy(filename,'../data/test_data/label_1/'+b)
    
    #模型
    model=VGG_13_model()
    adadelta=Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=1e-6)
    #sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    #adam=Adam(lr=0.06, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='binary_crossentropy',optimizer=adadelta,metrics=['accuracy'])
    model.summary()

    
    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
            #rotation_range=180,
            rescale=1./255,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='constant',
            channel_shift_range=10,
            cval=0)
    
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    
    train_generator = train_datagen.flow_from_directory(
            '../data/train_data',  # this is the target directory
            target_size=(499, 499),  # all images will be resized to 150x150
            batch_size=8,
            class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels
    
    
    validation_generator = test_datagen.flow_from_directory(
            '../data/test_data',
            target_size=(499, 499),
            batch_size=32,
            class_mode='binary')
    print(train_generator.class_indices)
    print(validation_generator.class_indices)

    #创建一个实例history
    history = LossHistory()
    
        
    checkpointer_1=ModelCheckpoint(filepath='../data/VGG13_models/model.hdf5', monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    checkpointer=ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0.00001)
    #model.fit(X_train, Y_train, epochs = 1000, batch_size = 64,callbacks=[checkpointer,checkpointer_1,history],validation_split=0.14)
    model.fit_generator(
            train_generator,
            epochs=100,
            verbose=1,
            validation_data=validation_generator,
            class_weight='auto',callbacks=[checkpointer,checkpointer_1,history],workers=4,shuffle=True)
    #绘制acc-loss曲线
    history.loss_plot('epoch')	
    
    model = load_model(r'../data/VGG13_models/model.hdf5')
    file_dir='../data/resize_test'
    fn=[]
    y_label=[]

    for file in os.listdir(file_dir):
        
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
    dataframe.to_csv(("../submit/submit_"+datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".csv"), index=False,sep=',')
