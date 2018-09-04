# -*- coding: utf-8 -*-

"""
Created on Sun Aug  5 09:58:26 2018

@author: é£çš„æ›´é«˜é˜Ÿä»£ç 

"""

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
#åŠ è½½kerasæ¨¡å—
#from keras import optimizers
from keras.optimizers import SGD, Adam, RMSprop, Adadelta
from keras.preprocessing.image import ImageDataGenerator

 #åˆ›å»ºæ–‡ä»¶å¤¹
def mkdir(path):
 
	folder = os.path.exists(path)
 
	if not folder:                   #åˆ¤æ–­æ˜¯å¦å­˜åœ¨æ–‡ä»¶å¤¹å¦‚æœä¸å­˜åœ¨åˆ™åˆ›å»ºä¸ºæ–‡ä»¶å¤¹
		os.makedirs(path)            #makedirs åˆ›å»ºæ–‡ä»¶æ—¶å¦‚æœè·¯å¾„ä¸å­˜åœ¨ä¼šåˆ›å»ºè¿™ä¸ªè·¯å¾„
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
    # æŸ¥çœ‹æ˜¯å¦å­˜åœ¨æ ‡æ³¨æ¡†ã€‚
    if bbox is None:
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        
    # éšæœºçš„æˆªå–å›¾ç‰‡ä¸­ä¸€ä¸ªå—ã€‚
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
        tf.shape(image), bounding_boxes=bbox, min_object_covered=1)
    #bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
        #tf.shape(image), bounding_boxes=bbox, min_object_covered=0.4)
    distorted_image = tf.slice(image, bbox_begin, bbox_size)

    # å°†éšæœºæˆªå–çš„å›¾ç‰‡è°ƒæ•´ä¸ºç¥ç»ç½‘ç»œè¾“å…¥å±‚çš„å¤§å°ã€‚
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
#æŠŠcatå’Œdogåˆèµ·æ¥ç»„æˆä¸€ä¸ªlistï¼ˆimgå’Œlabï¼‰
    image_list = np.hstack((zc, xc))
    label_list = np.hstack((label_zc, label_xc))
 
#åˆ©ç”¨shuffleæ‰“ä¹±é¡ºåº
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
 
#ä»æ‰“ä¹±çš„tempä¸­å†å–å‡ºlistï¼ˆimgå’Œlabï¼‰
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]                                                   
    return image_list,label_list
#è¿”å›ä¸¤ä¸ªlist åˆ†åˆ«ä¸ºå›¾ç‰‡æ–‡ä»¶ååŠå…¶æ ‡ç­¾ Â é¡ºåºå·²è¢«æ‰“ä¹±
            
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
    #åˆ›å»ºæ‰€æœ‰æ‰€éœ€çš„æ–‡ä»¶å¤¹
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
    #å¯¹æ‰€æœ‰å›¾ç‰‡è¿›è¡Œé¢„å¤„ç†
    i0=0
    i1=crop_img_xc('../data/xuelang_round1_train_part1_20180628/',499,3,'../data/all_data/label_1',i0)
    i2=crop_img_xc('../data/xuelang_round1_train_part2_20180705/',499,3,'../data/all_data/label_1',i1)
    i3=crop_img_xc('../data/xuelang_round1_train_part3_20180709/',499,3,'../data/all_data/label_1',i2)
    
    i4=0
    i5=crop_img_zc('../data/xuelang_round1_train_part1_20180628/æ­£å¸¸/',499,1,'../data/all_data/label_0',i4)
    i6=crop_img_zc('../data/xuelang_round1_train_part2_20180705/æ­£å¸¸/',499,1,'../data/all_data/label_0',i5)
    i7=crop_img_zc('../data/xuelang_round1_ˆvk3óÇL4p¶4›‰4µ]Ë4Ú8ğ3„`‚4Ã4-l4Ê5ÓÇº3xÑD4ÙÙ4ã:h4Â×*3çé‘3(GĞ4Mæ4²_4F.ü3ˆ@_4¹!—4Óğn5(u6×Ó=4xsc5ˆ‰½54Š5‡86¥Êƒ5=Ï5YŞY62-Ç54…5|C›5Ø³5‘Ôœ5*¥4çİ5ÒÍ5Ş6	³5ém¸4ÄC5ìƒ5Å2a5lSO5É6ò4DÛ…56lŸ5s¹Ë5§›*5¾J†5;Ê¸5š€ƒ5¸¨W5\şJ6â-©5	Î5KÏ68õ¸5N\¥5y)³5ÛœÂ5ãDC65|è4hh³5“¯5LIc6âoÀ5½Ñ`5HÏå4ãa¦5{ÂĞ5i54¡»X5–ê¥5øC6¥\5“ÒÃ4Õ6Ò5@ä5¹¾È4¡Ÿ5y5¦+Ù4]‡5£¾5•le5eq‚5'06»Va5Ï„5‰å5’³Ô5³r­5Üãâ5ŸÛ£4Ñõ†50Xî4‰r‹6ÄI’5ÆÒQ5Ì5Œ5Hr:5¸L5Jvç5Š5ÏTÀ5ª	6w6œ½5ñıH5/Ş5x(P6Ø2¿4ÚàN5v4É–F5‰mq5Öl6îe6e$5
î6`VÒ5z„Ü5ì·5Œµ5*îœ4•[6^yÅ5,6ı¨Š6šX5Š<5AØ6T³Ü6&86#n5¥¨>6'5h^~5¶6¼:.5èq6‘05%º›6åV6ñC;6£•4Ç%C5)À5œ6Ê)Ò5F_ü5Ñ²Ğ5-s5©aZ5Š½ö4ŠË‹5Ôò5ã3y5©³(5»j6èMÓ4‘í…5˜=-6î#§5<Yä5(4Ğ“56e05\’4Üa%5ZªŠ5ÖÅ 5d­5ÄÛµ5Ğª5¸Ó6F…6hz?5_¾4À|6î	5Z¡&6²îI6 vd5*Bô3ër¡3w¶A6ÒV4±´5¡.É5z#5²·5I;G6À?ì5îıE5C14+@ 5ò04»°v5Ãm–5·¯6uç6a±5^@:6>õ5­¬´4#q6\j¶5’û5(Ú4*$5ò‘5L„˜5HË6™X÷5°â…5J€Ù5òª”5„ìŸ5Ÿ´5İÅÃ5'5®š5Ó¦6#5û1Œ5èÏ-6ñK5ŞWE6ª@%5"Rø5õ·H4¥|‘5*P6²ê'5^ök5*¯4×Óà4É;	60¸,6İä5¦¨5H¹°4	†5:’596.6‰E¼4>l6Ó<55¥gï5û;º4`Èõ4O )6™ñI5Z<
6
ps5ÿ…5înŞ5¸ê!5Àp?6iEL5q±5¦œL629c6º=ƒ4j;5w²…5d%Ó5×â5»îã4lô5)2D6hÄØ4„0\5-@3¯Ê<4ãy+3>ê˜3üT£4 ‚^4HN‹5zî.4”`4Ñ¦¬4àc5104¤¯—4á—Ó2ìĞ43ù3‹‰ 4Œ «4·;4°94¿òü3N/A4”Ú24ÒZ÷3ÛB4(?&4¼g!48Ñˆ3OSd4¶˜4Ymı3
[Â4ßÛ50X5VE5¢Æ\5–	²4ŒÚ4­¡õ3„;¡4åtG5öw.46a–4H“¨4#5¯©p4hÆ4FB3ş‚4ÃVÀ3W—o2ÎÖ4Ï y4rş–4”G4˜35b‹†4ñ–32€4Iò3¾Ò¿3y%÷3½ÅŸ3R4®ü]4Æ?5 Ww4°¼3´û»3±—ì4æ>5¯0Œ4õÂ3I'²4Í/4H t5· °3¸Õ@4ãÜ‘4®Ø3Ú­3Éd4±Ù:4Í#ß4zd»4ù4ÀŒà4kH]4¯Éã4Ò’5‹Èá35€4Ú(3ÌB4†Qä4N)5l(V4„054Ÿ«O5¢ãÒ4`+4Nmj5Ô7ˆ4i,3W[42Â3‰)û5nıU4XT+3~Á¼3M©¤4Æ{Î404ïÜR4¤¥Ú4TÒ43QŒ4l5¯C‡3Xö¼4H=4Íæ5èTö4ïI4t«+3?½Ó3Åj“3N…n5†{†4$¦ 4÷0Æ4TµÉ3*—r4<Q4Š#¤3ÿû³4'â„3Á£3¾“L5J%3Äï”4tÌ»4Áşf4v‰b4ÑË83¤4Cç94Ã$'4M‘3Ô…3®_4­r4U! 44C=Ç4Ñçå4Õs5`5‘3éÇø2Òê–4W4:ÅÛ4øõ4P@<4;Ğ†3n•2Vm5¯¨Ã3AÎ3‚À4uêg4£Ú 5È£4E=5ñçb4ı½™3âú 4¡î3O‘4f4{ü446aJ4.G4ñns4P°$3ê5£à4ûâå3±3xäA3w½!4*y4¸Âû3\¡94ÜİJ4‰Öò4Ã4Vc4·§3Ø(5÷ã3›Ò05Oñ 5Ğl4‘òş3WbU5ÂZŞ3ŠE5@¢3tÅù3¤¯4T‰4®ÒÙ3ÒÖ4Š0¦4µri3şú4Ú	5L‡N4Š ¡4š÷¡4«
4‡>4z%4İX4htÉ4“ş3Z–5¥w 4pŒ÷4áä—3i©G4ÎAÿ414N 5”¢4¿4„›µ4iòz4AzÉ4Ò=¢3òZz46y€5xƒ4ÇµŒ3›ÇÖ3ç~Ø4­O%5…ı3U˜h4Ö„L5k
Ô4e£4¨×İ5ÜZ4•ç‰5i4»W—49å'5À\5ATŒ5böL5ÖF5¶CÑ4u'35¾4Ê$†5Ä¦S4•A5X-¶4Îqí41L5¦C¹4¯$5¼?5QS>5¦®ú4òo45Ğ!5
c-4T f5Ek4O15Ø>5Ëî
5¹\5‘’6–£‹5‹Éa5Õ‰5_5¹ 5Ğ¯5h'¡5óº»5›”4/Â5‡W5¥[°50¤@5È¼ş4Iè•4°‘5Ü24·!à3ñ‘4‡E*5ş•*5³¿Š4@ë4@±Ÿ4g]G5:G_4Ù¤¹5’¯æ4…É¾4ø‡
5½®4@¦4†ƒí4Ğš»5¦a5³Ç®4³`‰4f¦?5l\5˜R"5Ï/”4Í35oˆ94º‡5Økù3rğ5#¬N5×’¹4–$p4â^5@èì4~Ij5æİb5ûCÀ5ùp5M	5­¨5Nİ50–‚4ó(	5«^3é25ö©5ê]58357,G5ñj¤5o6Ş4/ÓD5÷2ï4Ê„.5.U4¹ô(5¾ÊÏ4.œ4êŒ5“î°4Ù4¿‡6Ô°5ó À4ŸÓ-5J5îú4mœô4 É§5.Ê¨57ï 6âşÒ4Ó{5˜ş¢4ºÜK5Ü›Í3îe®4¤4¥
.6é%5}4 P5—‚49jÔ4 4Ó Y4“óh5e/4£ËÈ4ç0Ó54€,å4’ıˆ5u5j>.5¨Êº4MB4-xà4§Dg4 mh4§É@4gœ'5DÚñ4Rm¨4l);5ÁÉ4šÛ5†.5§¿4]Iİ3Éz95®ã4ŠC6$<F5
úE4.ª…4 Í3®IÍ5¸Ò 4Ş|4­:æ4tíy4I1}5(ƒ5%¤‡5@Ş4ôÕ¡3— 5Qeâ4Ã45Ÿö;5q`‘5Ë#6Òß4a.5]ì4A–4Ğ)N5¼K\4%âí4íB4//S4S÷…5Óæ4]T5.vŠ5w´Ê4Â·‡5—D54¸4	5év5wl4lçú4Æ×5j¤°4¦3¹4oÃƒ5||[48A…5¨ß4)Z5^4ÏşL57!5 ^5ÂãÜ4äJN4ıõŞ4é@5•ò!5…ÙS4>¸”5Şv*4[-Í4d9Ã5;b5õíÜ5øéÄ4E¨5N˜Ç4XWS5æÓ-42€ı4¸Ó5&Õt5ÂNè4Û‘4ÈÌM57®¸5û—41g6Ípx4÷§ƒ59½À5¬(5ƒ†49Ö4ï¥4¹=5é‰4yƒ¿4´›4·ø4E5¶f5¶¦3â+5¿>x4h84z\'5ìÆÇ4f¥4Z5â±5‚‘q4•í“5]xù3g85m”Æ3z*@5Ê
’4Iş@4‹Äw4e5¹¶¹4Í‘¦4Ãğ˜5Fv5:*
4©…4M„$4¬35–(ß3ØAY4ĞÈ25ôô 5ïTŸ4N5D§º5¦?5«‹5vÿ4ëßÇ4§Ò85Úmç4zLÉ5ÿ3 4b‡5~øú4ÁE5Ğhİ4–È
5+4iw/5’aN4R4+4½š’4’"û4`XU4`z4s‘^4oƒ#5’öâ4CÄá3’ş+5ªŞ84OÎ±4F«ÿ4Ø^5û]5Â×‹4\Á5<5îÊÑ4ş^4ı5£è•5¶JÖ4 #ñ3æŠQ5){4ıíŞ5ıí‰3¦5$5}4S—h4¶¬	4ìâ•5E;65“ X4¨5íyU5›Š›5ıÍ«4Ññ6¼‡Æ5„l15!²4ih~3e9®3ÂÊ4lê5¢!5£O4Enm5òá5=˜›4‹fÙ5œ€4.C4ôòÙ4ûµT4Ú&5_~ò4_o%5E4w/05&66w'ı4¬4Òù³4Bñy4Œœ 5 ö5¹ªÜ4:eİ5–¢r4İîÃ4Ìá¤5	’¸4eÄm3·¤D5.yP4³Ç5ÆS4–4 nP5Tƒ4æÜ4˜AH4¸£…4¬R5=‰4çÁÇ3n°$6FÒë3‡Õ¼4¢û¸5ğU4™Œ4õÂ3À0535@­{5;˜B4ña4;éæ4„,5ê¨A5~e>4}í¤4Úb•5LÚ5Ğ(4ªä 4ÉÓÛ4z	ö3±™ì4mƒİ5[Ï?4¨u4[»2µ+A5]3]‰4»ˆ;4]5jÒ5£m6g ô4~>4lœT3†w05|[4ÁÚ95»¶W4–@5M<6,òà4bÓ_5=é5{4	o"6›_4{UÏ3Î‰)4–$4¨…®4VøÛ4ù¥4Yª
5,I•4ÆS5²Ey4™V4„â4Û^5æ¸3c^’4äß5d¡4®Aº5†/§5OÓ,4““Æ5É•Ä4±4sX©3í‚4ª9€4
ÎÉ4M™¹4Ÿ\é3)]Q4â5<²4‚€‰4§¤4QDQ4Êí‘5£5¿[î4ÄÓN6 R®4«¬5`04ĞF5±Ç‚3,üd4Œ35iwº4Ì¬5óˆ¯4fnU54bş5E©Í5}Ò“4<üC40ƒ5\‡í4öğ4Œ?4Âl£5<_À4ûÉb5WÃ4ş†5 ¯?52e­5ß
6{ª4áR 6;ƒW5'¼(5Î'6áæ5C5YoÙ6\
6Ê6:º5ha55¶A6–75j=·5åEş4l½6‰Ü®5µŒ5tM5Ó%
6¦	$6'Ä…5"Ÿ3