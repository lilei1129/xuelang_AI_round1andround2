from functions_V3 import *


#-------------------------------------第1步，将测试图片转化为训练图片-------------------------------------------------
#
#test_to_train('./data/official/xuelang_round1_answer_a_20180808','./test/xuelang_round1_test_a_20180709')
#test_to_train('./data/official/xuelang_round1_answer_b_20180808','./test/xuelang_round1_test_b')




#-------------------------------------第2步，将所有图片分为11类保存-------------------------------------------------
#rawDir=r'./data/raw'
#gen_data_group(r'./data/official', rawDir)#按9：1分配保存保存完整图片并提取对应xml参数保存成同名json文件






#-------------------------------------第3步，将所有图片进行滑动窗分割-------------------------------------------------
cutDir = r'./data/cut_raw'
###正常图片滑动步长为256
#gen_cut_step_zc(rawDir+'/train/0',600, 800, 600,0.09,0.95, cutDir+'/0')
####瑕疵图片滑动步长为64或者128，取与box重合部分0.2的为瑕疵小图，，，该部分可以根据瑕疵原图的数据集大小酌情修改，
##而且需要注意，有的box比真正的瑕疵大，所有可能最后20%交并比得到的小图里面根本没有真正的瑕疵，需要人工剔除
#gen_cut_step_xc(rawDir+'/train/1',600, 800,600,0.8,0.95, cutDir+'/1')
#gen_cut_step_xc(rawDir+'/train/2',600, 800,600,1,0.95, cutDir+'/2')
#gen_cut_step_xc(rawDir+'/train/3',512, 512,160,1,0.95, cutDir+'/3')
#gen_cut_step_xc(rawDir+'/train/4',512, 512,96,1,0.95, cutDir+'/4')
#gen_cut_step_xc(rawDir+'/train/5',512, 512,128,0.3,0.95, cutDir+'/5')
#gen_cut_step_xc(rawDir+'/train/6',512, 512,256,0.3,0.95, cutDir+'/6')
#gen_cut_step_xc(rawDir+'/train/7',512, 512,160,0.2,0.95, cutDir+'/7')
#gen_cut_step_xc(rawDir+'/train/8',512, 512,96,1,0.95, cutDir+'/8')
#gen_cut_step_xc(rawDir+'/train/9',512, 512,64,0.8,0.95, cutDir+'/9')
#gen_cut_step_xc(rawDir+'/train/99999',512, 512,96,1,0.95, cutDir+'/99999')
#
##-------------------------------------第4步，分出训练集和验证集，同时舍弃一定比例正常图片-------------------------------------------------
##normalNumP为舍弃的正常图片的比例
##percent验证集比例
split_for_var(cutDir,split_num=1,if_only_var=True,percent=0.2,normalNumP=0.8)
#

#-------------------------------------第5步，训练-------------------------------------------------


#img gen
#from keras.preprocessing.image import ImageDataGenerator
#from keras.preprocessing import image
#train_datagen = ImageDataGenerator(rescale=1./255,
#                                   shear_range=0.2,
#                                   zoom_range=0.1,
#                                   horizontal_flip=True,
#                                   vertical_flip=True,
#                                   rotation_range = 90,
#                                   fill_mode = 'constant',
#                                   width_shift_range=0.1,
#                                   height_shift_range=0.1,
#                                   channel_shift_range=10,
#                                   cval = 0)
#
#valic_datagen = ImageDataGenerator(rescale=1./255)
#
#trainGen320 = train_datagen.flow_from_directory(
#        r'./data/split_train_round2',
#        target_size=(512,512),
#        batch_size=12,
#        seed = seed,
#        class_mode='categorical')
#
#valicGen320 = valic_datagen.flow_from_directory(
#        r'./data/split_Verification_round2',
#        target_size=(512,512),
#        batch_size=12,
#        seed = seed,
#        class_mode='categorical')
#
#
#
#
##set model layers
#import keras
#from keras.models import Model, Sequential #导入模型
#from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, GlobalAveragePooling2D #导入连接层
##from spp.SpatialPyramidPooling import SpatialPyramidPooling
#from keras.applications.inception_resnet_v2 import InceptionResNetV2
#from keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions
#
#modelA = InceptionResNetV2(include_top=False,weights='imagenet',input_shape=(None, None, 3))
#x = modelA.output
#x = GlobalAveragePooling2D()(x)
#x = Dense(1024, activation='relu')(x)
#x = Dropout(0.5, seed=seed)(x)
#predictions = Dense(11, activation='softmax')(x)
#modelA = Model(inputs=modelA.input, outputs=predictions)
#
#
#
#
#
##train model 
#from keras.optimizers import Adam, SGD
#from keras.layers import *
#from keras.models import *
#from keras.optimizers import *
#from keras.callbacks import *
#from keras.callbacks import ModelCheckpoint
#resultsA=[]
#optimizer = SGD(lr=0.001, momentum=0.9,  decay=1e-6, nesterov=True)
#modelA.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
#
#filepath=r"./h5/cut320-{epoch:02d}-{val_acc:.4f}.h5"
#checkpoint= ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
#callbacks_list= [checkpoint]
#
#resultA = modelA.fit_generator(
#        trainGen320,
#        epochs=30,verbose=1,
#        callbacks=callbacks_list,
#        validation_data=valicGen320,
#        class_weight='auto')
#resultsA.append(resultA)
#
#
#
#
#
#modelA.save_weights(r'./h5/cut320.h5')
#
#plt.plot(resultA.history['acc'],'b')
#plt.plot(resultA.history['val_acc'],'r')
#plt.plot(resultA.history['loss'],'b')
#plt.plot(resultA.history['val_loss'],'r')
#plt.show()