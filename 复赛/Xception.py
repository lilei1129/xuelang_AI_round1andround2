import keras
#from keras.utils import multi_gpu_model
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D,Dropout
from keras.optimizers import SGD, Adam, RMSprop, Adadelta
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
#from keras.applications.vgg19 import VGG19
from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input
import tensorflow as tf
import keras.backend as K
import os
# 使用第一张与第三张GPU卡
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


#from sklearn import cross_validation,metrics
##from sklearn import svm
#from sklearn.metrics import average_precision_score

# def new_loss(y_true,y_pred):
    
#     y_pred /= tf.reduce_sum(y_pred,-1, True)
#     y_pred=tf.clip_by_value(y_pred,1e-10,1.0-1e-10)
# #    print(y_true.shape)
#     a_0=-(y_true[:,0] * tf.log(y_pred[:,0]) + (1-y_true[:,0]) * tf.log(1-y_pred[:,0]))
#     a_1=-(y_true[:,1] * tf.log(y_pred[:,1]) + (1-y_true[:,1]) * tf.log(1-y_pred[:,1]))
#     a_2=-(y_true[:,2] * tf.log(y_pred[:,2]) + (1-y_true[:,2]) * tf.log(1-y_pred[:,2]))
#     a_3=-(y_true[:,3] * tf.log(y_pred[:,3]) + (1-y_true[:,3]) * tf.log(1-y_pred[:,3]))
#     a_4=-(y_true[:,4] * tf.log(y_pred[:,4]) + (1-y_true[:,4]) * tf.log(1-y_pred[:,4]))
#     a_5=-(y_true[:,5] * tf.log(y_pred[:,5]) + (1-y_true[:,5]) * tf.log(1-y_pred[:,5]))
#     a_6=-(y_true[:,6] * tf.log(y_pred[:,6]) + (1-y_true[:,6]) * tf.log(1-y_pred[:,6]))
#     a_7=-(y_true[:,7] * tf.log(y_pred[:,7]) + (1-y_true[:,7]) * tf.log(1-y_pred[:,7]))
#     a_8=-(y_true[:,8] * tf.log(y_pred[:,8]) + (1-y_true[:,8]) * tf.log(1-y_pred[:,8]))
#     a_9=-(y_true[:,9] * tf.log(y_pred[:,9]) + (1-y_true[:,9]) * tf.log(1-y_pred[:,9]))
#     a_10=-(y_true[:,10] * tf.log(y_pred[:,10]) + (1-y_true[:,10]) * tf.log(1-y_pred[:,10]))
    
#     haha=963/3331*a_0 + 204/3331*a_1 + 192/3331*a_2 + 313/3331*a_3 + 179/3331*a_4 + 195/3331*a_5 +339/3331*a_6 +213/3331*a_7 + 210/3331*a_8 + 191/3331*a_9 + 332/3331*a_10
#     newloss=tf.reduce_mean(haha)
#     return newloss

# def auc(y_true,y_pred):

#     auc_1 = tf.metrics.auc(y_true[:,0], y_pred[:,0])[1]
#     map_1=tf.metrics.average_precision_at_k(y_true[:,1:],y_pred[:,1:],10)
#     K.get_session().run(tf.local_variables_initializer())
#     return 0.7*auc_1+0.3*map_1

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


height = 512
width = 512


base_model = Xception(include_top=False,
                      weights='imagenet',
                      input_shape=(height, width, 3))
# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu',kernel_regularizer=keras.regularizers.l2(0.0001))(x)
x = Dropout(0.5)(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(11, activation='softmax')(x)
# this is the model we will train
model = Model(input=base_model.input, output=predictions)

adadelta=Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=1e-6)
#sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
#adam=Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy',optimizer=adadelta,metrics=['accuracy'])
model.summary()


# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
 	    preprocessing_function= preprocess_input,
         rotation_range=30,
         #rescale=1./255,
         horizontal_flip=True,
         vertical_flip=True,
         fill_mode='constant',
         channel_shift_range=10,
         cval=0)
#train_datagen = ImageDataGenerator(preprocessing_function= preprocess_input,
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

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(preprocessing_function= preprocess_input)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        r'./data/split_train_round2',
        target_size=(512,512),
        batch_size=10,
        class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        r'./data/split_Verification_round2',
        target_size=(512,512),
        batch_size=10,
        class_mode='categorical')
print(train_generator.class_indices)
print(validation_generator.class_indices)
#创建一个实例history
history = LossHistory()


filepath=r"./Xception_models/cut512-{epoch:02d}-{val_acc:.4f}.h5"


checkpointer_1=ModelCheckpoint(filepath=filepath, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
checkpointer=ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
#model.fit(X_train, Y_train, epochs = 1000, batch_size = 64,callbacks=[checkpointer,checkpointer_1,history],validation_split=0.14)
model.fit_generator(
        train_generator,
        epochs=80,
        verbose=1,
        validation_data=validation_generator,
        class_weight='auto',callbacks=[checkpointer,checkpointer_1,history],workers=4,shuffle=True)
#绘制acc-loss曲线
history.loss_plot('epoch')

