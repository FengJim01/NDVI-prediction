import tensorflow as tf
import keras
import os
import time
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input, Activation, Add, Concatenate, BatchNormalization, \
    Cropping2D
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, Callback
from keras.initializers import glorot_normal, he_normal
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from model import *


# 設定迭代停止器
# 當loss function 低於某個值時，迭代自動停止
class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='val_loss', value=0.0001, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True
        # save the weights in every epoch
        self.model.save_weights("./weights/VGG_%d.h5" % epoch)


def datasetclip(path, input_size, stride_x, stride_y, num, mode=1):
    """
    crop the images and save.
    :param path: the image npy path
    :param input_size: the size of the input image dataset
    :param stride_x: the horizontal stride to move the mask
    :param stride_y: the vertical stride to move the mask
    :param num: the number of the image. 1 mean one image and 2 mean multiple images
    :param mode: the RGB image mode and the NDVI image mode. 1 is the RGB mode and 2 is the NDVI mode
    :return: the dataset npy
    """
    image = np.load(path, allow_pickle=True)  # (X,1280,960,3)
    save_np = []
    if num == 1:
        image = np.expand_dims(image, axis=0)
    else:
        pass
    width = image.shape[2]
    height = image.shape[1]
    boundary = 50
    if mode == 1:
        n = 0
        for i in image:
            x = y = boundary
            while x + input_size < width - boundary:
                while y + input_size < height - boundary:
                    save_np.append(i[y:y + input_size, x:x + input_size, :])
                    y += stride_y
                y = boundary
                x += stride_x
            print("image :", n, "/", len(image), end='\r')
            n += 1
    elif mode == 2:
        n = 0
        for i in image:
            x = y = boundary
            x1 = y1 = int(boundary + (input_size - 1) / 2)
            while x + input_size < width - boundary:
                while y + input_size < height - boundary:
                    save_np.append(i[y1, x1])
                    y += stride_y
                    y1 += stride_y
                x += stride_x
                x1 += stride_x
                y = boundary
                y1 = int(boundary + (input_size - 1) / 2)
            print("image", n, "/", len(image), end='\r')
            n += 1
    return np.asarray(save_np)


if __name__ == "__main__":
    Model = Dense1_relu()  # 從model.py中選模型架構使用
    rgb_path = os.path.join('.', 'dataset', 'train(20m)', 'train_20meter_RGB.npy')
    ndvi_path = os.path.join('.', 'dataset', 'train(20m)', 'train_20meter_NDVI.npy')
    rgb = datasetclip(rgb_path, 51, 20, 20, 2, mode=1)
    ndvi = datasetclip(ndvi_path, 51, 20, 20, 2, mode=2)

    # Data Augmentation
    datagen = ImageDataGenerator(
        zca_whitening=False,
        horizontal_flip=True,
        vertical_flip=True,
        # brightness_range=(0.1,0.5,1.0)
        # rotation_range=180,
        # width_shift_range=0.3,
        # height_shift_range=0.3,
    )

    print(rgb.size)
    print(ndvi.size)
    start = time.time()
    callbacks = [EarlyStoppingByLossVal(monitor='loss', value=1e-7, verbose=1)]  # 迭代停止器設定
    adam = optimizers.Adam(lr=1e-4)  # 設定優化器的種類和學習率
    Model.compile(loss='mean_squared_error', optimizer=adam, metrics=['acc'])
    # history = Model.fit(rgb, ndvi, epochs=5, batch_size=12, callbacks=callbacks, validation_split=0.001)
    history= Model.fit_generator(datagen.flow(rgb, ndvi, batch_size=12),
                                 samples_per_epoch=1000,
                                 nb_epoch=40,
                                 callbacks=callbacks)

    Model.save_weights("trained_model.h5")
    loss = history.history['loss']
    # val_loss = history.history['val_loss']
    np.save("loss_function.npy", np.asarray(loss))
    # np.save("val_loss.npy", np.asarray(val_loss))
    end = time.time()
    print("Time : %f sec" % (end - start))
