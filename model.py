import tensorflow as tf
import keras
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input, Activation, Add, Concatenate, BatchNormalization, \
    Cropping2D
from keras.models import Model, load_model
from keras.initializers import glorot_normal, he_normal
from keras.regularizers import l2, l1, l1_l2
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator


def CNN1():
    X_input = Input((51, 51, 3))
    # stage 1
    X = Conv2D(4, (3, 3), strides=(1, 1), name="conv1", padding="same")(X_input)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(4, (3, 3), strides=(1, 1), name="conv1_2", padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), strides=(2, 2))(X)
    # stage 2
    X = Conv2D(8, (3, 3), strides=(1, 1), name="conv2", padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(8, (3, 3), strides=(1, 1), name="conv2_2", padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), strides=(2, 2))(X)

    flatten = Flatten()(X)
    fcn = Dense(4096)(flatten)
    fcn = BatchNormalization()(fcn)
    fcn = Activation('relu')(fcn)
    fcn = Dense(1)(fcn)
    fcn = Activation('tanh')(fcn)
    model = Model(X_input, fcn)
    return model


def CNN2():
    X_input = Input((51, 51, 3))
    # stage 1
    X = Conv2D(4, (3, 3), strides=(1, 1), name="conv1", padding="same")(X_input)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(4, (3, 3), strides=(1, 1), name="conv1_2", padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), strides=(2, 2))(X)
    # stage 2
    X = Conv2D(8, (3, 3), strides=(1, 1), name="conv2", padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(8, (3, 3), strides=(1, 1), name="conv2_2", padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), strides=(2, 2))(X)
    # satge 3
    X = Conv2D(16, (3, 3), strides=(1, 1), name="conv3_1", padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(16, (3, 3), strides=(1, 1), name="conv3_2", padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), strides=(2, 2))(X)

    flatten = Flatten()(X)
    fcn = Dense(4096)(flatten)
    fcn = BatchNormalization()(fcn)
    fcn = Activation('relu')(fcn)
    fcn = Dense(1)(fcn)
    fcn = Activation('tanh')(fcn)
    model = Model(X_input, fcn)
    model.summary()
    return model


def CNN3():
    X_input = Input((51, 51, 3))
    # stage 1
    X = Conv2D(4, (3, 3), strides=(1, 1), name="conv1", padding="same")(X_input)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(4, (3, 3), strides=(1, 1), name="conv1_2", padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), strides=(2, 2))(X)
    # stage 2
    X = Conv2D(8, (3, 3), strides=(1, 1), name="conv2", padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(8, (3, 3), strides=(1, 1), name="conv2_2", padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), strides=(2, 2))(X)
    # satge 3
    X = Conv2D(16, (3, 3), strides=(1, 1), name="conv3_1", padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(16, (3, 3), strides=(1, 1), name="conv3_2", padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), strides=(2, 2))(X)
    # stage 4
    X = Conv2D(32, (3, 3), strides=(1, 1), name="conv4_1", padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(32, (3, 3), strides=(1, 1), name="conv4_2", padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), strides=(2, 2))(X)

    flatten = Flatten()(X)
    fcn = Dense(4096)(flatten)
    fcn = BatchNormalization()(fcn)
    fcn = Activation('relu')(fcn)
    fcn = Dense(1)(fcn)
    fcn = Activation('tanh')(fcn)
    model = Model(X_input, fcn)
    return model


def ResNet1():
    X_input = Input((51, 51, 3))
    # Conv1
    X = Conv2D(4, (3, 3), strides=(1, 1), name="conv1", padding="same")(X_input)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    # ResNet1
    X_shortcut1 = X
    X = Conv2D(4, (3, 3), strides=(1, 1), name='res1_1', padding='same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(4, (3, 3), strides=(1, 1), name="res1_2", padding="same")(X)
    Res = Add()([X_shortcut1, X])
    X = BatchNormalization()(Res)
    X = Activation('relu')(X)
    # ResNet2
    X_shortcut2 = X
    X = Conv2D(8, (3, 3), strides=(2, 2), name="res2_1", padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(8, (3, 3), strides=(1, 1), name="res2_2", padding="same")(X)
    X_shortcut2 = Conv2D(8, (3, 3), strides=(2, 2), name='res2_3', padding='same')(X_shortcut2)
    Res2 = Add()([X, X_shortcut2])
    X = BatchNormalization()(Res2)
    X = Activation('relu')(X)
    # ResNet3
    X_shortcut3 = X
    X = Conv2D(8, (3, 3), strides=(1, 1), name="res3_1", padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(8, (3, 3), strides=(1, 1), name="res3_2", padding="same")(X)
    Res3 = Add()([X, X_shortcut3])
    X = BatchNormalization()(Res3)
    X = Activation('relu')(X)

    flatten = Flatten()(X)
    fcn = Dense(4096)(flatten)
    fcn = BatchNormalization()(fcn)
    fcn = Activation('relu')(fcn)
    fcn = Dense(1)(fcn)
    fcn = Activation('tanh')(fcn)
    model = Model(X_input, fcn)
    return model


def ResNet2():
    X_input = Input((51, 51, 3))
    # Conv1
    X = Conv2D(4, (3, 3), strides=(1, 1), name="conv1", padding="same")(X_input)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    # ResNet1
    X_shortcut1 = X
    X = Conv2D(4, (3, 3), strides=(1, 1), name='res1_1', padding='same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(4, (3, 3), strides=(1, 1), name="res1_2", padding="same")(X)
    Res = Add()([X_shortcut1, X])
    X = BatchNormalization()(Res)
    X = Activation('relu')(X)
    # ResNet2
    X_shortcut2 = X
    X = Conv2D(8, (3, 3), strides=(2, 2), name="res2_1", padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(8, (3, 3), strides=(1, 1), name="res2_2", padding="same")(X)
    X_shortcut2 = Conv2D(8, (3, 3), strides=(2, 2), name='res2_3', padding='same')(X_shortcut2)
    Res2 = Add()([X, X_shortcut2])
    X = BatchNormalization()(Res2)
    X = Activation('relu')(X)
    # ResNet3
    X_shortcut3 = X
    X = Conv2D(8, (3, 3), strides=(1, 1), name="res3_1", padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(8, (3, 3), strides=(1, 1), name="res3_2", padding="same")(X)
    Res3 = Add()([X, X_shortcut3])
    X = BatchNormalization()(Res3)
    X = Activation('relu')(X)
    # ResNet4
    X_shortcut4 = X
    X = Conv2D(16, (3, 3), strides=(2, 2), name="res4_1", padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(16, (3, 3), strides=(1, 1), name="res4_2", padding="same")(X)
    X_shortcut4 = Conv2D(16, (3, 3), strides=(2, 2), name='res4_3', padding='same')(X_shortcut4)
    Res4 = Add()([X, X_shortcut4])
    X = BatchNormalization()(Res4)
    X = Activation('relu')(X)
    # ResNet5
    X_shortcut5 = X
    X = Conv2D(16, (3, 3), strides=(1, 1), name="res5_1", padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(16, (3, 3), strides=(1, 1), name="res5_2", padding="same")(X)
    Res5 = Add()([X, X_shortcut5])
    X = BatchNormalization()(Res5)
    X = Activation('relu')(X)

    flatten = Flatten()(X)
    fcn = Dense(4096)(flatten)
    fcn = BatchNormalization()(fcn)
    fcn = Activation('relu')(fcn)
    fcn = Dense(1)(fcn)
    fcn = Activation('tanh')(fcn)
    model = Model(X_input, fcn)
    model.summary()
    return model


def ResNet3():
    X_input = Input((51, 51, 3))
    # Conv1
    X = Conv2D(4, (3, 3), strides=(1, 1), name="conv1", padding="same")(X_input)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    # ResNet1
    X_shortcut1 = X
    X = Conv2D(4, (3, 3), strides=(1, 1), name='res1_1', padding='same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(4, (3, 3), strides=(1, 1), name="res1_2", padding="same")(X)
    Res = Add()([X_shortcut1, X])
    X = BatchNormalization()(Res)
    X = Activation('relu')(X)
    # ResNet2
    X_shortcut2 = X
    X = Conv2D(8, (3, 3), strides=(2, 2), name="res2_1", padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(8, (3, 3), strides=(1, 1), name="res2_2", padding="same")(X)
    X_shortcut2 = Conv2D(8, (3, 3), strides=(2, 2), name='res2_3', padding='same')(X_shortcut2)
    Res2 = Add()([X, X_shortcut2])
    X = BatchNormalization()(Res2)
    X = Activation('relu')(X)
    # ResNet3
    X_shortcut3 = X
    X = Conv2D(16, (3, 3), strides=(2, 2), name="res4_1", padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(16, (3, 3), strides=(1, 1), name="res4_2", padding="same")(X)
    X_shortcut3 = Conv2D(16, (3, 3), strides=(2, 2), name='res4_3', padding='same')(X_shortcut3)
    Res3 = Add()([X, X_shortcut3])
    X = BatchNormalization()(Res3)
    X = Activation('relu')(X)

    flatten = Flatten()(X)
    fcn = Dense(4096)(flatten)
    fcn = BatchNormalization()(fcn)
    fcn = Activation('relu')(fcn)
    fcn = Dense(1)(fcn)
    fcn = Activation('tanh')(fcn)
    model = Model(X_input, fcn)
    return model


def ResNet25():
    X_input = Input((25, 25, 3))
    # Conv1
    X = Conv2D(4, (3, 3), strides=(1, 1), name="conv1", padding="same")(X_input)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    # ResNet1
    X_shortcut1 = X
    X = Conv2D(4, (3, 3), strides=(1, 1), name='res1_1', padding='same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(4, (3, 3), strides=(1, 1), name="res1_2", padding="same")(X)
    Res = Add()([X_shortcut1, X])
    X = BatchNormalization()(Res)
    X = Activation('relu')(X)
    # ResNet2
    X_shortcut2 = X
    X = Conv2D(8, (3, 3), strides=(2, 2), name="res2_1", padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(8, (3, 3), strides=(1, 1), name="res2_2", padding="same")(X)
    X_shortcut2 = Conv2D(8, (3, 3), strides=(2, 2), name='res2_3', padding='same')(X_shortcut2)
    Res2 = Add()([X, X_shortcut2])
    X = BatchNormalization()(Res2)
    X = Activation('relu')(X)
    # ResNet3
    X_shortcut3 = X
    X = Conv2D(16, (3, 3), strides=(2, 2), name="res3_1", padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(16, (3, 3), strides=(1, 1), name="res3_2", padding="same")(X)
    X_shortcut3 = Conv2D(16, (3, 3), strides=(2, 2), name='res3_3', padding='same')(X_shortcut3)
    Res3 = Add()([X, X_shortcut3])
    X = BatchNormalization()(Res3)
    X = Activation('relu')(X)

    flatten = Flatten()(X)
    fcn = Dense(4096)(flatten)
    fcn = BatchNormalization()(fcn)
    fcn = Activation('relu')(fcn)
    fcn = Dense(1)(fcn)
    fcn = Activation('tanh')(fcn)
    model = Model(X_input, fcn)
    return model


def ResNet101():
    X_input = Input((101, 101, 3))
    # Conv1
    X = Conv2D(4, (3, 3), strides=(1, 1), name="conv1", padding="same")(X_input)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    # ResNet1
    X_shortcut1 = X
    X = Conv2D(4, (3, 3), strides=(1, 1), name='res1_1', padding='same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(4, (3, 3), strides=(1, 1), name="res1_2", padding="same")(X)
    Res = Add()([X_shortcut1, X])
    X = BatchNormalization()(Res)
    X = Activation('relu')(X)
    # ResNet2
    X_shortcut2 = X
    X = Conv2D(8, (3, 3), strides=(2, 2), name="res2_1", padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(8, (3, 3), strides=(1, 1), name="res2_2", padding="same")(X)
    X_shortcut2 = Conv2D(8, (3, 3), strides=(2, 2), name='res2_3', padding='same')(X_shortcut2)
    Res2 = Add()([X, X_shortcut2])
    X = BatchNormalization()(Res2)
    X = Activation('relu')(X)
    # ResNet3
    X_shortcut3 = X
    X = Conv2D(16, (3, 3), strides=(2, 2), name="res3_1", padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(16, (3, 3), strides=(1, 1), name="res3_2", padding="same")(X)
    X_shortcut3 = Conv2D(16, (3, 3), strides=(2, 2), name='res3_3', padding='same')(X_shortcut3)
    Res3 = Add()([X, X_shortcut3])
    X = BatchNormalization()(Res3)
    X = Activation('relu')(X)

    flatten = Flatten()(X)
    fcn = Dense(4096)(flatten)
    fcn = BatchNormalization()(fcn)
    fcn = Activation('relu')(fcn)
    fcn = Dense(1)(fcn)
    fcn = Activation('tanh')(fcn)
    model = Model(X_input, fcn)
    return model


def SPPNet():
    X_input = Input((51, 51, 3))
    # Conv1
    X = Conv2D(4, (3, 3), strides=(1, 1), name="conv1", padding="same")(X_input)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    # ResNet1
    X_shortcut1 = X
    X = Conv2D(4, (3, 3), strides=(1, 1), name='res1_1', padding='same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(4, (3, 3), strides=(1, 1), name="res1_2", padding="same")(X)
    Res = Add()([X_shortcut1, X])
    X = BatchNormalization()(Res)
    X = Activation('relu')(X)
    # ResNet2
    X_shortcut2 = X
    X = Conv2D(8, (3, 3), strides=(2, 2), name="res2_1", padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(8, (3, 3), strides=(1, 1), name="res2_2", padding="same")(X)
    X_shortcut2 = Conv2D(8, (3, 3), strides=(2, 2), name='res2_3', padding='same')(X_shortcut2)
    Res2 = Add()([X, X_shortcut2])
    X = BatchNormalization()(Res2)
    X = Activation('relu')(X)
    # ResNet3
    X_shortcut3 = X
    X = Conv2D(16, (3, 3), strides=(2, 2), name="res3_1", padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(16, (3, 3), strides=(1, 1), name="res3_2", padding="same")(X)
    X_shortcut3 = Conv2D(16, (3, 3), strides=(2, 2), name='res3_3', padding='same')(X_shortcut3)
    Res3 = Add()([X, X_shortcut3])
    X = BatchNormalization()(Res3)
    X = Activation('relu')(X)

    # SPP
    row = int(X.shape[1])
    column = int(X.shape[2])
    pyramid1 = MaxPooling2D((row, column), strides=(1, 1))(X)
    pyramid2 = MaxPooling2D((int(row / 2) + 1, int(column / 2) + 1), strides=(int(row / 2), int(column / 2)))(X)
    pyramid3 = MaxPooling2D((int(row / 3) + 1, int(column / 3) + 1), strides=(int(row / 3), int(column / 3)))(X)

    pyramid1 = Flatten()(pyramid1)
    pyramid2 = Flatten()(pyramid2)
    pyramid3 = Flatten()(pyramid3)
    flatten = Concatenate()([pyramid3, pyramid2, pyramid1])

    fcn = Dense(4096)(flatten)
    fcn = BatchNormalization()(fcn)
    fcn = Activation('relu')(fcn)
    fcn = Dense(1)(fcn)
    fcn = Activation('tanh')(fcn)
    model = Model(X_input, fcn)
    return model


def FCN1():
    X_input = Input((51, 51, 3))
    # Conv1
    X = Conv2D(4, (3, 3), strides=(1, 1), name="conv1", padding="same")(X_input)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    # ResNet1
    X_shortcut1 = X
    X = Conv2D(4, (3, 3), strides=(1, 1), name='res1_1', padding='same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(4, (3, 3), strides=(1, 1), name="res1_2", padding="same")(X)
    Res = Add()([X_shortcut1, X])
    X = BatchNormalization()(Res)
    X = Activation('relu')(X)
    # ResNet2
    X_shortcut2 = X
    X = Conv2D(8, (3, 3), strides=(2, 2), name="res2_1", padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(8, (3, 3), strides=(1, 1), name="res2_2", padding="same")(X)
    X_shortcut2 = Conv2D(8, (3, 3), strides=(2, 2), name='res2_3', padding='same')(X_shortcut2)
    Res2 = Add()([X, X_shortcut2])
    X = BatchNormalization()(Res2)
    X = Activation('relu')(X)
    # ResNet3
    X_shortcut3 = X
    X = Conv2D(16, (3, 3), strides=(2, 2), name="res3_1", padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(16, (3, 3), strides=(1, 1), name="res3_2", padding="same")(X)
    X_shortcut3 = Conv2D(16, (3, 3), strides=(2, 2), name='res3_3', padding='same')(X_shortcut3)
    Res3 = Add()([X, X_shortcut3])
    X = BatchNormalization()(Res3)
    X = Activation('relu')(X)
    # Conv2
    X = Conv2D(32, (3, 3), strides=(2, 2), name='conv2')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    # Conv3
    X = Conv2D(32, (3, 3), strides=(2, 2), padding='same', name='conv3')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    # Conv4
    X = Conv2D(1, (3, 3), strides=(2, 2), name='conv4')(X)
    X = Activation('tanh')(X)
    X = Flatten()(X)
    model = Model(X_input, X)
    return model


def FCN2():
    X_input = Input((51, 51, 3))
    # Conv1
    X = Conv2D(4, (3, 3), strides=(1, 1), name="conv1", padding="same")(X_input)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    # ResNet1
    X_shortcut1 = X
    X = Conv2D(4, (3, 3), strides=(1, 1), name='res1_1', padding='same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(4, (3, 3), strides=(1, 1), name="res1_2", padding="same")(X)
    Res = Add()([X_shortcut1, X])
    X = BatchNormalization()(Res)
    X = Activation('relu')(X)
    # ResNet2
    X_shortcut2 = X
    X = Conv2D(8, (3, 3), strides=(2, 2), name="res2_1", padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(8, (3, 3), strides=(1, 1), name="res2_2", padding="same")(X)
    X_shortcut2 = Conv2D(8, (3, 3), strides=(2, 2), name='res2_3', padding='same')(X_shortcut2)
    Res2 = Add()([X, X_shortcut2])
    X = BatchNormalization()(Res2)
    X = Activation('relu')(X)
    # ResNet3
    X_shortcut3 = X
    X = Conv2D(16, (3, 3), strides=(2, 2), name="res3_1", padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(16, (3, 3), strides=(1, 1), name="res3_2", padding="same")(X)
    X_shortcut3 = Conv2D(16, (3, 3), strides=(2, 2), name='res3_3', padding='same')(X_shortcut3)
    Res3 = Add()([X, X_shortcut3])
    X = BatchNormalization()(Res3)
    X = Activation('relu')(X)

    # Conv2
    X = Conv2D(4096, (13, 13), strides=(1, 1), name='conv2')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    # Conv3
    X = Conv2D(1, (1, 1), strides=(1, 1), name='conv3')(X)
    X = Activation('tanh')(X)
    X = Flatten()(X)
    model = Model(X_input, X)
    return model


def FCN2_Xe():
    X_input = Input((51, 51, 3))
    # Conv1
    X = Conv2D(4, (3, 3), strides=(1, 1), name="conv1", padding="same", kernel_initializer=glorot_normal())(X_input)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    # ResNet1
    X_shortcut1 = X
    X = Conv2D(4, (3, 3), strides=(1, 1), name='res1_1', padding='same', kernel_initializer=glorot_normal())(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(4, (3, 3), strides=(1, 1), name="res1_2", padding="same", kernel_initializer=glorot_normal())(X)
    Res = Add()([X_shortcut1, X])
    X = BatchNormalization()(Res)
    X = Activation('relu')(X)
    # ResNet2
    X_shortcut2 = X
    X = Conv2D(8, (3, 3), strides=(2, 2), name="res2_1", padding="same", kernel_initializer=glorot_normal())(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(8, (3, 3), strides=(1, 1), name="res2_2", padding="same", kernel_initializer=glorot_normal())(X)
    X_shortcut2 = Conv2D(8, (3, 3), strides=(2, 2), name='res2_3', padding='same', kernel_initializer=glorot_normal())(
        X_shortcut2)
    Res2 = Add()([X, X_shortcut2])
    X = BatchNormalization()(Res2)
    X = Activation('relu')(X)
    # ResNet3
    X_shortcut3 = X
    X = Conv2D(16, (3, 3), strides=(2, 2), name="res3_1", padding="same", kernel_initializer=glorot_normal())(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(16, (3, 3), strides=(1, 1), name="res3_2", padding="same", kernel_initializer=glorot_normal())(X)
    X_shortcut3 = Conv2D(16, (3, 3), strides=(2, 2), name='res3_3', padding='same', kernel_initializer=glorot_normal())(
        X_shortcut3)
    Res3 = Add()([X, X_shortcut3])
    X = BatchNormalization()(Res3)
    X = Activation('relu')(X)

    # Conv2
    X = Conv2D(4096, (13, 13), strides=(1, 1), name='conv2', kernel_initializer=glorot_normal())(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    # Conv3
    X = Conv2D(1, (1, 1), strides=(1, 1), name='conv3', kernel_initializer=glorot_normal())(X)
    X = Activation('tanh')(X)
    X = Flatten()(X)
    model = Model(X_input, X)
    return model


def FCN2_He():
    X_input = Input((51, 51, 3))
    # Conv1
    X = Conv2D(4, (3, 3), strides=(1, 1), name="conv1", padding="same", kernel_initializer=he_normal())(X_input)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    # ResNet1
    X_shortcut1 = X
    X = Conv2D(4, (3, 3), strides=(1, 1), name='res1_1', padding='same', kernel_initializer=he_normal())(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(4, (3, 3), strides=(1, 1), name="res1_2", padding="same", kernel_initializer=he_normal())(X)
    Res = Add()([X_shortcut1, X])
    X = BatchNormalization()(Res)
    X = Activation('relu')(X)
    # ResNet2
    X_shortcut2 = X
    X = Conv2D(8, (3, 3), strides=(2, 2), name="res2_1", padding="same", kernel_initializer=he_normal())(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(8, (3, 3), strides=(1, 1), name="res2_2", padding="same", kernel_initializer=he_normal())(X)
    X_shortcut2 = Conv2D(8, (3, 3), strides=(2, 2), name='res2_3', padding='same', kernel_initializer=he_normal())(
        X_shortcut2)
    Res2 = Add()([X, X_shortcut2])
    X = BatchNormalization()(Res2)
    X = Activation('relu')(X)
    # ResNet3
    X_shortcut3 = X
    X = Conv2D(16, (3, 3), strides=(2, 2), name="res3_1", padding="same", kernel_initializer=he_normal())(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(16, (3, 3), strides=(1, 1), name="res3_2", padding="same", kernel_initializer=he_normal())(X)
    X_shortcut3 = Conv2D(16, (3, 3), strides=(2, 2), name='res3_3', padding='same', kernel_initializer=he_normal())(
        X_shortcut3)
    Res3 = Add()([X, X_shortcut3])
    X = BatchNormalization()(Res3)
    X = Activation('relu')(X)

    # Conv2
    X = Conv2D(4096, (13, 13), strides=(1, 1), name='conv2', kernel_initializer=he_normal())(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    # Conv3
    X = Conv2D(1, (1, 1), strides=(1, 1), name='conv3', kernel_initializer=he_normal())(X)
    X = Activation('tanh')(X)
    X = Flatten()(X)
    model = Model(X_input, X)
    return model


def FCN2_mix():
    X_input = Input((51, 51, 3))
    # Conv1
    X = Conv2D(4, (3, 3), strides=(1, 1), name="conv1", padding="same", kernel_initializer=he_normal())(X_input)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    # ResNet1
    X_shortcut1 = X
    X = Conv2D(4, (3, 3), strides=(1, 1), name='res1_1', padding='same', kernel_initializer=he_normal())(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(4, (3, 3), strides=(1, 1), name="res1_2", padding="same", kernel_initializer=he_normal())(X)
    Res = Add()([X_shortcut1, X])
    X = BatchNormalization()(Res)
    X = Activation('relu')(X)
    # ResNet2
    X_shortcut2 = X
    X = Conv2D(8, (3, 3), strides=(2, 2), name="res2_1", padding="same", kernel_initializer=he_normal())(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(8, (3, 3), strides=(1, 1), name="res2_2", padding="same", kernel_initializer=he_normal())(X)
    X_shortcut2 = Conv2D(8, (3, 3), strides=(2, 2), name='res2_3', padding='same', kernel_initializer=he_normal())(
        X_shortcut2)
    Res2 = Add()([X, X_shortcut2])
    X = BatchNormalization()(Res2)
    X = Activation('relu')(X)
    # ResNet3
    X_shortcut3 = X
    X = Conv2D(16, (3, 3), strides=(2, 2), name="res3_1", padding="same", kernel_initializer=he_normal())(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(16, (3, 3), strides=(1, 1), name="res3_2", padding="same", kernel_initializer=he_normal())(X)
    X_shortcut3 = Conv2D(16, (3, 3), strides=(2, 2), name='res3_3', padding='same', kernel_initializer=he_normal())(
        X_shortcut3)
    Res3 = Add()([X, X_shortcut3])
    X = BatchNormalization()(Res3)
    X = Activation('relu')(X)

    # Conv2
    X = Conv2D(4096, (13, 13), strides=(1, 1), name='conv2', kernel_initializer=he_normal())(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    # Conv3
    X = Conv2D(1, (1, 1), strides=(1, 1), name='conv3', kernel_initializer=glorot_normal())(X)
    X = Activation('tanh')(X)
    X = Flatten()(X)
    model = Model(X_input, X)
    return model


def Dense13_Xe():
    X_input = Input((51, 51, 3))
    Cropping = Cropping2D(cropping=((19, 19), (19, 19)))(X_input)
    # Conv1
    X = Conv2D(4, (3, 3), strides=(1, 1), name="conv1", padding="same", kernel_initializer=glorot_normal())(X_input)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    # ResNet1
    X_shortcut1 = X
    X = Conv2D(4, (3, 3), strides=(1, 1), name='res1_1', padding='same', kernel_initializer=glorot_normal())(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(4, (3, 3), strides=(1, 1), name="res1_2", padding="same", kernel_initializer=glorot_normal())(X)
    Res = Add()([X_shortcut1, X])
    X = BatchNormalization()(Res)
    X = Activation('relu')(X)
    # ResNet2
    X_shortcut2 = X
    X = Conv2D(8, (3, 3), strides=(2, 2), name="res2_1", padding="same", kernel_initializer=glorot_normal())(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(8, (3, 3), strides=(1, 1), name="res2_2", padding="same", kernel_initializer=glorot_normal())(X)
    X_shortcut2 = Conv2D(8, (3, 3), strides=(2, 2), name='res2_3', padding='same', kernel_initializer=glorot_normal())(
        X_shortcut2)
    Res2 = Add()([X, X_shortcut2])
    X = BatchNormalization()(Res2)
    X = Activation('relu')(X)
    # ResNet3
    X_shortcut3 = X
    X = Conv2D(16, (3, 3), strides=(2, 2), name="res3_1", padding="same", kernel_initializer=glorot_normal())(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(16, (3, 3), strides=(1, 1), name="res3_2", padding="same", kernel_initializer=glorot_normal())(X)
    X_shortcut3 = Conv2D(16, (3, 3), strides=(2, 2), name='res3_3', padding='same', kernel_initializer=glorot_normal())(
        X_shortcut3)
    Res3 = Add()([X, X_shortcut3])
    X = BatchNormalization()(Res3)
    X = Activation('relu')(X)
    # concatenate
    X = Concatenate()([X, Cropping])
    # Conv2
    X = Conv2D(4096, (13, 13), strides=(1, 1), name='conv2', kernel_initializer=glorot_normal())(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    # Conv3
    X = Conv2D(1, (1, 1), strides=(1, 1), name='conv3', kernel_initializer=glorot_normal())(X)
    X = Activation('tanh')(X)
    X = Flatten()(X)
    model = Model(X_input, X)
    return model


def Dense13_He():
    X_input = Input((51, 51, 3))
    Cropping = Cropping2D(cropping=((19, 19), (19, 19)))(X_input)
    # Conv1
    X = Conv2D(4, (3, 3), strides=(1, 1), name="conv1", padding="same", kernel_initializer=he_normal())(X_input)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    # ResNet1
    X_shortcut1 = X
    X = Conv2D(4, (3, 3), strides=(1, 1), name='res1_1', padding='same', kernel_initializer=he_normal())(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(4, (3, 3), strides=(1, 1), name="res1_2", padding="same", kernel_initializer=he_normal())(X)
    Res = Add()([X_shortcut1, X])
    X = BatchNormalization()(Res)
    X = Activation('relu')(X)
    # ResNet2
    X_shortcut2 = X
    X = Conv2D(8, (3, 3), strides=(2, 2), name="res2_1", padding="same", kernel_initializer=he_normal())(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(8, (3, 3), strides=(1, 1), name="res2_2", padding="same", kernel_initializer=he_normal())(X)
    X_shortcut2 = Conv2D(8, (3, 3), strides=(2, 2), name='res2_3', padding='same', kernel_initializer=he_normal())(
        X_shortcut2)
    Res2 = Add()([X, X_shortcut2])
    X = BatchNormalization()(Res2)
    X = Activation('relu')(X)
    # ResNet3
    X_shortcut3 = X
    X = Conv2D(16, (3, 3), strides=(2, 2), name="res3_1", padding="same", kernel_initializer=he_normal())(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(16, (3, 3), strides=(1, 1), name="res3_2", padding="same", kernel_initializer=he_normal())(X)
    X_shortcut3 = Conv2D(16, (3, 3), strides=(2, 2), name='res3_3', padding='same', kernel_initializer=he_normal())(
        X_shortcut3)
    Res3 = Add()([X, X_shortcut3])
    X = BatchNormalization()(Res3)
    X = Activation('relu')(X)
    # concatenate
    X = Concatenate()([X, Cropping])
    # Conv2
    X = Conv2D(4096, (13, 13), strides=(1, 1), name='conv2', kernel_initializer=he_normal())(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    # Conv3
    X = Conv2D(1, (1, 1), strides=(1, 1), name='conv3', kernel_initializer=he_normal())(X)
    X = Activation('tanh')(X)
    X = Flatten()(X)
    model = Model(X_input, X)
    return model


def Dense1_mix():
    X_input = Input((51, 51, 3))
    Cropping = Cropping2D(cropping=((25, 25), (25, 25)))(X_input)
    # Conv1
    X = Conv2D(4, (3, 3), strides=(1, 1), name="conv1", padding="same", kernel_initializer=he_normal())(X_input)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    # ResNet1
    X_shortcut1 = X
    X = Conv2D(4, (3, 3), strides=(1, 1), name='res1_1', padding='same', kernel_initializer=he_normal())(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(4, (3, 3), strides=(1, 1), name="res1_2", padding="same", kernel_initializer=he_normal())(X)
    Res = Add()([X_shortcut1, X])
    X = BatchNormalization()(Res)
    X = Activation('relu')(X)
    # ResNet2
    X_shortcut2 = X
    X = Conv2D(8, (3, 3), strides=(2, 2), name="res2_1", padding="same", kernel_initializer=he_normal())(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(8, (3, 3), strides=(1, 1), name="res2_2", padding="same", kernel_initializer=he_normal())(X)
    X_shortcut2 = Conv2D(8, (3, 3), strides=(2, 2), name='res2_3', padding='same', kernel_initializer=he_normal())(
        X_shortcut2)
    Res2 = Add()([X, X_shortcut2])
    X = BatchNormalization()(Res2)
    X = Activation('relu')(X)
    # ResNet3
    X_shortcut3 = X
    X = Conv2D(16, (3, 3), strides=(2, 2), name="res3_1", padding="same", kernel_initializer=he_normal())(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(16, (3, 3), strides=(1, 1), name="res3_2", padding="same", kernel_initializer=he_normal())(X)
    X_shortcut3 = Conv2D(16, (3, 3), strides=(2, 2), name='res3_3', padding='same', kernel_initializer=he_normal())(
        X_shortcut3)
    Res3 = Add()([X, X_shortcut3])
    X = BatchNormalization()(Res3)
    X = Activation('relu')(X)
    # Conv2
    X = Conv2D(4096, (13, 13), strides=(1, 1), name='conv2', kernel_initializer=he_normal())(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    # concatenate
    X = Concatenate()([X, Cropping])
    # Conv3
    X = Conv2D(1, (1, 1), strides=(1, 1), name='conv3', kernel_initializer=glorot_normal())(X)
    # X = BatchNormalization()(X)
    X = Activation('tanh')(X)
    X = Flatten()(X)
    model = Model(X_input, X)
    return model


def Dense13_mix():
    X_input = Input((51, 51, 3))
    Cropping = Cropping2D(cropping=((19, 19), (19, 19)))(X_input)
    # Conv1
    X = Conv2D(4, (3, 3), strides=(1, 1), name="conv1", padding="same", kernel_initializer=he_normal())(X_input)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    # ResNet1
    X_shortcut1 = X
    X = Conv2D(4, (3, 3), strides=(1, 1), name='res1_1', padding='same', kernel_initializer=he_normal())(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(4, (3, 3), strides=(1, 1), name="res1_2", padding="same", kernel_initializer=he_normal())(X)
    Res = Add()([X_shortcut1, X])
    X = BatchNormalization()(Res)
    X = Activation('relu')(X)
    # ResNet2
    X_shortcut2 = X
    X = Conv2D(8, (3, 3), strides=(2, 2), name="res2_1", padding="same", kernel_initializer=he_normal())(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(8, (3, 3), strides=(1, 1), name="res2_2", padding="same", kernel_initializer=he_normal())(X)
    X_shortcut2 = Conv2D(8, (3, 3), strides=(2, 2), name='res2_3', padding='same', kernel_initializer=he_normal())(
        X_shortcut2)
    Res2 = Add()([X, X_shortcut2])
    X = BatchNormalization()(Res2)
    X = Activation('relu')(X)
    # ResNet3
    X_shortcut3 = X
    X = Conv2D(16, (3, 3), strides=(2, 2), name="res3_1", padding="same", kernel_initializer=he_normal())(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(16, (3, 3), strides=(1, 1), name="res3_2", padding="same", kernel_initializer=he_normal())(X)
    X_shortcut3 = Conv2D(16, (3, 3), strides=(2, 2), name='res3_3', padding='same', kernel_initializer=he_normal())(
        X_shortcut3)
    Res3 = Add()([X, X_shortcut3])
    X = BatchNormalization()(Res3)
    X = Activation('relu')(X)
    # concatenate
    X = Concatenate()([X, Cropping])
    # Conv2
    X = Conv2D(4096, (13, 13), strides=(1, 1), name='conv2', kernel_initializer=he_normal())(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    # Conv3
    X = Conv2D(1, (1, 1), strides=(1, 1), name='conv3', kernel_initializer=glorot_normal())(X)
    # X = BatchNormalization()(X)
    X = Activation('tanh')(X)
    X = Flatten()(X)
    model = Model(X_input, X)
    return model


def Dense1_relu():
    X_input = Input((51, 51, 3))
    Cropping = Cropping2D(cropping=((25, 25), (25, 25)))(X_input)
    # Conv1
    X = Conv2D(4, (3, 3), strides=(1, 1), name="conv1", padding="same", kernel_initializer=he_normal())(X_input)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    # ResNet1
    X_shortcut1 = X
    X = Conv2D(4, (3, 3), strides=(1, 1), name='res1_1', padding='same', kernel_initializer=he_normal())(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(4, (3, 3), strides=(1, 1), name="res1_2", padding="same", kernel_initializer=he_normal())(X)
    Res = Add()([X_shortcut1, X])
    X = BatchNormalization()(Res)
    X = Activation('relu')(X)
    # ResNet2
    X_shortcut2 = X
    X = Conv2D(8, (3, 3), strides=(2, 2), name="res2_1", padding="same", kernel_initializer=he_normal())(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(8, (3, 3), strides=(1, 1), name="res2_2", padding="same", kernel_initializer=he_normal())(X)
    X_shortcut2 = Conv2D(8, (3, 3), strides=(2, 2), name='res2_3', padding='same', kernel_initializer=he_normal())(
        X_shortcut2)
    Res2 = Add()([X, X_shortcut2])
    X = BatchNormalization()(Res2)
    X = Activation('relu')(X)
    # ResNet3
    X_shortcut3 = X
    X = Conv2D(16, (3, 3), strides=(2, 2), name="res3_1", padding="same", kernel_initializer=he_normal())(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(16, (3, 3), strides=(1, 1), name="res3_2", padding="same", kernel_initializer=he_normal())(X)
    X_shortcut3 = Conv2D(16, (3, 3), strides=(2, 2), name='res3_3', padding='same', kernel_initializer=he_normal())(
        X_shortcut3)
    Res3 = Add()([X, X_shortcut3])
    X = BatchNormalization()(Res3)
    X = Activation('relu')(X)
    # Conv2
    X = Conv2D(4096, (13, 13), strides=(1, 1), name='conv2', kernel_initializer=he_normal())(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    # concatenate
    X = Concatenate()([X, Cropping])
    # Conv3
    X = Conv2D(1, (1, 1), strides=(1, 1), name='conv3', kernel_initializer=he_normal())(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Flatten()(X)
    model = Model(X_input, X)
    return model
