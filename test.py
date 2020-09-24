import math
import os
from PIL import Image
import numpy as np
import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Input, Activation, Add, Concatenate
from keras.models import Model, load_model
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.visualization import LogStretch, LinearStretch
import mpl_scatter_density
from model import *
from keras import optimizers
from train import datasetclip
from sklearn.metrics import r2_score
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.stats as state


def single_draw_picture(NDVI, predict, RGB):
    # make a window to put the figure
    fig = plt.figure(figsize=(20, 10))
    plt.rcParams['font.sans-serif'] = ["DFKai-SB"]
    plt.rcParams['axes.unicode_minus'] = False
    # fig.suptitle("Prediction")

    # Resize
    image_shape = (RGB.shape[1], RGB.shape[0])
    NDVI1 = np.reshape(NDVI, image_shape).T
    predict1 = np.reshape(predict, image_shape).T
    # calculate the error map
    error = np.abs(NDVI1 - predict1)
    # normalization
    # norm = plt.Normalize(vmin=-1., vmax=1.)
    # ground_truth_jet = plt.cm.jet(norm(NDVI1))  # normalize ground truth image
    # prediction_jet = plt.cm.jet(norm(predict1))  # normalize prediction image
    norm1 = ImageNormalize(vmin=0, vmax=2, stretch=LinearStretch())  # normalize Error map
    norm = ImageNormalize(vmin=-1., vmax=1.)
    # paint the figure on the window
    # original image
    original = fig.add_subplot(1, 3, 1)
    original.set_title("原始影像", size=20)
    original.axis('off')
    original.imshow(RGB)
    # Ground Truth
    ground_truth = fig.add_subplot(1, 3, 2)
    ground_truth.set_title("地真影像 (NDVI)", size=20)
    ground_truth.axis('off')
    # divider = make_axes_locatable(ground_truth)
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    cm = plt.cm.get_cmap('jet')
    cm.set_under("1.")
    cm.set_over("-1.")
    ground_truth.imshow(NDVI1, cmap=cm, norm=norm)
    # fig.colorbar(color, cax=cax, orientation='vertical')
    # prediction
    prediction = fig.add_subplot(1, 3, 3)
    prediction.set_title("預測影像", size=20)
    prediction.axis('off')
    divider = make_axes_locatable(prediction)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cm1 = plt.cm.get_cmap('jet')
    cm1.set_under("1.")
    cm1.set_over("-1.")
    color2 = prediction.imshow(predict1, cmap=cm1, norm=norm)
    fig.colorbar(color2, cax=cax, orientation='vertical')
    # # Error map
    # error_map = fig.add_subplot(2, 2, 4)
    # error_map.axis('off')
    # error_map.title.set_text("殘差影像")
    # divider = make_axes_locatable(error_map)
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # color3 = error_map.imshow(error, cmap=plt.cm.Reds, norm=norm1)
    # fig.colorbar(color3, cax=cax, orientation='vertical')

    # density map
    # fig2 = plt.figure(figsize=(12, 10))
    # norm2 = ImageNormalize(vmin=0, vmax=100, stretch=LogStretch())
    # distribution = fig2.add_subplot(111, projection='scatter_density')
    # # distribution.set_title("點密度圖", size=20)
    # density=distribution.scatter_density(predict, NDVI, cmap=plt.cm.Blues, norm=norm2)
    # distribution.plot([-1, 1], [-1, 1], 'black')
    # distribution.set_xlim(-1, 1)
    # distribution.set_ylim(-1, 1)
    # distribution.set_xticklabels([])
    # distribution.set_yticklabels([])
    # colorbar = plt.colorbar(density)
    # colorbar.set_ticks(np.linspace(0, 100, 5))
    # colorbar.set_ticklabels([0, '', 50, '', 100])
    # colorbar.ax.tick_params(labelsize=50)

    # save and show
    # plt.show()
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    fig.savefig("./result1.png")
    # fig2.savefig("./result2.png")


def muti_draw_picture(NDVI, predict):
    # Density Map
    # set the Chinese font type
    plt.rcParams['font.sans-serif'] = ["DFKai-SB"]
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure(figsize=(12, 10))  # set the figure and the size
    density_map = fig.add_subplot(111, projection='scatter_density')  # Add a subplot in the figure
    # density_map.set_title("點密度圖",size=20) #set the title
    normalize = ImageNormalize(vmin=0, vmax=100, stretch=LogStretch())
    density = density_map.scatter_density(predict, NDVI, cmap=plt.cm.Blues, norm=normalize)
    density_map.plot([-1, 1], [-1, 1], 'black')
    density_map.set_xlim(-1, 1)
    density_map.set_ylim(-1, 1)
    density_map.set_xticklabels([])
    density_map.set_yticklabels([])
    # divider = make_axes_locatable(density_map)
    # cax = divider.append_axes('right', size='5%', pad=0.1)
    colorbar = plt.colorbar(density)  # , cax=cax, orientation='vertical')
    colorbar.set_ticks(np.linspace(0, 100, 5))
    colorbar.set_ticklabels([0, '', 50, '', 100])
    colorbar.ax.tick_params(labelsize=50)
    fig.savefig("./result2.png")
    # plt.show()


def crop(fullimage, edge):
    """
    去除周邊edge長度的範圍。
    :param fullimage:the path of the image
    :param edge:去除的長度
    :return: the array of image after cropping
    """
    fullimg = np.load(fullimage)
    edge_length = int((edge - 1) / 2) + 50
    fullimg1 = fullimg[edge_length:-1 * edge_length - 1, edge_length:-1 * edge_length - 1, :]
    return fullimg1


def loss_function(loss, val_loss, rgb, ndvi):
    """
    模型在每個迭代的loss function走向
    :param loss: 模型訓練時的loss function
    :param val_loss: 模型驗證資料的loss function
    :param rgb: 測試的RGB影像資料
    :param ndvi: 測試的NDVI影像資料
    """
    fig2 = plt.figure(figsize=(20, 11))
    # fig2.suptitle("損失函式折線圖", fontsize=40)
    ax = plt.axes()
    model_x = np.linspace(1, loss.size, num=loss.size)
    ax.plot(model_x, np.squeeze(loss), 'r', lw=5, label="訓練資料損失函式")
    ax.scatter(model_x, np.squeeze(loss), c='r', s=70)
    test_loss = np.load("./model/17/test_loss.npy")
    test_x = np.linspace(1, test_loss.size, num=test_loss.size)
    ax.plot(test_x, np.squeeze(test_loss), 'b', lw=5, label="測試資料損失函式")
    ax.scatter(test_x, np.squeeze(test_loss), c='b', s=70)

    # label
    ax.legend(loc='upper right', fontsize=40)
    # ax.set_xlabel('迭代', fontsize=50)
    # ax.set_ylabel("MSE", fontsize=50)
    # plt.xlim([0, 20])
    plt.xticks(np.arange(1, 21, 2), fontsize=40)
    plt.yticks(fontsize=40)
    plt.savefig("./result4.png")
    # plt.show()


if __name__ == "__main__":
    # read the data
    # building\farm\grass\river\road\tree
    rgb_path = os.path.join('.', 'dataset', 'testing', 'test_15meter_RGB.npy')
    ndvi_path = os.path.join('.', 'dataset', 'testing', 'test_15meter_NDVI.npy')
    # rgb_path = os.path.join('.', 'dataset', 'testing', 'single', 'test_15meter_single_tree_RGB.npy')
    # ndvi_path = os.path.join('.', 'dataset', 'testing', 'single', 'test_15meter_single_tree_NDVI.npy')

    # import model
    model = FCN2_mix()  # 選擇與訓練時相同的model
    adam = optimizers.Adam(lr=0.0001) #Optimizer
    model.compile(loss='mean_squared_error', optimizer=adam, metrics=['acc'])
    # load the weights
    weight = os.path.join(".", "trained_model.h5")
    model.load_weights(weight)
    # load the loss amd validation loss
    loss = np.load(os.path.join(model_path, "loss_function.npy"))
    # val_loss = np.load(os.path.join(model_path, 'val_loss.npy'))

    # make the dataset
    # if the dataset includes only one image, the shape of image will be (1280,960,3)
    # the input of the function should be 4 dimension, i.e. (x,1280,960,3)
    rgb = datasetclip(rgb_path, 51, 20, 20, 2, mode=1)
    ndvi = datasetclip(ndvi_path, 51, 20, 20, 2, mode=2)
    fullimage = crop(rgb_path, 51)

    predict = np.squeeze(model.predict(rgb))
    lossfunc = model.evaluate(rgb, ndvi)
    assert predict.shape == ndvi.shape, "Prediction維度和NDVI不同"

    rmse = math.sqrt(np.mean(np.square(ndvi - predict)))
    r2 = r2_score(ndvi, predict)
    correlation = state.pearsonr(ndvi, predict)
    print("Final RMSE : %f" % rmse)
    print("Final R Square = %f" % r2)
    print("Final Correlation = %s" % correlation[0])
    # print("Final Loss : %f" % lossfunc[0])

    #看dataset是用哪種，loss_function()是觀看模型在迭代的loss function走向，single_draw_picture()是觀看單一影像的成果，
    #muti_draw_picture是觀看多張影像的成果
    # muti_draw_picture(ndvi, predict)
    # single_draw_picture(ndvi, predict, fullimage)
    # loss_function(loss, val_loss, rgb, ndvi)
