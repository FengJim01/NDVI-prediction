import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.metrics import r2_score
import scipy.stats as state
import os
from PIL import Image
from mpl_toolkits.axes_grid1 import make_axes_locatable

if __name__ == "__main__":
    # read the data
    # building\farm\grass\river\road\tree
    rgb_path = os.path.join('.', 'dataset', 'testing', 'single', 'test_15meter_single_tree_RGB.npy')
    ndvi_path = os.path.join('.', 'dataset', 'testing', 'single', 'test_15meter_single_tree_NDVI.npy')

    rgb_img = np.load(rgb_path)
    NDVI = np.load(ndvi_path)

    red = rgb_img[:, :, 0].astype('int16')
    green = rgb_img[:, :, 1].astype('int16')
    blue = rgb_img[:, :, 2].astype('int16')

    # Visible VIs
    # GRVI
    GRVI = np.true_divide((green - red), (red + green))
    GRVI[np.isnan(GRVI)] = -1
    GRVI[np.isinf(GRVI)] = 1
    GRVI_r2 = r2_score(NDVI, GRVI)
    GRVI_corr = state.pearsonr(GRVI.flatten(), NDVI.flatten())
    print("GRVI correlation = %s" % GRVI_corr[0])
    print("GRVI R2= %f" % GRVI_r2)
    # ExG
    ExG = red + 2 * green + blue
    ExG_r2 = r2_score(NDVI, ExG)
    ExG_corr = state.pearsonr(ExG.flatten(), NDVI.flatten())
    print("ExG correlation = %s" % ExG_corr[0])
    print("ExG R2 = %f" % ExG_r2)
    # TGI
    TGI = red * (-0.39) + green - 0.61 * blue
    TGI_r2 = r2_score(NDVI, TGI)
    TGI_corr = state.pearsonr(TGI.flatten(), NDVI.flatten())
    print("TGI correlation = %s" % TGI_corr[0])
    print("TGI R2 = %f" % TGI_r2)
    # VARI
    VARI = np.true_divide((green - red), (green + red - blue))
    VARI[np.isnan(VARI)] = -255
    VARI[np.isinf(VARI)] = 255
    # VARI[VARI > 1] = 1
    # VARI[VARI < 1] = -1
    VARI_r2 = r2_score(NDVI, VARI)
    VARI_corr = state.pearsonr(VARI.flatten(), NDVI.flatten())
    print("VARI correlation = %s" % VARI_corr[0])
    print("VARI R2 = %f" % VARI_r2)

    # color map
    fig = plt.figure()
    cm = plt.cm.get_cmap('jet')
    plt.rcParams['font.sans-serif'] = ["DFKai-SB"]
    plt.rcParams['axes.unicode_minus'] = False
    # GRVI
    GRVI_plot = fig.add_subplot(233)
    GRVI_plot.axis('off')
    divider = make_axes_locatable(GRVI_plot)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cb1 = GRVI_plot.imshow(GRVI, cmap=cm)
    colorbar = plt.colorbar(cb1, cax=cax, orientation='vertical')
    colorbar.ax.tick_params(labelsize=20)
    # NDVI
    NDVI_plot = fig.add_subplot(232)
    NDVI_plot.axis('off')
    divider = make_axes_locatable(NDVI_plot)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cb2 = NDVI_plot.imshow(NDVI, cmap=cm)
    colorbar2 = plt.colorbar(cb2, cax=cax, orientation='vertical')
    colorbar2.ax.tick_params(labelsize=20)
    # ExG
    ExG_plot = fig.add_subplot(234)
    ExG_plot.axis('off')
    divider = make_axes_locatable(ExG_plot)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cb3 = ExG_plot.imshow(ExG, cmap=cm)
    colorbar3 = plt.colorbar(cb3, cax=cax, orientation='vertical')
    colorbar3.ax.tick_params(labelsize=20)
    # TGI
    TGI_plot = fig.add_subplot(235)
    TGI_plot.axis('off')
    divider = make_axes_locatable(TGI_plot)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cb4 = TGI_plot.imshow(TGI, cmap=cm)
    colorbar4 = plt.colorbar(cb4, cax=cax, orientation='vertical')
    colorbar4.ax.tick_params(labelsize=20)
    # VARI
    VARI_plot = fig.add_subplot(236)
    VARI_plot.axis('off')
    divider = make_axes_locatable(VARI_plot)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cb5 = VARI_plot.imshow(VARI, cmap=cm)
    colorbar5 = plt.colorbar(cb5, cax=cax, orientation='vertical')
    colorbar5.ax.tick_params(labelsize=20)
    # VARI
    original_plot = fig.add_subplot(231)
    original_plot.axis('off')
    original_plot.imshow(rgb_img)
    # scatter=fig.add_subplot(111)
    # scatter.scatter(GRVI,NDVI,s=2)
    plt.show()
