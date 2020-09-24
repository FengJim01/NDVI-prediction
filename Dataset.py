import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os, glob
import random


class dataset():
    def __init__(self):
        self.image_path = "15meter"  # 15meter/20meter/10meter/2meter 拍攝高度
        self.classes = "building"  # building/grass/farm/river/road/tree/all 選用的地形影像
        self.mode = 'test'  # train/test 訓練用/測試用
        self.photo = 'single'  # multiple/single 張數設定

        self.make_data()

    def make_data(self):
        image_size = (1280, 960, 3)  # 設定影像大小
        if self.classes == "all":
            image_path = os.path.join(".", "dataset", "Image", self.image_path, "class")  # 設定影像"相對位置"
            # new_size = self.resize(image_size, scale=1)
            if self.photo == "single":  # 因使用所有地形影像的方法不會使用單張影像的預測
                print("Could not use the single mode, Fuck You")
                pass
            else:  # multiple mode
                RGB_save = []
                NDVI_save = []
                for classes in os.listdir(image_path):
                    for ids in os.listdir(os.path.join(image_path, classes)):
                        print("Class : ", classes, "ID : ", ids)

                        image_RGB = os.path.join(image_path, classes, ids, "GSD_RGB.tiff")  # id_rgb.png #GlobalReg.tif
                        image_NIR = os.path.join(image_path, classes, ids, "GSD_nir.tiff")
                        # id_nir.png # "IMG_%s_4toIMG_%s_2_registered.tif" % (ids,ids)
                        # image resampling(path -> Image)
                        img_RGB, img_NIR = self.resampling(image_RGB, image_NIR, image_size)
                        img_NDVI = self.NDVI_make(img_RGB, img_NIR)  # make the NDVI image(image -> array)
                        # input the image array to the list
                        RGB_save.append(np.asarray(img_RGB))
                        NDVI_save.append(img_NDVI)
                np.save("./%s_%s_RGB.npy" % (self.mode, self.image_path), np.asarray(RGB_save))
                np.save("./%s_%s_NDVI.npy" % (self.mode, self.image_path), np.asarray(NDVI_save))
                RGB_save.clear()
                NDVI_save.clear()
        else:#single landcover
            image_path = os.path.join(".", "photo", self.image_path, "class", self.classes)
            ids = os.listdir(image_path)
            # new_size = self.resize(image_size, scale=1.)
            if self.photo == 'single':
                #choose one image randomly
                random.seed(2)
                random_id = random.choice(ids)
                print("Choose Image : %s" % random_id)
                image_RGB = os.path.join(image_path, random_id, "GSD_RGB.tiff")  # id_rgb.png
                image_NIR = os.path.join(image_path, random_id, "GSD_nir.tiff")  # id_nir.png
                img_RGB, img_NIR = self.resampling(image_RGB, image_NIR, image_size)
                img_NDVI = self.NDVI_make(img_RGB, img_NIR)
                np.save("./%s_%s_single_%s_RGB.npy" % (self.mode, self.image_path, self.classes), img_RGB)
                np.save("./%s_%s_single_%s_NDVI.npy" % (self.mode, self.image_path, self.classes), img_NDVI)
            else: #multiple
                rgb_save = []
                ndvi_save = []
                for id in ids:
                    print(self.classes, id)
                    image_RGB = os.path.join(image_path, id, "GSD_RGB.tiff")  # id_rgb.png
                    image_NIR = os.path.join(image_path, id, "GSD_nir.tiff")  # id_nir.png
                    img_RGB, img_NIR = self.resampling(image_RGB, image_NIR, image_size)
                    img_NDVI = self.NDVI_make(img_RGB, img_NIR)
                    rgb_save.append(np.asarray(img_RGB))
                    ndvi_save.append(img_NDVI)
                np.save("./%s_%s_multiple_%s_RGB.npy" % (self.mode, self.image_path, self.classes),
                        np.asarray(rgb_save))
                np.save("./%s_%s_multiple_%s_NDVI.npy" % (self.mode, self.image_path, self.classes),
                        np.asarray(ndvi_save))
                rgb_save.clear()
                ndvi_save.clear()

    def resampling(self, image_path, nir_path, size):
        """
        resmapling the image
        :param image_path: RGB image path
        :param nir_path: NIR image path
        :param size: the scale of image after resampling
        :return: the array of RGB image and NIR image
        """
        RGB_img = Image.open(image_path)
        NIR_img = Image.open(nir_path)
        RGB_resampling = RGB_img.resize((size[0], size[1]), Image.CUBIC)
        NIR_resampling = NIR_img.resize((size[0], size[1]), Image.CUBIC)
        return RGB_resampling, NIR_resampling

    def NDVI_make(self, image, NIR):
        image = np.asarray(image)
        # NIR = np.asarray(NIR) / 256.
        nir = np.asarray(NIR, dtype='float')
        red = image[:, :, 0].astype('int16')
        if np.max(nir) > 255:
            nir = np.asarray(NIR, dtype='float') / 65535 * 255

        assert np.max(nir) <= 255, "NIR pixel value 大於255(%f)" % (np.max(nir))
        assert np.min(nir) >= 0, "NIR pixel value 小於0(%f)" % (np.min(nir))
        assert np.max(red) <= 255, "RED pixel value 大於255(%f)" % (np.max(red))
        assert np.min(red) >= 0, "RED pixel value 小於0(%f)" % (np.min(red))
        NDVI = np.true_divide(nir - red, red + nir)
        NDVI[np.isnan(NDVI)] = -1
        NDVI[np.isinf(NDVI)] = 1
        assert np.max(NDVI) <= 1, "NDVI pixel value 大於1(%f)" % (np.max(NDVI))
        assert np.min(NDVI) >= -1, "NDVI pixel value 小於-1(%f)" % (np.min(NDVI))
        return NDVI

    # def resize(self, size, scale=1.):
    #     width = int(size[0] * scale)
    #     height = int(size[1] * scale)
    #     new_size = (width, height, size[2])
    #     return new_size


if __name__ == "__main__":
    dataset()
