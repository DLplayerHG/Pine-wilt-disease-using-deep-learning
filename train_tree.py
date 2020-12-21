import numpy as np
import tensorflow as tf
import cv2
from tensorflow.python import pywrap_tensorflow
import xml.etree.ElementTree as ET
import scipy
import random
import micasense.plotutils as plotutils
import micasense.metadata as metadata
import micasense.sequoiautils as msutils
import matplotlib.pyplot as plt


import lib.config.config as cfg
from lib.nets.vgg16 import vgg16

try:
    import cPickle as pickle
except ImportError:
    import pickle
import os, glob


figsize = (30, 23)
train_loss_txt = open('train_loss.txt', 'w')
val_loss_txt = open('validation_loss.txt', 'w')

Main_path = os.path.join('.', 'data', 'Tree_dji')
Training_path = os.path.join(Main_path, 'training')
Vaildation_path = os.path.join(Main_path, 'validation')

image_path_RGB = os.path.join(Training_path, 'RGB_Images(1)')
image_path_Red = os.path.join(Training_path, 'Red_Images(1)')
image_path_NIR = os.path.join(Training_path, 'NIR_Images(1)')
image_path_Green = os.path.join(Training_path, 'Green_Images(1)')
image_path_Red_Edge = os.path.join(Training_path, 'Red_Edge_Images(1)')


annotation_path = os.path.join(Training_path, 'Annotations')
validation_annotation_path = os.path.join(Vaildation_path, 'Annotations')


imageNames_RGB = glob.glob(os.path.join(image_path_RGB, '*.jpg'))
imageNames_Red = glob.glob(os.path.join(image_path_Red, '*.jpg'))
imageNames_NIR = glob.glob(os.path.join(image_path_NIR, '*.jpg'))
imageNames_Green = glob.glob(os.path.join(image_path_Green, '*.jpg'))
imageNames_Red_Edge = glob.glob(os.path.join(image_path_Red_Edge, '*.jpg'))


annotation = glob.glob(os.path.join(annotation_path, '*.xml'))
validation_annotation_1 = glob.glob(os.path.join(validation_annotation_path, '*.xml'))

imageNames_RGB.sort()
annotation.sort()



num_classes = 3
classes = ('__background__',  # always index 0
            'dead', 'browning')
class_to_ind = dict(list(zip(classes, list(range(num_classes)))))




def create_feed_dict(data_index):
    # create feed dict
    current_image_RGB = cv2.imread(imageNames_RGB[data_index])
    current_image_Red = cv2.imread(imageNames_Red[data_index])
    current_image_NIR = cv2.imread(imageNames_NIR[data_index])
    current_image_Red_Edge = cv2.imread(imageNames_Red_Edge[data_index])
    current_image_Green = cv2.imread(imageNames_Green[data_index])

    current_image_green, _, _ = cv2.split(current_image_Green)
    current_image_red, _, _ = cv2.split(current_image_Red)
    current_image_nir, _, _ = cv2.split(current_image_NIR)
    current_image_red_edge,_, _ = cv2.split(current_image_Red_Edge)

    channel_marge = cv2.merge([current_image_red, current_image_nir, current_image_red_edge])
    channel_6 = np.concatenate((channel_marge, current_image_RGB), axis=2)


    ndvi = (current_image_nir - current_image_red) / (current_image_nir + current_image_red)


    VIgreen = (current_image_green - current_image_red) / (current_image_green + current_image_red)

    gndvi = (current_image_nir - current_image_green) / (current_image_nir + current_image_green)
    ndre = (current_image_nir - current_image_red_edge) / (current_image_nir + current_image_red_edge)
    savi = ((current_image_nir - current_image_red) / (current_image_nir + current_image_red) + 0.5) * 1.5

    data_index = str(data_index)

    masked_ndvi = np.ma.masked_outside(ndvi, 0.45, 1.2)
    fig, axis = plotutils.plotwithcolorbar(masked_ndvi, 'ndvi', figsize=figsize)
    plt.draw()
    fig.savefig(Training_path + data_index + '_ndvi.png')
    plt.close(fig)

    masked_vigreen = np.ma.masked_outside(VIgreen, 0.0, 0.2)
    fig, axis = plotutils.plotwithcolorbar(masked_vigreen, 'vi_green', figsize=figsize)
    plt.draw()
    fig.savefig(Training_path + data_index + '_vigreen.png')
    plt.close(fig)

    masked_gndvi = np.ma.masked_outside(gndvi, -1, 1)
    fig, axis = plotutils.plotwithcolorbar(masked_gndvi, 'gndvi', figsize=figsize)
    plt.draw()
    fig.savefig(Training_path + data_index + '_gndvi.png')
    plt.close(fig)

    masked_ndre = np.ma.masked_outside(ndre, -1, 1)
    fig, axis = plotutils.plotwithcolorbar(masked_ndre, 'ndre', figsize=figsize)
    plt.draw()
    fig.savefig(Training_path + data_index + '_ndre.png')
    plt.close(fig)

    masked_savi = np.ma.masked_outside(savi, -1, 1)
    fig, axis = plotutils.plotwithcolorbar(masked_savi, 'savi', figsize=figsize)
    plt.draw()
    fig.savefig(Training_path + data_index + '_savi.png')
    plt.close(fig)




class Train:
    def __init__(self):

        # Create network
        if cfg.FLAGS.network == 'vgg16':
            self.net = vgg16(batch_size=cfg.FLAGS.ims_per_batch)
        else:
            raise NotImplementedError

        self.output_dir = "output\\"
        
    def train(self):

        for epoch in range((int(cfg.FLAGS.max_epochs))):
            # Learning rate
            # Get training data, one batch at a time

            shuffle_idx = np.random.permutation(len(imageNames_RGB))

            print('==================Start new epoch %d =================='%(epoch))

            for idx in shuffle_idx:

                create_feed_dict(idx)
                #print("idx:", idx)
                #print(blobs)
                # Compute the graph without summary




if __name__ == '__main__':
    train = Train()
    train.train()
    
