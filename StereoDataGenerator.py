import tensorflow as tf
import numpy as np
import math
import cv2 as cv
import re
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage
from Augmentation import Augmenter
from tensorflow.keras.utils import Sequence

class StereoDataGenerator(Sequence):
    """description of class"""

    def __init__(self, filenames, shuffle, batch_size, resize_only, seqs, im_dim, output_dim):
        self.filenames = filenames
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.resize_only = resize_only
        self.seqs = seqs
        self.im_dim = im_dim
        self.output_dim = output_dim
        self.anchor_wire_length = 0.053123636693654344 # 0.11 0.046552#
        self.anchor_disparity = 0.10034577835696407 # 0.11 	0.132489247 #	

        self.batch_idx = []

    # Inverse sigmoid
    def inv_sig(self,y):
        return math.log(y/(1-y))

    # Inverse exponential
    def inv_exp(self,value,anchor):
        return math.log(value/anchor)
        
    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.filenames)

    def __len__(self):
        # Denotes the number of batches per epoch
        return len(self.filenames) // self.batch_size

    def __getitem__(self, batch_index):  # batch index
        # Generate one batch of data
        selected_filenames = self.filenames[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]
        self.batch_idx.append(batch_index)

        # Generate data
        left_images, right_images, labels = self.get_data(selected_filenames)

        return [left_images, right_images],  [ labels[:,:,:,0:1], labels[:,:,:,1:7]]


    def get_data(self, selected_filenames):
        # allocate
        left_images = np.zeros((self.batch_size, self.im_dim, self.im_dim, 3), dtype=np.uint8)
        right_images = np.zeros((self.batch_size, self.im_dim, self.im_dim, 3), dtype=np.uint8)
        labels = np.zeros((self.batch_size,self.output_dim,self.output_dim,7))

        # loop over images within one batch
        for i in range(len(selected_filenames)):
            # left images
            left_img_filename = selected_filenames[i] + '_L.png'
            left_img = cv.imread(left_img_filename, cv.IMREAD_COLOR)
            left_img = cv.normalize(left_img, None, alpha=0, beta=1,norm_type=cv.NORM_MINMAX,dtype=cv.CV_32F)

            # right images
            right_img_filename = selected_filenames[i] + '_R.png'
            right_img = cv.imread(right_img_filename, cv.IMREAD_COLOR)
            right_img = cv.normalize(right_img, None, alpha=0, beta=1,norm_type=cv.NORM_MINMAX,dtype=cv.CV_32F)


            # load labels
            left_txt_filename = left_img_filename.replace('img', 'pos')
            right_txt_filename = right_img_filename.replace('img', 'pos')
            left_txt_filename = left_txt_filename.replace('png', 'txt')
            right_txt_filename = right_txt_filename.replace('png', 'txt')

            left_gts = []
            right_gts = []
            with open(left_txt_filename,'r') as left_file:
                with open(right_txt_filename, 'r') as right_file:
                    for left_line, right_line in zip(left_file, right_file):
                        left_gts.append(np.float_(re.findall(r'-?\d\.\d+', left_line)))
                        right_gts.append(np.float_(re.findall(r'-?\d\.\d+', right_line)))

            # Augmentation
            if self.resize_only:
                augmenter = Augmenter(self.seqs,True,left_img,right_img,left_gts,right_gts,(left_img.shape[1],left_img.shape[0]),self.im_dim)
            else:
                augmenter = Augmenter(self.seqs,False,left_img,right_img,left_gts,right_gts,(left_img.shape[1],left_img.shape[0]),self.im_dim)
            left_img,right_img,left_gts,right_gts = augmenter.augment()
            

            yolo = np.zeros((self.output_dim, self.output_dim, 7))
            for left_gt, right_gt in zip(left_gts, right_gts):
                x_center_left = (left_gt[2]+left_gt[0])/2
                y_center_left = (left_gt[3]+left_gt[1])/2
                x_center_right = (right_gt[2]+right_gt[0])/2
                y_center_right = (right_gt[3]+right_gt[1])/2
                x_coord = self.output_dim * x_center_left
                y_coord = self.output_dim * y_center_left
                x_idx = int(np.floor(x_coord))
                y_idx = int(np.floor(y_coord))
                x_remainder = x_coord - x_idx
                y_remainder = y_coord - y_idx

                yolo[x_idx, y_idx, 0] = 1.0
                yolo[x_idx, y_idx, 1] = self.inv_sig(x_remainder)
                yolo[x_idx, y_idx, 2] = self.inv_sig(y_remainder)
                wire_length = math.sqrt(math.pow(left_gt[2]-left_gt[0],2)+math.pow(left_gt[3]-left_gt[1],2))
                yolo[x_idx, y_idx, 3] = self.inv_exp(wire_length,self.anchor_wire_length)
                vec_left = np.array([left_gt[2]-left_gt[0],left_gt[3]-left_gt[1]])
                vec_left_norm = vec_left/math.sqrt(math.pow(vec_left[0],2)+math.pow(vec_left[1],2))
                yolo[x_idx, y_idx, 4] = np.arctanh(vec_left_norm[0])
                yolo[x_idx, y_idx, 5] = np.arctanh(vec_left_norm[1])
                #angle = math.atan2(left_gt[3]-left_gt[1],left_gt[2]-left_gt[0])
                #yolo[x_idx, y_idx, 4] = np.arctanh(angle/math.pi)
                disparity = math.fabs(x_center_right-x_center_left)
                yolo[x_idx, y_idx, 6] = self.inv_exp(disparity,self.anchor_disparity)

                labels[i] = yolo

            # Resize
            #left_resized = cv.resize(left_img, (416,416), interpolation = cv.INTER_CUBIC)
            #right_resized = cv.resize(right_img, (416,416), interpolation = cv.INTER_CUBIC)
            

            left_images[i] = np.reshape(left_img,(1,*left_img.shape))
            right_images[i] = np.reshape(right_img,(1,*right_img.shape))

        return left_images, right_images, labels