import numpy as np
import cv2 as cv
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage

class Augmenter:
    def __init__(self,seqs,resize_only,left_img,right_img,left_dets,right_dets,in_im_size,out_im_size):
        self.seqs = seqs
        self.resize_only = resize_only
        self.im_width = in_im_size[0]
        self.im_height = in_im_size[1]
        self.out_im_size = out_im_size
        self.setData(left_img,right_img,left_dets,right_dets)

    def setData(self,left_img,right_img,left_dets,right_dets):
        self.left_img = left_img
        self.right_img = right_img
        left_kps = []
        for left_det in left_dets:
            left_kps.append(Keypoint(x=left_det[0]*self.im_width, y=left_det[1]*self.im_height))
            left_kps.append(Keypoint(x=left_det[2]*self.im_width, y=left_det[3]*self.im_height))
        right_kps = []
        for right_det in right_dets:
            right_kps.append(Keypoint(x=right_det[0]*self.im_width, y=right_det[1]*self.im_height))
            right_kps.append(Keypoint(x=right_det[2]*self.im_width, y=right_det[3]*self.im_height))
        self.left_kps = KeypointsOnImage(left_kps,left_img.shape)
        self.right_kps = KeypointsOnImage(right_kps,right_img.shape)

    def augment(self):
        idx = None
        if self.resize_only:
            idx = 0
        else:
            idx = np.random.randint(0,3)
        selected_seq = self.seqs[idx]
        for augmenter in selected_seq:
            if 'fliplr' in augmenter.name.lower():
                left_image_aug, left_kps_aug = selected_seq(image=self.right_img, keypoints=self.right_kps)
                right_image_aug, right_kps_aug = selected_seq(image=self.left_img, keypoints=self.left_kps)
            
            else:
                left_image_aug, left_kps_aug = selected_seq(image=self.left_img, keypoints=self.left_kps)
                right_image_aug, right_kps_aug = selected_seq(image=self.right_img, keypoints=self.right_kps)

        left_kps_out = []
        right_kps_out = []
        for i in range(0,len(left_kps_aug.keypoints),2):
            left_kp_1 = left_kps_aug.keypoints[i]
            left_kp_2 = left_kps_aug.keypoints[i+1]
            right_kp_1 = right_kps_aug.keypoints[i]
            right_kp_2 = right_kps_aug.keypoints[i+1]
            left_kps_out.append([left_kp_1.x/self.out_im_size,left_kp_1.y/self.out_im_size,left_kp_2.x/self.out_im_size,left_kp_2.y/self.out_im_size])
            right_kps_out.append([right_kp_1.x/self.out_im_size,right_kp_1.y/self.out_im_size,right_kp_2.x/self.out_im_size,right_kp_2.y/self.out_im_size])

        return left_image_aug, right_image_aug, left_kps_out, right_kps_out