from copyreg import dispatch_table
import numpy as np
import math
import cv2 as cv
import os
import re

class Evaluator:
	def __init__(self,inferred,test_filenames,im_width,im_height,fold):
		self.inferred = inferred
		self.test_filenames = test_filenames
		self.im_width = im_width
		self.im_height = im_height
		self.fold = fold

	def sigmoid(self,x):
		return 1 / (1 + math.exp(-x))

	def run(self):
		conf_thresh = 0.5
		anch_l, anch_d, =0.053123636693654344, 0.10034577835696407  #0.11300873462575244, 0.12381872213338498 # 0.046552, 0.132489247
		for image_num in range(self.inferred.shape[0]):
			left_img_path = self.test_filenames[image_num]+'_L.png'
			right_img_path = self.test_filenames[image_num]+'_R.png'
			left_txt_path = left_img_path.replace('png','txt')
			right_txt_path = right_img_path.replace('png','txt')
			left_txt_path = left_txt_path.replace('img','pos')
			right_txt_path = right_txt_path.replace('img','pos')

			encoded = self.inferred[image_num] # now of shape (13,13,6)

			left_img = cv.imread(left_img_path, cv.IMREAD_COLOR)
			right_img = cv.imread(right_img_path, cv.IMREAD_COLOR)

			left_gts = []
			right_gts = []
			with open(left_txt_path,'r') as left_file:
				with open(right_txt_path, 'r') as right_file:
					for left_line, right_line in zip(left_file, right_file):
						left_gts.append(np.float_(re.findall(r'-?\d\.\d+', left_line)))
						right_gts.append(np.float_(re.findall(r'-?\d\.\d+', right_line)))

			for left_gt,right_gt in zip(left_gts,right_gts):
				cv.circle(left_img,(int(float(left_gt[0])*self.im_width),int(float(left_gt[1])*self.im_height)),2,(0,0,255),-1)
				cv.circle(left_img,(int(float(left_gt[2])*self.im_width),int(float(left_gt[3])*self.im_height)),2,(0,0,255),-1)
				cv.circle(right_img,(int(float(right_gt[0])*self.im_width),int(float(right_gt[1])*self.im_height)),2,(0,0,255),-1)
				cv.circle(right_img,(int(float(right_gt[2])*self.im_width),int(float(right_gt[3])*self.im_height)),2,(0,0,255),-1)
				cv.line(left_img,(int(float(left_gt[0])*self.im_width),int(float(left_gt[1])*self.im_height)),(int(float(left_gt[2])*self.im_width),int(float(left_gt[3])*self.im_height)),(0,0,255),1)
				cv.line(right_img,(int(float(right_gt[0])*self.im_width),int(float(right_gt[1])*self.im_height)),(int(float(right_gt[2])*self.im_width),int(float(right_gt[3])*self.im_height)),(0,0,255),1)

			left_pred_x = []
			left_pred_y = []
			wire_lengths = []
			x_vecs = []
			y_vecs = []
			disparities = []
			for row in range(encoded.shape[0]):
				for col in range(encoded.shape[1]):
					conf = encoded[row,col,0]
					if conf > conf_thresh:
						left_pred_x.append((row+self.sigmoid(encoded[row,col,1]))/13*self.im_width)
						left_pred_y.append((col+self.sigmoid(encoded[row,col,2]))/13*self.im_height)
						wire_lengths.append(anch_l*math.exp(encoded[row,col,3]))   # anchor!
						x_vecs.append(np.tanh(encoded[row,col,4]))
						y_vecs.append(np.tanh(encoded[row,col,5]))
						disparities.append(anch_d*math.exp(encoded[row,col,6])*self.im_width) # anchor!
			X = []
			for x,y,wire_length,x_vec,y_vec,disparity in zip (left_pred_x,left_pred_y,wire_lengths,x_vecs,y_vecs,disparities):
				x_1 = x+x_vec*wire_length/2*self.im_width
				x_2 = x-x_vec*wire_length/2*self.im_width
				y_1 = y+y_vec*wire_length/2*self.im_height
				y_2 = y-y_vec*wire_length/2*self.im_height
				cv.circle(left_img,(int(x_1),int(y_1)),2,(255,0,0),-1)
				cv.circle(left_img,(int(x_2),int(y_2)),2,(255,0,0),-1)
				cv.circle(right_img,(int(x_1-disparity),int(y_1)),2,(255,0,0),-1)
				cv.circle(right_img,(int(x_2-disparity),int(y_2)),2,(255,0,0),-1)
				cv.line(left_img,(int(x_1),int(y_1)),(int(x_2),int(y_2)),(255,0,0),1)
				cv.line(right_img,(int(x_1-disparity),int(y_1)),(int(x_2-disparity),int(y_2)),(255,0,0),1)
				X.append([x_1,y_1,x_2,y_2, disparity])

			im = np.concatenate((left_img, right_img), axis=1)
			cv.imwrite(os.path.join('..','eval',str('cv_' + str(self.fold)),os.path.basename(self.test_filenames[image_num]).split('.')[0]+'.png'),im)
			np.savetxt(fname = os.path.join('..','eval',str('cv_' + str(self.fold)),os.path.basename(self.test_filenames[image_num]).split('.')[0]+'.txt'),X = X, fmt='%10.15f', delimiter=',', newline='\n',encoding=None)
            #cv.imshow("test",im)
			#cv.waitKey(0) 

	def compareAfterAug(self,left_imgs,right_imgs):
		conf_thresh = 0.5
		anch_l, anch_d, = 0.053123636693654344, 0.10034577835696407  #0.11300873462575244, 0.12381872213338498 #0.046552, 0.132489247#
		for image_num in range(self.inferred.shape[0]):
			encoded = self.inferred[image_num] # now of shape (13,13,6)

			left_img = left_imgs[image_num]
			right_img = right_imgs[image_num]

			left_pred_x = []
			left_pred_y = []
			wire_lengths = []
			x_vecs = []
			y_vecs = []
			disparities = []
			for row in range(encoded.shape[0]):
				for col in range(encoded.shape[1]):
					conf = encoded[row,col,0]
					if conf > conf_thresh:
						left_pred_x.append((row+self.sigmoid(encoded[row,col,1]))/13*self.im_width)
						left_pred_y.append((col+self.sigmoid(encoded[row,col,2]))/13*self.im_height)
						wire_lengths.append(anch_l*math.exp(encoded[row,col,3]))
						x_vecs.append(np.tanh(encoded[row,col,4]))
						y_vecs.append(np.tanh(encoded[row,col,5]))
						disparities.append(anch_d*math.exp(encoded[row,col,6])*self.im_width)

			for x,y,wire_length,x_vec,y_vec,disparity in zip (left_pred_x,left_pred_y,wire_lengths,x_vecs,y_vecs,disparities):
				x_1 = x+x_vec*wire_length/2*self.im_width
				x_2 = x-x_vec*wire_length/2*self.im_width
				y_1 = y+y_vec*wire_length/2*self.im_height
				y_2 = y-y_vec*wire_length/2*self.im_height
				cv.circle(left_img,(int(x_1),int(y_1)),2,(255,0,0),-1)
				cv.circle(left_img,(int(x_2),int(y_2)),2,(255,0,0),-1)
				cv.circle(right_img,(int(x_1-disparity),int(y_1)),2,(255,0,0),-1)
				cv.circle(right_img,(int(x_2-disparity),int(y_2)),2,(255,0,0),-1)
				cv.line(left_img,(int(x_1),int(y_1)),(int(x_2),int(y_2)),(255,0,0),1)
				cv.line(right_img,(int(x_1-disparity),int(y_1)),(int(x_2-disparity),int(y_2)),(255,0,0),1)

			im = np.concatenate((left_img, right_img), axis=1)

			cv.imwrite(os.path.join('..','eval',os.path.basename(self.test_filenames[image_num]).split('.')[0]+'.png'),im)

			#cv.imshow("test",im)
			#cv.waitKey(0)