import os
import pandas as pd
import numpy as np
import cv2

# the 'testset' you see in the path below actually has the big data bulk
TRAIN_PATH = '/media/kalfasyan/HGST_4TB/Datasets/'+\
			 'handwritten_documents/cvl-database-1-1/testset/pages/'

TEST_PATH = '/media/kalfasyan/HGST_4TB/Datasets/'+\
			 'handwritten_documents/cvl-database-1-1/trainset/pages/'

def preprocess_and_save_images(datapath):
	for i, img in enumerate(os.listdir(datapath)):
		image = cv2.imread(datapath+os.listdir(datapath)[i], cv2.IMREAD_GRAYSCALE)
		# Otsu's thresholding after Gaussian filtering
		blur = cv2.GaussianBlur(image,(5,5),0)
		ret,otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
		otsu_inv = cv2.bitwise_not(otsu)
		# Remove bottom black
		crop_image = otsu_inv[:-900, :]
		resize_image = cv2.resize(crop_image, (299, 299))

		# In order to save the data to the right path
		if datapath == TRAIN_PATH:
			datasplit = 'train'
		elif datapath == TEST_PATH:
			datasplit = 'test'
		# Save original upright image			
		# Loop to rotate and save each image for each degree rotated
		for degree in range(-90,90):
			rows, cols = resize_image.shape
			M = cv2.getRotationMatrix2D((cols/2,rows/2), degree,1)
			dst = cv2.warpAffine(resize_image,M,(cols,rows))
			cv2.imwrite('./preprocessed_data/{}/image_{}_{}.png'.format(datasplit,i,degree), dst)
			# cv2.imshow("Image", dst)
			# cv2.waitKey(0)
			# cv2.destroyAllWindows()
			# print("ST")

preprocess_and_save_images(TEST_PATH)


#cv2.imshow("Image", resize_image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()