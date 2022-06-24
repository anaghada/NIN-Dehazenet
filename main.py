from __future__ import unicode_literals

import cv2
import tensorflow as tf
import numpy as np
import glob
import sys
from models import *
import yaml
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import utils
import ops


with open("config.yaml") as file:
	data = yaml.safe_load(file)
	training_params = data['training_params']
	learning_rate = float(training_params['learning_rate'])
	batch_size = int(training_params['batch_size'])
	epoch_size = int(training_params['epochs'])
	data_path = training_params['data_path']
	model_params= data['model_params']
	descrip = model_params['descrip']
	log_dir = model_params['log_dir']
	mode = model_params['mode']
	if len(descrip)==0:
		print("Please give a proper description of the model you are training.")


######## Making Directory #########
print("log dir:",log_dir)
model_path = log_dir
# model_path = log_dir+sys.argv[1]
print ("Model Path: ", model_path)
if not os.path.exists(model_path):
	os.makedirs(model_path)
	os.makedirs(model_path+"/results")
	os.makedirs(model_path+"/tf_graph")
	os.makedirs(model_path+"/saved_model")

# print("aaaaa")
######### Loading Data ###########

print(data_path+"train_haze_clear.npy")
# print("hellllpooooooooooooooooooooooooooooooooooooooooo")
train_img1 = 1/255.0*np.load(data_path+"train_haze_clear.npy",allow_pickle=True) # First image of each pair is hazy image and second image is clear images
train_img2 = 1/255.0*np.load(data_path+"train_trans.npy",allow_pickle=True)
val_img1 = 1/255.0*np.load(data_path+"val_haze_clear.npy",allow_pickle=True)
val_img2 = 1/255.0*np.load(data_path+"val_trans.npy",allow_pickle=True)
print ("Data Loaded")

nnet = MSCNN(model_path)
mode="predict"

if mode=='train':
	os.system('cp config.yaml '+model_path+'/config.yaml')
	os.system('cp models.py '+model_path+'/model.py')

	nnet.build_model()
	print ("Model Build......")
	nnet.train_model([train_img1, train_img2], [val_img1, val_img2], learning_rate, batch_size, epoch_size)
# else:
#
# 	predict = nnet.test(train_img1[:,0,:,:,:], batch_size)
# 	print(predict)
# 	print(type(predict))
# 	print("+++++++++++++++++++++++++++++++++++")
#
# 	for i in range(train_img2.shape[0]):
# 		print("predicted transformation")
# 		print(predict)
# 		print(predict.shape)
# 		print(predict[i].shape)
# 		# cv2.imshow("aaaaaa",predict[i])
# 		# cv2.imwrite("aaaa1010.jpg",predict[i]*255.0)
# 		# cv2.imwrite("img11.jpg",predict[i])
# 		# cv2.waitKey(0)
# 		# from  PIL import Image
# 		# print(type(a))
# 		# break
# 		clear_img = utils.clearImg(train_img1[i,0,:,:,:], predict[i])
# 		pair = np.hstack((train_img2[i], predict[i]))
# 		pair2 = np.hstack((train_img1[i,0,:,:,:],train_img1[i,1,:,:,:], clear_img))
# 		plt.imshow(pair[:,:,0])
# 		plt.show()
# 		plt.imshow(pair2)
# 		plt.show()
# 		cv2.imwrite(str(i)+"_clear.jpg", 255.0*clear_img)
# 		print(str(i)+"_trans.jpg")
# 		cv2.imwrite(str(i)+"_trans.jpg", 255.0*train_img1[i,1,:,:,:])
#
#
# 		from Eval import measures
# 		h,j=measures(str(i)+"_clear.jpg",str(i)+"_trans.jpg")
# 		print("psnr:",h)
# 		print("ssim",j)
#
# 		break


def pdt(filepath):
	# aa = cv2.imread("C:\\Users\\ANAGHA DHILEEP\\PycharmProjects\\NIN_Dehazenet\\static\\image\\a.jpg")
	aa = cv2.imread(filepath)
	# cv2.imwrite("aa.jpg",aa)

	print(aa.shape)


	gh=train_img1[:, 0, :, :, :]
	print("shshsh lik")
	print(gh.shape)
	####tttttt
	image="C:\\Users\\ANAGHA DHILEEP\\PycharmProjects\\NIN_Dehazenet\\static\\image\\a.jpg"
	final_size=240
	img = cv2.imread(image)
	resize_img=img
	print("end file")
	print(image, ":image loc")
	w = img.shape[1]
	h = img.shape[0]
	print("height and width")
	ar = float(w) / float(h)
	# print("a1")
	if w < h:
		# print("a11")
		new_w = final_size
		new_h = int(new_w / ar)
		a = new_h - final_size
		resize_img = cv2.resize(img, dsize=(new_w, new_h))

	print("uuuuuuuuu")
	cv2.imwrite("resize.jpg",resize_img)






	###endt


	# predict = nnet.test(train_img1[:, 0, :, :, :], batch_size)
	predict = nnet.test(resize_img,batch_size)
	cv2.imwrite("z.jpg",predict)

	# cv2.imshow("kkkk",predict*255)
	# cv2.waitKey(45)



	# predict=nnet.test(aa,batch_size)
	print(predict)
	print(type(predict))
	print("+++++++++++++++++++++++++++++++++++")
	my=0
	# train_img2.shape[0]
	for i in range(len(predict)):
		print("predicted transformation")
		print(predict)
		print(predict.shape)
		# print(predict[i].shape)
		try:
			cv2.imshow("aaaaaa",predict[i])
			cv2.waitKey(1)
			cv2.imwrite("aaaa1010.jpg",predict[i])

		except Exception as ex:
			print(ex)
		# cv2.imwrite("img11.jpg",predict[i])
		# cv2.waitKey(0)
		# from  PIL import Image
		# print(type(a))
		# break

		# print(train_img1[i, 0, :, :, :].shape,"-----",my,"myyyy")
		my+=1
		clear_img= utils.clearImg(resize_img, predict[i])

		# print(k,"hoooooooooooooooooooooooo")
		cv2.imwrite("abc.jpg",clear_img*255)
		from kl import mkall
		ks=mkall(filepath)




		from testdehaze import finalenhancement

		a= finalenhancement(resize_img)

		f="C:\\Users\\ANAGHA DHILEEP\\PycharmProjects\\NIN_Dehazenet\\static\\"

		cv2.imwrite(f+"K.jpg",a)

		# pap=np.hstack((clear_img,predict[i]))
		# pair = np.hstack((train_img2[i], predict[i]))
		# pair2 = np.hstack((train_img1[i, 0, :, :, :], train_img1[i, 1, :, :, :], clear_img))
		# plt.imshow(pair[:, :, 0])
		# plt.show()
		# plt.imshow(pap)
		# plt.show()
		cv2.imwrite("C:\\Users\\ANAGHA DHILEEP\\PycharmProjects\\NIN_Dehazenet\\static\\"+str(i) + "_clear.jpg", 2 * clear_img)


		# ms=train_img1[i, 1, :, :, :]




		print("ghffhfgfgf")
		# print(ms.shape)

		print("C:\\Users\\ANAGHA DHILEEP\\PycharmProjects\\NIN_Dehazenet\\sta+tic\\"+str(i) + "_trans.jpg")
		# cv2.imwrite("C:\\Users\\ANAGHA DHILEEP\\PycharmProjects\\NIN_Dehazenet\\static\\"+str(i) + "_trans.jpg", 255.0 *ms )
		cv2.imwrite("C:\\Users\\ANAGHA DHILEEP\\PycharmProjects\\NIN_Dehazenet\\static\\image\\c.jpg",clear_img)
		cv2.imwrite("C:\\Users\\ANAGHA DHILEEP\\PycharmProjects\\NIN_Dehazenet\\static\\image\\d.jpg",ks)



		# from Eval import measures
		# # h, j = measures(str(i) + "_clear.jpg", str(i) + "_trans.jpg")
		# h, j = measures("C:\\Users\\ANAGHA DHILEEP\\PycharmProjects\\NIN_Dehazenet\\static\\image\\a.jpg", "C:\\Users\\ANAGHA DHILEEP\\PycharmProjects\\NIN_Dehazenet\\static\\K.jpg")
		#
		# print("psnr:", h)
		# print("ssim", j)


