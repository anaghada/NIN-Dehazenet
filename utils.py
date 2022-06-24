from scipy.io import loadmat
import glob
import h5py
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

def resize_img(x_img_list, y_img_list, z_img_list, name):
	final_size = 240
	
	x_image_list = []
	y_image_list = []
	z_image_list= []
	
	for image in x_img_list:
		# print image
		print("read file")
		img = cv2.imread(image)
		print("end file")
		print(image,":image loc")
		w = img.shape[1]
		h = img.shape[0]
		print("height and width")
		ar = float(w)/float(h)
		# print("a1")
		if w<h:
			# print("a11")
			new_w = final_size
			new_h = int(new_w/ar)
			a = new_h - final_size
			resize_img = cv2.resize(img, dsize=(new_w, new_h))
			final_image = resize_img[a/2:a/2+final_size,:]
		elif w>h:
			# print("a12")
			new_h =final_size
			new_w = int(new_h*ar)
			a = new_w - final_size

			# resize_img = cv2.resize(img,dsize=(new_w, new_h))
			resize_img = cv2.resize(img,dsize=(new_w, new_h))
			final_image = resize_img[:,int(a/2):int(a/2)+final_size ]
		else:
			# print("a13")

			# resize_img = cv2.resize(img,dsize=(final_size, final_size))
			resize_img = cv2.resize(img,dsize=(final_size, final_size))
			final_image = resize_img
		# print("a3")
		x_image_list.append(final_image)


	for image in y_img_list:
		# print image		
		img = cv2.imread(image, 0) # Opencv by defaults load an grayscale as an bgr image, with all three channels have same values. 
		w = img.shape[1]			# to load specifically 1 channel, we have to mention that '0'.
		h = img.shape[0]
		ar = float(w)/float(h)
		if w<h:
			new_w = final_size
			new_h = int(new_w/ar)
			a = new_h - final_size
			resize_img = cv2.resize(img, dsize=(new_w, new_h))
			final_image = resize_img[a/2:a/2+final_size,:]
		elif w>h:
			new_h =final_size
			new_w = int(new_h*ar)
			a = new_w - final_size
			resize_img = cv2.resize(img,dsize=(new_w, new_h))
			final_image = resize_img[:,int(a/2):int(a/2)+final_size ]
		else:
			resize_img = cv2.resize(img,dsize=(final_size, final_size))
			final_image = resize_img

		y_image_list.append(final_image)

	for image in z_img_list:
		# print image
		img = cv2.imread(image)
		w = img.shape[1]
		h = img.shape[0]
		ar = float(w)/float(h)
		if w<h:
			new_w = final_size
			new_h = int(new_w/ar)
			a = new_h - final_size
			resize_img = cv2.resize(img, dsize=(new_w, new_h))
			final_image = resize_img[a/2:a/2+final_size,:]
		elif w>h:
			new_h =final_size
			new_w = int(new_h*ar)
			a = new_w - final_size
			resize_img = cv2.resize(img,dsize=(new_w, new_h))
			final_image = resize_img[:,int(a/2):int(a/2)+final_size ]
		else:
			resize_img = cv2.resize(img,dsize=(final_size, final_size))
			final_image = resize_img
		
		z_image_list.append(final_image)
	
	npy = []

	print(len(x_image_list),len(z_image_list),"length")


	for i in range(len(x_image_list)):
		pair = [x_image_list[i], z_image_list[i]]
		npy.append(pair)

	npy = np.array(npy)
	print (npy.shape)
	trans_npy = np.array(y_image_list)
	trans_npy.resize((trans_npy.shape[0], trans_npy.shape[1], trans_npy.shape[2], 1))
	print (trans_npy.shape)
	np.save("D:\\projects\\dataset\\"+name+"_haze_clear.npy", npy)
	np.save("D:\\projects\\dataset\\"+name+"_trans.npy", trans_npy)

# def get_airlight(hzimg,transMap):
# 	airlight = np.zeros(hzimg.shape)
# 	kernel = np.ones((15,15),np.uint8)
# 	for i in range(3):
# 		img = cv2.erode(hzimg[:,:,i],kernel,iterations = 1)
# 		airlight[:,:,i] = np.amax(img)
# 	return airlight

def get_airlight(hzimg,transMap):
	k=0
	m=255
	airlight = np.zeros(hzimg.shape)
	kernel = np.ones((15,15),np.uint8)
	for i in range(3):
		img = cv2.erode(hzimg[:,:,i],kernel,iterations = 1)
		airlight[:,:,i] = np.amax(img)
		k=max(np.max(img),k)
		m=min(np.max(img),k) #chnges to mean to median
	return airlight,k,m

def clearImg(hzimg, transMap):
	print("hzimg_shape:",hzimg.shape)
	print("transmap_shape:",transMap.shape)

	airlight,k,l = get_airlight(hzimg, transMap)

	print("k,l",k,l)
	print(type(airlight))
	print("shapeee today")
	print(airlight.shape)


	file=cv2.imwrite("filename.jpg",airlight)
	print(file)
	print("airlight acquired",airlight)
	clearImg = np.zeros(hzimg.shape)
	print("clearimg_shape",clearImg.shape)
	transMap = transMap.reshape((transMap.shape[0], transMap.shape[1]))
	# print("Avg Airlight",math.avg(transMap))
	print("tm_h_w",transMap.shape)


	cv2.imwrite("q.jpg",hzimg)


	cv2.imwrite("x.jpg",transMap)
	# val=1-airlight

	jk=1-((k-l)/k)
	# jk = 1 - ((k - l))
	print(jk)
	print("jkkkkkkkkk")

	print(transMap,"transss")
	bais=((k-l)/255)
	print("baisss",bais)
	val=1-((bais+jk)/2)
	# val=1-bais
	jv=(jk-val)/2
	print("jv",jv)
	print("vaaal",val)
	constant_matrix = np.ones_like(transMap)*val
	# constant_matrix = np.ones_like(transMap)*bais

	# ((l-k)* 255)
	#


	print(constant_matrix,"consstst")



	# constant_matrix = np.ones_like(transMap)* val

	# clearImg[:,:,0] = (hzimg[:,:,0]-(airlight[:,:,0]*0.1))/np.maximum(constant_matrix, transMap) + (airlight[:,:,0]*0.1)
	# clearImg[:,:,1] = (hzimg[:,:,1]-(airlight[:,:,1]*0.1))/np.maximum(constant_matrix, transMap) + (airlight[:,:,1]*0.1)
	# clearImg[:,:,2] = (hzimg[:,:,2]-(airlight[:,:,2]*0.1))/np.maximum(constant_matrix, transMap) + (airlight[:,:,2]*0.1)
####orginal
	# clearImg[:, :, 0] = (hzimg[:, :, 0] - airlight[:, :, 0]) / np.maximum(constant_matrix, transMap) + airlight[:, :, 0]
	# clearImg[:, :, 1] = (hzimg[:, :, 1] - airlight[:, :, 1]) / np.maximum(constant_matrix, transMap) + airlight[:, :, 1]
	# clearImg[:, :, 2] = (hzimg[:, :, 2] - airlight[:, :, 2]) / np.maximum(constant_matrix, transMap) + airlight[:, :, 2]

	# clearImg[:, :, 0] = ((hzimg[:, :, 0] - airlight[:, :, 0]) / np.maximum(constant_matrix, transMap) + airlight[:, :, 0])+(airlight[:, :, 0]*(1-bais))
	# clearImg[:, :, 1] = ((hzimg[:, :, 1] - airlight[:, :, 1]) / np.maximum(constant_matrix, transMap) + airlight[:, :, 1])+(airlight[:, :, 1]*(1-bais))
	# clearImg[:, :, 2] = ((hzimg[:, :, 2] - airlight[:, :, 2]) / np.maximum(constant_matrix, transMap) + airlight[:, :, 2])+(airlight[:, :, 2]*(1-bais))
	# 																																		#bais
	clearImg[:, :, 0] = ((hzimg[:, :, 0] - airlight[:, :, 0]) / np.maximum(constant_matrix, transMap) + airlight[:, :, 0])+(airlight[:, :, 0]*jv)
	clearImg[:, :, 1] = ((hzimg[:, :, 1] - airlight[:, :, 1]) / np.maximum(constant_matrix, transMap) + airlight[:, :, 1])+(airlight[:, :, 1]*jv)
	clearImg[:, :, 2] = ((hzimg[:, :, 2] - airlight[:, :, 2]) / np.maximum(constant_matrix, transMap) + airlight[:, :, 2])+(airlight[:, :, 2]*jv)

	# clearImg[:, :, 0] = (hzimg[:, :, 0] - (airlight[:, :, 0])* np.maximum(constant_matrix, transMap)) / transMap
	# clearImg[:, :, 1] = (hzimg[:, :, 1] - (airlight[:, :, 1]) * np.maximum(constant_matrix, transMap)) / transMap
	# clearImg[:, :, 2] = (hzimg[:, :, 2] - (airlight[:, :, 2]) * np.maximum(constant_matrix, transMap)) / transMap
	# clearImg[:,:,0] = (hzimg[:,:,0]-airlight[:,:,0])*(1-transMap)/ airlight[:,:,0]
	# clearImg[:,:,1] = (hzimg[:,:,1]-airlight[:,:,1])*(1-transMap)/ airlight[:,:,1]
	# clearImg[:,:,2] = (hzimg[:,:,2]-airlight[:,:,2])*(1-transMap)/ airlight[:,:,2]
	# clearImg[:, :, 0] = (hzimg[:, :, 0] - airlight[:, :, 0]) / (np.maximum(constant_matrix, transMap)* airlight[:, :, 0]) + airlight[:, :, 0]
	# clearImg[:, :, 1] = (hzimg[:, :, 1] - airlight[:, :, 1]) / (np.maximum(constant_matrix, transMap)* airlight[:, :, 1])+ airlight[:, :, 1]
	# clearImg[:, :, 2] = (hzimg[:, :, 2] - airlight[:, :, 2]) / (np.maximum(constant_matrix, transMap)* airlight[:, :, 2]) + airlight[:, :, 2]

	# clearImg[clearImg<0.0]=0.0
	# clearImg[clearImg>1.0]=1.0
	print(clearImg.shape)
	# cv2.imwrite("clr.jpg",clearImg)


	return clearImg

def test_npy(trans, hazy):
	hazy =  np.load(hazy,allow_pickle=True)
	trans = np.load(trans,allow_pickle=True)
	for i in range(len(trans)):
		plt.imshow(hazy[i][1])
		plt.show()
		plt.imshow(hazy[i][0])
		plt.show()
		plt.imshow(trans[i][:,:,0])
		plt.show()

def main():
	# path = "/home/hitech/Downloads/ChinaMM18dehaze/train/"
	path = "D:\\projects\\dataset\dataset\\"

	clear_imgs = glob.glob(path+"clear/*.png")
	trans_imgs = []
	haze_imgs = []
	# trans_imgs = glob.glob(path + "trans/*.png")
	# haze_imgs = glob.glob(path + "hazy/*.png")
	print (len(clear_imgs))
	import os

	for img in clear_imgs:
		print(img)
		head, tail = os.path.split(img)
		print(head,tail)

		trans_imgs.append(path+"trans\\"+tail)
		haze_imgs.append(path+"hazy\\"+tail)
		# trans_imgs.append(path+"trans\\"+tail)
		# haze_imgs.append(path+"hazy\\"+tail)

	# for c,t,h in zip(clear_imgs, trans_imgs, haze_imgs):
	# 	cv2.imshow("clear", cv2.imread(c))
	# 	cv2.imshow("trans", cv2.imread(t))
	# 	cv2.imshow("hazy", cv2.imread(h))
	# 	k = cv2.waitKey(0)
	#
	# resize_img(haze_imgs[:120], trans_imgs[:120], clear_imgs[:120], "train")
	# resize_img(haze_imgs[120:], trans_imgs[120:], clear_imgs[120:], "val")

	# print("okkks")

	test_npy("D:\\projects\\dataset\\train_trans.npy","D:\\projects\\dataset\\train_haze_clear.npy")

if __name__ == "__main__":
	main()
