import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np


def SSIM_LOSS(img1, img2, size = 11, sigma = 1.5):
	window = _tf_fspecial_gauss(size, sigma)  # window shape [size, size]
	k1 = 0.01
	k2 = 0.03
	L = 1  # depth of image (255 in case the image has a different scale)
	c1 = (k1 * L) ** 2
	c2 = (k2 * L) ** 2
	mu1 = tf.nn.conv2d(img1, window, strides = [1, 1, 1, 1], padding = 'VALID')
	mu2 = tf.nn.conv2d(img2, window, strides = [1, 1, 1, 1], padding = 'VALID')
	mu1_sq = mu1 * mu1
	mu2_sq = mu2 * mu2
	mu1_mu2 = mu1 * mu2
	sigma1_sq = tf.nn.conv2d(img1 * img1, window, strides = [1, 1, 1, 1], padding = 'VALID') - mu1_sq
	sigma2_sq = tf.nn.conv2d(img2 * img2, window, strides = [1, 1, 1, 1], padding = 'VALID') - mu2_sq
	sigma1_2 = tf.nn.conv2d(img1 * img2, window, strides = [1, 1, 1, 1], padding = 'VALID') - mu1_mu2

	# value = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
	ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma1_2 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
	value = tf.reduce_mean(ssim_map, axis = [1, 2, 3])
	return value


# def SSIM_LOSS(img1, img2):
# 	# window = _tf_fspecial_gauss(size, sigma)  # window shape [size, size]
# 	K1 = 0.01
# 	K2 = 0.03
# 	L = 1  # depth of image (255 in case the image has a different scale)
# 	C1 = (K1 * L) ** 2
# 	C2 = (K2 * L) ** 2
# 	C3=C2/2
#
#
# 	mu1=tf.reduce_mean(img1, axis=[1,2])
# 	mu2=tf.reduce_mean(img2, axis=[1,2])
# 	mu1_sq = tf.multiply(mu1,mu1)
# 	mu2_sq = tf.multiply(mu2,mu2)
#
#
# 	img_ones=tf.ones_like(img1)
# 	for i in range(img1.shape[0]):
# 		m1 = tf.expand_dims(img_ones[i,:,:,:]*mu1[i],axis=0)
# 		m2 = tf.expand_dims(img_ones[i, :, :, :] * mu2[i],axis=0)
# 		if i>0:
# 			mu1_imgshape=tf.concat([mu1_imgshape,m1],axis=0)
# 			mu2_imgshape = tf.concat([mu2_imgshape, m2], axis = 0)
# 		else:
# 			mu1_imgshape=m1
# 			mu2_imgshape=m2
#
# 	sigma1_sq=tf.reduce_mean(tf.square(img1-mu1_imgshape),axis=[1,2])
# 	sigma2_sq=tf.reduce_mean(tf.square(img2-mu2_imgshape),axis=[1,2])
# 	sigma12=tf.reduce_mean(tf.multiply(img1-mu1_imgshape,img2-mu2_imgshape),axis=[1,2])
# 	sigma1=tf.sqrt(sigma1_sq)
# 	sigma2=tf.sqrt(sigma2_sq)
#
#
# 	# # mu1 = tf.nn.conv2d(img1, window, strides = [1, 1, 1, 1], padding = 'VALID')
# 	# # mu2 = tf.nn.conv2d(img2, window, strides = [1, 1, 1, 1], padding = 'VALID')
# 	# mu1_mu2 = mu1 * mu2
# 	# # sigma1_sq = tf.nn.conv2d(img1 * img1, window, strides = [1, 1, 1, 1], padding = 'VALID') - mu1_sq
# 	# # sigma2_sq = tf.nn.conv2d(img2 * img2, window, strides = [1, 1, 1, 1], padding = 'VALID') - mu2_sq
# 	# # sigma12 = tf.nn.conv2d(img1 * img2, window, strides = [1, 1, 1, 1], padding = 'VALID') - mu1_mu2
#
#
# 	value1=(2.0 * mu1 * mu2 + C1) / (mu1_sq + mu2_sq + C1)
# 	value2=(2.0 * sigma1 * sigma2 + C2) / (sigma1_sq +sigma2_sq + C2)
# 	value3=(sigma12 + C3) / (sigma1*sigma2 + C3)
# 	value=tf.multiply(value1, value2)
# 	value=tf.multiply(value,value3)
# 	return value

def _tf_fspecial_gauss(size, sigma):
	"""Function to mimic the 'fspecial' gaussian MATLAB function
	"""
	x_data, y_data = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]

	x_data = np.expand_dims(x_data, axis = -1)
	x_data = np.expand_dims(x_data, axis = -1)

	y_data = np.expand_dims(y_data, axis = -1)
	y_data = np.expand_dims(y_data, axis = -1)

	x = tf.constant(x_data, dtype = tf.float32)
	y = tf.constant(y_data, dtype = tf.float32)

	g = tf.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
	return g / tf.reduce_sum(g)


def L1_LOSS(batchimg):
	_,h,w,_=batchimg.get_shape().as_list()
	L1_norm = tf.reduce_sum(tf.abs(batchimg), axis = [1, 2])
	L1_norm=L1_norm/(h*w)
	# tf.norm(batchimg, axis = [1, 2], ord = 1) / int(batchimg.shape[1])
	E = tf.reduce_mean(L1_norm)
	return E


def Per_LOSS(batchimg):
	_, h, w, c = batchimg.get_shape().as_list()
	fro_2_norm = tf.reduce_sum(tf.square(batchimg),axis=[1,2,3])
	loss=fro_2_norm / (h * w * c)
	return loss


def Fro_LOSS(batchimg):
	fro_norm = tf.square(tf.norm(batchimg, axis = [1, 2], ord = 'fro')) / (int(batchimg.shape[1]) * int(batchimg.shape[2]))
	E = tf.reduce_mean(fro_norm)
	return fro_norm