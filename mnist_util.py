import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage import interpolation


def elastic_transform(image, alpha=36, sigma=5, random_state=None):
	"""Elastic deformation of images as described in [Simard2003]_.
	.. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
	   Convolutional Neural Networks applied to Visual Document Analysis", in
	   Proc. of the International Conference on Document Analysis and
	   Recognition, 2003.

	:param image: a 28x28 image
	:param alpha: scale for filter
	:param sigma: the standard deviation for the gaussian
	:return: distorted 28x28 image
	"""
	image = image.reshape((28,28))
	assert len(image.shape) == 2

	if random_state is None:
		random_state = np.random.RandomState(None)

	shape = image.shape

	dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
	dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

	x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
	indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))

	return map_coordinates(image, indices, order=1).reshape(shape).reshape(784)

def moments(image):
	"""
	Deskew
	credit: Dibya Ghosh
	"""
	c0,c1 = np.mgrid[:image.shape[0],:image.shape[1]] # A trick in numPy to create a mesh grid
	totalImage = np.sum(image) #sum of pixels
	m0 = np.sum(c0*image)/totalImage #mu_x
	m1 = np.sum(c1*image)/totalImage #mu_y
	m00 = np.sum((c0-m0)**2*image)/totalImage #var(x)
	m11 = np.sum((c1-m1)**2*image)/totalImage #var(y)
	m01 = np.sum((c0-m0)*(c1-m1)*image)/totalImage #covariance(x,y)
	mu_vector = np.array([m0,m1]) # Notice that these are \mu_x, \mu_y respectively
	covariance_matrix = np.array([[m00,m01],[m01,m11]]) # Do you see a similarity between the covariance matrix
	return mu_vector, covariance_matrix

def deskew(image):
	"""
	Deskew
	credit: Dibya Ghosh
	"""
	image = image.reshape((28,28))
	c,v = moments(image)
	alpha = v[0,1]/v[0,0]
	affine = np.array([[1,0],[alpha,1]])
	ocenter = np.array(image.shape)/2.0
	offset = c-np.dot(affine,ocenter)
	return interpolation.affine_transform(image,affine,offset=offset).reshape((784,))

def batch_mean(X):
	X=X.astype('float64')
	mu = np.mean(X, axis=0)
	return mu

def preprocess_elastic(train_data, train_label, X_test, n):
	print("start preprocessing...")
	indices = np.arange(len(train_data))
	train_label = one_hot(train_label)
	np.random.shuffle(indices)
	training_indices = indices[:50000]
	validation_indices = indices[50000:]
	featurize_indices = np.random.random_integers(low=0, high=59999, size=n)
	featurize_data = np.array(list(map(elastic_transform, train_data[featurize_indices])))
	train_new = np.concatenate([train_data[training_indices], featurize_data])
	train_deskew = np.array(list(map(deskew, np.array(list(train_new)))))
	label_new = np.concatenate([train_label[training_indices], train_label[featurize_indices]])
	mu = batch_mean(train_deskew)
	train_final = (train_deskew-mu)/255.0
	valid_data = (np.array(list(map(deskew,train_data[validation_indices])))-mu)/255.0
	valid_label = train_label[validation_indices]
	return np.array(list(zip(train_final, label_new))), np.array(list(zip(valid_data, valid_label))), (np.array(list(map(deskew, X_test)))-mu)/std

def one_hot(labels_train):
	"""
	Convert categorical labels 0,1,2,....9 to standard basis vectors in R^{10}
	"""
	rtn = np.zeros((labels_train.shape[0], 10))
	for i in range(labels_train.shape[0]):
		rtn[i][labels_train[i]] = 1
	return rtn


