import cv2				    # working with, mainly resizing, images
import numpy as np		    # dealing with arrays
import os				    # dealing with directories
from random import shuffle  # mixing up or currently ordered data that might lead our network astray in training.

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import scipy.io as sc


'''
Dropout is a regularization technique where during each iteration of 
gradient descent, we drop a set of neurons selected at random. 
By drop, what we mean is that we essentially act as if they do not exist.

'''
import matplotlib.pyplot as plt
import math


#wiki set
#TRAIN_DIR = 'D:\\Data\\wiki_crop\\train'
#TEST_DIR = 'D:\\Data\\wiki_crop\\test'
#mat = sc.loadmat('D:\\Data\\wiki_crop\\wiki.mat')
#data = mat['wiki']

#imdb set
TRAIN_DIR = 'D:\\Data\\imdb_crop\\train'
TEST_DIR = 'D:\\Data\\imdb_crop\\test'
mat = sc.loadmat('D:\\Data\\imdb_crop\\imdb.mat')
data = mat['imdb']

IMG_SIZE = 50
LR = 1e-5                  #Learning rate, How fast weights change

# just so we remember which saved model is which, sizes must match
MODEL_NAME = 'wikifaces-{}-{}.model'.format(LR, '2conv-basic')


gender_as_list = data['gender']
gender_as_list = gender_as_list[0][0].tolist()
gender_as_list = ''.join(str(x) for x in gender_as_list)
gender_as_list = gender_as_list.split(',')
gender_as_list = [i.strip(' ') for i in gender_as_list]
gender_as_list = [i.strip('[') for i in gender_as_list]
gender_as_list = [i.strip(']') for i in gender_as_list]

dob_as_list = data['dob']
dob_as_list = dob_as_list[0][0].tolist()
dob_as_list = ''.join(str(x) for x in dob_as_list)
dob_as_list = dob_as_list.split(',')
dob_as_list = [i.strip(' ') for i in dob_as_list]
dob_as_list = [i.strip('[') for i in dob_as_list]
dob_as_list = [i.strip(']') for i in dob_as_list]

photo_taken_as_list = data['photo_taken']
photo_taken_as_list = photo_taken_as_list[0][0].tolist()
photo_taken_as_list = ''.join(str(x) for x in photo_taken_as_list)
photo_taken_as_list = photo_taken_as_list.split(',')
photo_taken_as_list = [i.strip(' ') for i in photo_taken_as_list]
photo_taken_as_list= [i.strip('[') for i in photo_taken_as_list]
photo_taken_as_list = [i.strip(']') for i in photo_taken_as_list]

full_path_as_list = data['full_path']
full_path_as_list = full_path_as_list[0][0]
full_path_as_list = full_path_as_list[0]
full_path_as_list = [str(x) for x in full_path_as_list]
full_path_as_list = [i.strip(' ') for i in full_path_as_list]
full_path_as_list = [i.strip('[') for i in full_path_as_list]   
full_path_as_list = [i.strip(']') for i in full_path_as_list]
full_path_as_list = [i.strip('\'') for i in full_path_as_list]

#helper function that acquires and associates image properties
#with an image
def label_img(img_id):
	img_gender = gender_as_list[img_id]
	img_dob = dob_as_list[img_id]
	img_date = photo_taken_as_list[img_id]
	year_birth = float(img_dob) // 365
	year_taken = float(img_date)
	age = math.floor(year_taken - year_birth)
	if (age >= 0 and age <= 18):
		word_label = 'child'
	elif age > 18 and age <= 30:
		word_label = 'young'
	elif age > 30 and age <= 50:
		word_label = 'mid'
	elif age > 50 and age <= 140:
		word_label = 'old'
	else:
		return 'ERROR'
	if img_gender == 'nan':
		return 'ERROR'
	img_gender = float(img_gender)
	img_gender = math.floor(img_gender)
	if img_gender == 0:
		word_label = word_label + 'F'
	elif img_gender == 1:
		word_label = word_label + 'M'
	else:
		return 'ERROR'
	# conversion to one-hot array
	if word_label == 'childM': return [1, 0, 0, 0, 0, 0, 0, 0]
	elif word_label == 'youngM': return [0, 1, 0, 0, 0, 0, 0, 0]
	elif word_label == 'midM': return [0, 0, 1, 0, 0, 0, 0, 0]
	elif word_label == 'oldM': return [0, 0, 0, 1, 0, 0, 0, 0]
	elif word_label == 'childF': return [0, 0, 0, 0, 1, 0, 0, 0]
	elif word_label == 'youngF': return [0, 0, 0, 0, 0, 1, 0, 0]
	elif word_label == 'midF': return [0, 0, 0, 0, 0, 0, 1, 0]
	elif word_label == 'oldF': return [0, 0, 0, 0, 0, 0, 0, 1]


def create_train_data():
	training_data = []
	train_dirs = os.listdir(TRAIN_DIR)
	for curr_dir in train_dirs:
		files_inside = os.listdir(TRAIN_DIR + '\\' + curr_dir)
		for curr_file in files_inside:
			full_filename = curr_dir + '/' + curr_file
			img_id = full_path_as_list.index(full_filename)
			label = label_img(img_id)
			if label == 'ERROR':
				continue
			path = TRAIN_DIR + '\\' + curr_dir + '\\' + curr_file
			img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
			img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
			training_data.append([np.array(img), np.array(label)])
		print('Added ' + curr_dir + ' to training set.')
	shuffle(training_data)
	np.save('train_data.npy', training_data)
	return training_data


def process_test_data():
	testing_data = []
	test_dirs = os.listdir(TEST_DIR)
	for curr_dir in test_dirs:
		files_inside = os.listdir(TEST_DIR + '\\' + curr_dir)
		for curr_file in files_inside:
			path = TEST_DIR + '\\' + curr_dir + '\\' + curr_file
			img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
			img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
			full_filename = curr_dir + '/' + curr_file
			img_id = full_path_as_list.index(full_filename)
			testing_data.append([np.array(img), img_id])
	shuffle(testing_data)
	np.save('test_data.npy', testing_data)
	return testing_data


# if you need to create the data:
train_data = create_train_data()

# If you have already created the dataset:
#train_data = np.load('train_data.npy')




'''
Here we create a network with 7 layers.

First five layers are convolutional, and the last two are fully connected.
Every conv. layer has it own input and output.
Both input and output have their own (potentially different) sizes, in
the form of width*height*depth.

Input for the first conv. layer is a picture, if it's a colored picture,
the dimensions are width*height*depth (or R*G*B).

But in this example the picture is transformed into a black and white and
scaled to dimensions of IMG_SIZE*IMG_SIZE.
'None' in the first line means that there will be several, undefined number,
of pictures.

Convolution in 2D is basically summ of weights for each pixel and pixels around
it( 5x5 for example).
Every value is multiplied by some weight value(those factors are trained by
the network). That is called convolutional filter.

The result of calculating summ of each pixel is a 2d matrix similar to the
original picture called feature map.
The resulting image is the same size as the original image, or slightly smaller
due to the pixels on the edge not having 5*5 pixels around them.

One convolutional layer can have, and mostly has, more convolutional filters,
and every one of them produces it's own feature map.
So if the layer has 32 convolutional filers, and the area for summ calculation
is 5x5, the output of the whole layer will be 5*5*32.    

max_pool is a downsampling strategy in which a block of dimensions x*y is
represtented by the largest value contained within that block                                           
'''


convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

#make a new conv. layer with the dimensions of the previous one,
#with 32 filters, and the dimensions of filers are 5*5
convnet = conv_2d(convnet, 32, 5, activation='relu')#convolutional = convnet
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 8, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')


train = train_data[:-500]
test = train_data[-500:]

X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=5, validation_set=({'input': test_x}, {'targets': test_y}),
	snapshot_step=500, show_metric=True, run_id=MODEL_NAME)


# if you need to create the data:
test_data = process_test_data()

# if you already have some saved:
#test_data = np.load('test_data.npy')

fig = plt.figure()

for num, data in enumerate(test_data[:12]):
	# cat: [1,0]
	# dog: [0,1]

	img_num = data[1]
	img_data = data[0]

	y = fig.add_subplot(3, 4, num + 1)
	orig = img_data
	data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
	# model_out = model.predict([data])[0]
	model_out = model.predict([data])
	
	if np.argmax(model_out) == 0:
		str_label = 'childM'
	elif np.argmax(model_out) == 1:
		str_label = 'youngM'
	elif np.argmax(model_out) == 2:
		str_label = 'midM'
	elif np.argmax(model_out) == 3:
		str_label = 'oldM'
	elif np.argmax(model_out) == 4:
		str_label = 'childF'
	elif np.argmax(model_out) == 5:
		str_label = 'youngF'
	elif np.argmax(model_out) == 6:
		str_label = 'midF'
	elif np.argmax(model_out) == 7:
		str_label = 'oldF'
	else:
		str_label = '?????????'
	y.imshow(orig, cmap='gray')
	plt.title(str_label)
	y.axes.get_xaxis().set_visible(False)
	y.axes.get_yaxis().set_visible(False)
plt.show()
