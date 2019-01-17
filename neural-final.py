import numpy as np
import cv2
import os
import math
from random import shuffle
# Keras
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.models import load_model, Model
# VGG16
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
# GUI
from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import ImageTk, Image

import scipy.io as sc

ROOT_DIR = 'D:\\Data\\program\\'
img_dir_WIKI = ROOT_DIR + 'wiki_crop'
img_dir_IMDB = ROOT_DIR + 'imdb_crop'

IMG_SIZE = 50
LEARNING_RATE_GENDER = 1e-5 #Learning rate, How fast weights change
LEARNING_RATE_AGE = 1e-5    #Learning rate, How fast weights change
EPOCH_GENDER = 2
EPOCH_AGE = 2
BATCH_SIZE_GENDER = 64
BATCH_SIZE_AGE = 64
MODEL_NAME = 'imdb_wiki_faces-{}'.format('CV_model6')
IMAGESET_NAME = 'image_set_imdb.npy'
# Total of 523051 files in datasets
#62329
TOTAL_SIZE = 62329
# Training - validation - testing ratio -> 80-10-10

#this data set is used to adjust the weights on the neural network.
TRAINING_END_INDEX = int(TOTAL_SIZE * 0.8)

'''
 Validation data set is used to minimize overfitting. 
 You're not adjusting the weights of the network with this data set, 
 you're just verifying that any increase in accuracy 
 over the training data set actually yields an increase 
 in accuracy over a data set that has not been shown 
 to the network before, 
 or at least the network hasn't trained on it 
 (i.e. validation data set). 
 If the accuracy over the training data set increases, 
 but the accuracy over the validation data set 
 stays the same or decreases, 
 then you're overfitting your neural network 
 and you should stop training
'''
VALIDATION_END_INDEX = int(TRAINING_END_INDEX + TOTAL_SIZE * 0.1)

# Age groups
AGE_BOUNDARY1 = 18
AGE_BOUNDARY2 = 25
AGE_BOUNDARY3 = 35
AGE_BOUNDARY4 = 45
AGE_BOUNDARY5 = 60

def prepare_data(data, dataType):
	if(dataType != 'full_path'):
		dataType = data[dataType]
		dataType = dataType[0][0].tolist()
		dataType = ''.join(str(x) for x in dataType)
		dataType = dataType.split(',')
		dataType = [i.strip(' ') for i in dataType]
		dataType = [i.strip('[') for i in dataType]
		dataType = [i.strip(']') for i in dataType]
		return dataType    
	else:
		dataType = data[dataType]
		dataType = dataType[0][0]
		dataType = dataType[0]
		dataType = [str(x) for x in dataType]
		dataType = [i.strip(' ') for i in dataType]
		dataType = [i.strip('[') for i in dataType]   
		dataType = [i.strip(']') for i in dataType]
		dataType = [i.strip('\'') for i in dataType]  
	return dataType

# Metadata from wiki dataset
matWiki = sc.loadmat(ROOT_DIR + 'wiki_crop\\wiki.mat')
dataWiki = matWiki['wiki']

gender_as_list_WIKI = prepare_data(dataWiki, 'gender')
dob_as_list_WIKI = prepare_data(dataWiki, 'dob')
photo_taken_as_list_WIKI = prepare_data(dataWiki, 'photo_taken')
full_path_as_list_WIKI = prepare_data(dataWiki, 'full_path')

'''
# Metadata from imdb dataset
matIMDB = sc.loadmat(ROOT_DIR + 'imdb_crop\\imdb.mat')
dataIMDB = matIMDB['imdb']

gender_as_list_IMDB = prepare_data(dataIMDB, 'gender')
dob_as_list_IMDB = prepare_data(dataIMDB, 'dob')
photo_taken_as_list_IMDB = prepare_data(dataIMDB, 'photo_taken')
full_path_as_list_IMDB = prepare_data(dataIMDB, 'full_path')
'''
# One element has a different name in .mat file

for (i,string) in enumerate(full_path_as_list_WIKI):
	if string.startswith('63/670663'):
		full_path_as_list_WIKI[i] = '63/670663_2015-02-16UTC08_04_1941.jpg'
		break

#join images with their metadata and form that into a set
def create_image_set():
	image_set = []
	#datasets = ['WIKI', 'IMDB']
	datasets = ['WIKI']
	for current_dataset in datasets:
		img_dir = eval('img_dir_' + current_dataset)
        #eval runs the code that is in the ()
		full_path_as_list = eval('full_path_as_list_' + current_dataset)
		gender_as_list = eval('gender_as_list_' + current_dataset)
		dob_as_list = eval('dob_as_list_' + current_dataset)
		photo_taken_as_list = eval('photo_taken_as_list_' + current_dataset)
		# Add images to image set
		train_dirs = os.listdir(img_dir)
        #one file in the directory is not a directory, so we remove it
		train_dirs.remove(str(current_dataset).lower() + '.mat')
		for curr_dir in train_dirs:
			files_inside = os.listdir(img_dir + '\\' + curr_dir)
			for curr_file in files_inside:
				full_filename = curr_dir + '/' + curr_file
				img_id = full_path_as_list.index(full_filename)
				# Get image metadata
				gender = gender_as_list[img_id]
				img_dob = dob_as_list[img_id]
				img_date = photo_taken_as_list[img_id]
				year_birth = float(img_dob) // 365
				year_taken = float(img_date)
				age = math.floor(year_taken - year_birth)
				# Weed out unspecified data
				if not(age > 0 and age < 100):
					continue
				if gender == 'nan':
					continue
				path = img_dir + '\\' + curr_dir + '\\' + curr_file
				# Transform original image
				img = cv2.imread(path)
				img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
				# Add image with metadata
				age_label = get_age_label(age)
				gender = int(float(gender))
				gender = get_gender_label(gender)
				image_set.append([np.array(img), gender, age_label])
			print('Added ' + current_dataset + ' ' + curr_dir + ' to dataset.')
	shuffle(image_set)
	np.save(IMAGESET_NAME, image_set)
	print('Dataset created.')
	return image_set

def get_gender_label(gender):
	vector = [0, 0]
	# Female
	if(gender == 0):
		vector[0] = 1
	# Male
	elif(gender == 1):
		vector[1] = 1
	return vector

def get_age_label(age):
	vector = [0, 0, 0, 0, 0, 0]
	if(age < AGE_BOUNDARY1):
		vector[0] = 1
	elif(age >= AGE_BOUNDARY1 and age < AGE_BOUNDARY2):
		vector[1] = 1
	elif(age >= AGE_BOUNDARY2 and age < AGE_BOUNDARY3):
		vector[2] = 1
	elif(age >= AGE_BOUNDARY3 and age < AGE_BOUNDARY4):
		vector[3] = 1
	elif(age >= AGE_BOUNDARY4 and age < AGE_BOUNDARY5):
		vector[4] = 1
	elif(age >= AGE_BOUNDARY5):
		vector[5] = 1
	return vector

image_set = create_image_set()
# Alternatively import image set
#image_set = np.load(IMAGESET_NAME)

# Split data
training_data = image_set[:TRAINING_END_INDEX]
validation_data = image_set[TRAINING_END_INDEX:VALIDATION_END_INDEX]
test_data = image_set[VALIDATION_END_INDEX:]

# Training data
x_train = list()
# Gender
y_train = list()
# Age
z_train = list()
for i in training_data:
	x_train.append(i[0].reshape(IMG_SIZE,IMG_SIZE,3))
	y_train.append(i[1])
	z_train.append(i[2])
x_train = np.array(x_train)
y_train = np.array(y_train)
z_train = np.array(z_train)
# Test data
x_test = list()
# Gender
y_test = list()
# Age
z_test = list()
for i in test_data:
	x_test.append(i[0].reshape(IMG_SIZE,IMG_SIZE,3))
	y_test.append(i[1])
	z_test.append(i[2])
x_test = np.array(x_test)
y_test = np.array(y_test)
z_test = np.array(z_test)
# Validation data
x_val = list()
# Gender
y_val = list()
# Age
z_val = list()
for i in validation_data:
	x_val.append(i[0].reshape(IMG_SIZE,IMG_SIZE,3))
	y_val.append(i[1])
	z_val.append(i[2])
x_val = np.array(x_val)
y_val = np.array(y_val)
z_val = np.array(z_val)


# Gender recognition
modelGEN = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
layerInputGender = modelGEN.input
layerOutputGender = Dense(2, activation='softmax', name='output_gender')
layerFlatten = Flatten()
layerOutputGender = layerOutputGender(layerFlatten(modelGEN.output))
modelGEN_TURBO = Model(layerInputGender, layerOutputGender)

sgd = SGD(lr=LEARNING_RATE_GENDER, decay=1e-6, momentum=0.9, nesterov=True)
modelGEN_TURBO.compile(loss='categorical_crossentropy', optimizer=sgd)

modelGEN_TURBO.fit(x_train, y_train, batch_size=BATCH_SIZE_GENDER, epochs=EPOCH_GENDER, validation_data = (x_val, y_val))
modelGEN_TURBO.save('{}GENDER.h5'.format(MODEL_NAME))
#scoreGEN = modelGEN_TURBO.evaluate(x_test, y_test, batch_size=32)
#print(scoreGEN)


# Age recognition
modelAGE = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
layerInputGender = modelAGE.input
layerOutputGender = Dense(6, activation='softmax', name='output_gender')
layerFlatten = Flatten()
layerOutputGender = layerOutputGender(layerFlatten(modelAGE.output))
modelAGE_TURBO = Model(layerInputGender, layerOutputGender)

sgd = SGD(lr=LEARNING_RATE_AGE, decay=1e-6, momentum=0.9, nesterov=True)
modelAGE_TURBO.compile(loss='categorical_crossentropy', optimizer=sgd)

modelAGE_TURBO.fit(x_train, z_train, batch_size=BATCH_SIZE_AGE, epochs=EPOCH_AGE, validation_data = (x_val, z_val))
modelAGE_TURBO.save('{}AGE.h5'.format(MODEL_NAME))
#scoreAGE = modelAGE_TURBO.evaluate(x_test, z_test, batch_size=32)
#print(scoreAGE)


# GUI
window = Tk()
window.title('Age and gender recognition')
window.geometry('500x500')
image_file_path = ''
model_file_pathAGE = ''
model_file_pathGEN = ''
panel = Label(window, image = '')
first_image = True
S = Scrollbar(window)
T = Text(window, height=6, width=50)
def select_image():
	global image_file_path
	image_file_path = askopenfilename()
def select_age_model():
	global model_file_pathAGE
	global modelAGE_TURBO
	model_file_pathAGE = askopenfilename()
	if model_file_pathAGE == '':
		return
	modelAGE_TURBO = load_model(model_file_pathAGE)
def select_gender_model():
	global model_file_pathGEN
	global modelGEN_TURBO
	model_file_pathGEN = askopenfilename()
	if model_file_pathGEN == '':
		return
	modelGEN_TURBO = load_model(model_file_pathGEN)
def process_image():
	global panel
	global image_file_path
	global first_image
	global T
	if image_file_path == '':
		print('Image not specified.')
		return
	if modelAGE_TURBO == '':
		print('Age model not specified.')
		return
	if modelGEN_TURBO == '':
		print('Gender model not specified.')
		return
	img = ImageTk.PhotoImage(Image.open(image_file_path))
	if first_image:
		panel = Label(window, image = img)
		panel.pack(side = 'top', fill = 'both', expand = 'yes')
		first_image = False
	else:
		panel.configure(image = img)
		panel.image = img
	image_input = cv2.imread(image_file_path)
	image_input = cv2.resize(image_input, (IMG_SIZE, IMG_SIZE))
	predict_input = image_input.reshape(-1, IMG_SIZE,IMG_SIZE,3)
	predict_input = np.array(predict_input)
	predictionGEN = modelGEN_TURBO.predict(predict_input, batch_size=None, verbose=0, steps=None)
	predictionAGE = modelAGE_TURBO.predict(predict_input, batch_size=None, verbose=0, steps=None)
	T.delete(1.0, END)
	textContent = 'Results:\n'
	T.insert(END, textContent)
	pG = predictionGEN[0]
	pA = predictionAGE[0]
	textContent=('F     {}\nM     {}\n<18   {}\n18-25 {}\n25-35 {}\n35-45 {}\n45-60 {}\n>60   {}\n'.format(pG[0], pG[1], pA[0],pA[1],pA[2],pA[3],pA[4],pA[5]))
	T.insert(END, textContent)
	print(predictionGEN)
	print(predictionAGE)
	window.mainloop()

button1 = Button(window, text='Select image', command=select_image)
button2 = Button(window, text='Select age model', command=select_age_model)
button3 = Button(window, text='Process image', command=process_image)
button4 = Button(window, text='Select gender model', command=select_gender_model)
button1.pack()
button4.pack()
button2.pack()
button3.pack()
S.pack(side=RIGHT)
T.pack(side=BOTTOM)
S.config(command=T.yview)
T.config(yscrollcommand=S.set)
textContent = 'Please select models and image.'
T.insert(END, textContent)
window.mainloop()