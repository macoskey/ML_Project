
# parameters you have to adjust: nb_classes (varies with the problem), batch_size, nb_epoch

import numpy as np
np.random.seed(412)

import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')

import os
import glob
import cv2
import math
import pickle
import datetime
import pandas as pd
import numpy

from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import model_from_json
from sklearn.metrics import log_loss

from keras import backend as K
K.set_image_dim_ordering('th')

from matplotlib import pyplot as plt #for imshow
from sklearn.feature_extraction import image #image patching

import warnings #turn-off warnings
warnings.filterwarnings("ignore",".*GUI is implemented.*")

#------- Parameters -------------#
global nb_classes
global img_rows
global img_cols
global channel

nb_classes = 2
img_rows, img_cols = 90,90 # input image dimensions (be resized to this!)
channel = 1

os.remove("C:\\Users\\labuser\\Google Drive\\Class\\EECS 545\\LocalProject\\Data\\cache\\test.dat")
os.remove("C:\\Users\\labuser\\Google Drive\\Class\\EECS 545\\LocalProject\\Data\\cache\\train.dat")

def get_im(path):
    # Load as grayscale
    img = cv2.imread(path, 0)
    # Reduce size
    resized = cv2.resize(img, (img_rows, img_cols))
    return resized


def load_train():
    X_train = []
    y_train = []
    print('Read train images')
    for j in range(1,3):
        print('Load folder c{}'.format(j))
        path = os.path.join('.', 'ts_imgs', 'train', str(j), '*.jpg')
        files = glob.glob(path)
        for fl in files:
            img = get_im(fl)
            X_train.append(img)
            y_train.append(j)
            #~ print('cnt = ',cnt,'j = ',j)
    return X_train, y_train

def load_test():
    X_test = []
    y_test = []
    print('Read test images')
    for j in range(1,3):
        print('Load folder test_c{}'.format(j))
        path = os.path.join('.', 'ts_imgs', 'test', str(j), '*.jpg')
        files = glob.glob(path)
        for fl in files:
            img = get_im(fl)
            X_test.append(img)
            y_test.append(j)

    return X_test, y_test

def cache_data(data, path):
    if os.path.isdir(os.path.dirname(path)):
        file = open(path, 'wb')
        pickle.dump(data, file)
        file.close()
    else:
        print('Directory doesnt exists')

def restore_data(path):
	data = dict()
	if os.path.isfile(path):
		file = open(path,'rb')
		data = pickle.load(file)
	return data

#~ def restore_data(path):
	#~ data = dict()
    #~ if os.path.isfile(path):
		#~ file = open(path, 'rb')
		#~ data = pickle.load(file)
	#~ return data


cache_path = os.path.join('cache', 'train.dat')
if not os.path.isfile(cache_path):
	train_data, train_target = load_train()
	cache_data((train_data, train_target), cache_path)
else:
	print('Restore train from cache!')
	(train_data, train_target) = restore_data(cache_path)

cache_path = os.path.join('cache', 'test.dat')
if not os.path.isfile(cache_path):
	test_data, test_target = load_test()
	cache_data((test_data, test_target), cache_path)
else:
	print('Restore train from cache!')
	(test_data, test_target) = restore_data(cache_path)



train_data = np.array(train_data, dtype=np.uint8)
train_target = np.array(train_target, dtype=np.uint8)

test_data = np.array(test_data, dtype=np.uint8)
test_target = np.array(test_target, dtype=np.uint8)

# imshow
#~ plt.imshow(train_data[0])
#~ plt.pause(1)


#patches = image.extract_patches_2d(train_data[0], (50, 50)) # dense griding patches
#print(patches.shape)

train_data = train_data.reshape(train_data.shape[0], channel, img_rows, img_cols)
train_target = np_utils.to_categorical(train_target, nb_classes)
train_data = train_data.astype('float32')
train_data /= 255
print('Train feat:', train_data.shape)
print('Train target:', train_target.shape)


test_data = test_data.reshape(test_data.shape[0], channel, img_rows, img_cols)
test_target = np_utils.to_categorical(test_target, nb_classes)
test_data = test_data.astype('float32')
test_data /= 255

X_train = train_data
Y_train = train_target
X_test = test_data
Y_test = test_target

# 7. Define model architecture
model = Sequential()

model.add(Convolution2D(32, 5, 5, activation='relu', input_shape=(channel, img_rows, img_cols)))
model.add(MaxPooling2D(pool_size=(4,4)))
model.add(Convolution2D(32, 5, 5, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes, activation='softmax'))

# 8. Compile model
model.compile(loss='categorical_crossentropy',
			  optimizer='adam',
			  metrics=['accuracy'])

# 9. Fit model on training data
model.fit(X_train, Y_train, 
		  batch_size=5, epochs=2, verbose=1)

# 10. Evaluate model on test data
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test accuarcy:', str(score))


# 11. Predicted labels
predictions = model.predict([X_test])
print('Prediction shape:', predictions.shape)
print("Prediction: %s" % str(predictions[0:5]))
print("Target: %s" % str(Y_test[0:5]))  # only show first 2 probas

real_labels = np.argmax(Y_test, axis=1)
p_labels = np.argmax(predictions, axis=1)

print('Recognition Rate:', 100*float(sum(real_labels == p_labels))/float(len(p_labels)))

print(model)
print(real_labels)
print(p_labels)
print(predictions)
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)
print(train_data.shape)
print(test_data.shape)
print(Y_test)


# Visualization of trained kernels
def plot_filters(layer, x, y):
	#~ print(layer.kernel)
	filters = K.get_value(layer.kernel)
	print(filters.shape)
	fig = plt.figure()
	for j in range(32):
		ax = fig.add_subplot(y, x, j+1)
		ax.matshow(filters[:,:,0,j]) #for 2nd layer, 3rd arg should also be variable! (caz its 3x3x32x32, not 3x3x1x32) 
		plt.xticks(np.array([]))
		plt.yticks(np.array([]))
	plt.tight_layout()
	plt.show()
	return plt

def visualizeThings():
	visualize_filters = 1
	if 	visualize_filters:
		plot_filters(model.layers[0], 8,4)
		plot_filters(model.layers[1], 8,4)

visualizeThings()
