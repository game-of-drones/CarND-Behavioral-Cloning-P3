import csv
import cv2
import numpy as np
import os

relative_path_of_data = './myTraining'
## Read CSV file
lines = []
with open(os.path.join(relative_path_of_data, 'driving_log.csv')) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []

for line in lines:
    center_img = line[0].split(os.sep)[-1]
    #filename = source_path.split(os.sep)[-1]
    current_path = os.path.join(relative_path_of_data,'IMG',center_img)
    image = cv2.imread(current_path)
    measure = float(line[3])
    images.append(image)
    measurements.append(measure)
    # mirror
    images.append(cv2.flip(image,1))
    measurements.append(-measure)

    left_img = line[1].split(os.sep)[-1]
    #filename = source_path.split(os.sep)[-1]
    current_path = os.path.join(relative_path_of_data,'IMG',left_img)
    image = cv2.imread(current_path)
    measure = float(line[3]) + 0.1
    images.append(image)
    measurements.append(measure)
    # mirror
    images.append(cv2.flip(image,1))
    measurements.append(-measure)

    right_img = line[2].split(os.sep)[-1]
    #filename = source_path.split(os.sep)[-1]
    current_path = os.path.join(relative_path_of_data,'IMG',right_img)
    image = cv2.imread(current_path)
    measure = float(line[3]) - 0.1
    images.append(image)
    measurements.append(measure)
    # mirror
    images.append(cv2.flip(image,1))
    measurements.append(-measure)

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Activation, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

from keras.layers import Cropping2D

data = {'X_train':X_train, 'y_train':y_train}

def model_simplest():
    model = Sequential()
    model.add(Cropping2D(cropping=((70,20), (0,0)), input_shape=(160,320,3)))
    model.add(Flatten())
    model.add(Dense(1))
    return model

def model_LeNet():
    model = Sequential()
    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))

    model.add(Cropping2D(cropping=((70,25), (0,0))))

    model.add(Convolution2D(6,5,5, activation='relu'))
    model.add(MaxPooling2D())

    model.add(Convolution2D(6,5,5, activation='relu'))
    model.add(MaxPooling2D())

    model.add(Flatten())

    #model.add(Dense(120))
    model.add(Dense(84))
    model.add(Activation('relu'))    
    model.add(Dense(1))

    return model

def model_nvidia():
    model = Sequential()
    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25), (0,0))))

    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    #model.add(MaxPooling2D())
    model.add(Dropout(0.2))

    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    #model.add(MaxPooling2D())
    model.add(Dropout(0.2))

    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))    
    model.add(Dropout(0.2))

    model.add(Convolution2D(64,3,3, activation='relu'))    

    model.add(Flatten())

    #model.add(Dense(120))
    model.add(Dense(100))
    model.add(Activation('relu'))

    model.add(Dense(50))
    model.add(Activation('relu'))    

    model.add(Dense(10))
    model.add(Activation('relu'))

    model.add(Dense(1))

    return model

def train_and_save(model, data):
    model.compile(loss='mse', optimizer='adam')
    model.fit(data['X_train'], data['y_train'], validation_split=0.3, shuffle=True, nb_epoch=5)
    model.save('model.h5')

train_and_save(model_nvidia(), data)