import csv
import cv2
import numpy as np
import sklearn
import random
from sklearn.model_selection import train_test_split

# append the original image including angle
# and its flipped image
def load_image_and_measurement(img_path, measurement, images, measurements):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    images.append(img)
    measurements.append(measurement)

    images.append(cv2.flip(img, 1))
    measurements.append(measurement * (-1.0))

def load_samples():
    samples = []
    with open('simulator_data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    return samples

def generator(samples, batch_size = 32, correction = 0.2):
    num_samples = len(samples)
    while 1:
        random.shuffle(samples)

        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            measurements = []
            for line in batch_samples:
                source_path = line[0]
                measurement = float(line[3])

                img_center = line[0].split('/')[-1]
                img_left   = line[1].split('/')[-1]
                img_right  = line[2].split('/')[-1]
                img_dir    = 'simulator_data/IMG/'

                load_image_and_measurement(img_dir + img_center, measurement,
                        images, measurements)
                load_image_and_measurement(img_dir + img_left, measurement + correction,
                        images, measurements)
                load_image_and_measurement(img_dir + img_right, measurement - correction,
                        images, measurements)

            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)


from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Convolution2D, Cropping2D
from keras.layers import MaxPooling2D

def preprocess_image():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5,
        input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70,25), (0,0))))
    return model

def LeNet():
    model = preprocess_image()
    model.add(Convolution2D(6, 5, 5, activation = "relu"))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6, 5, 5, activation = "relu"))
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    return model

def Nvidia():
    model = preprocess_image()
    model.add(Convolution2D(24, 5, 5, subsample = (2, 2), activation = 'relu'))
    model.add(Convolution2D(36, 5, 5, subsample = (2, 2), activation = 'relu'))
    model.add(Convolution2D(48, 5, 5, subsample = (2, 2), activation = 'relu'))
    model.add(Convolution2D(64, 3, 3, activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(Convolution2D(64, 3, 3, activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(1))
    return model


samples = load_samples()
train_samples, validation_samples = train_test_split(samples, test_size = 0.2)

train_generator = generator(train_samples, batch_size = 32)
valid_generator = generator(validation_samples, batch_size = 32)

#model = LeNet()
model = Nvidia()

model.compile(loss = 'mse', optimizer = 'adam')
history = model.fit_generator(train_generator, samples_per_epoch = len(train_samples) * 3,
                    validation_data = valid_generator, nb_val_samples = len(validation_samples) * 3,
                    nb_epoch = 8, verbose = 1)

model.save('model.h5')

# generating plot
print(history.history.keys())

import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('')
plt.xlabel('')
plt.legend(['training set', 'validation set'], loc = 'upper right')
plt.savefig('result/train_valid_loss.png')
plt.show()
