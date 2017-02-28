import os
import csv
import cv2
import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras import backend
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda, SpatialDropout2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
from keras.activations import relu
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator


class AutonomousDriver:

    def __init__(self, driving_log, driving_images, validation_split, batch_size, epochs, learning_rate, all_cameras, model_file):
        self.__driving_log    = driving_log
        self.__driving_images = driving_images
        self.__valid_split    = validation_split
        self.__batch_size     = batch_size
        self.__epochs         = epochs
        self.__learning_rate  = learning_rate
        self.__all_cameras    = all_cameras
        self.__model_file     = model_file
        self.__samples        = []
        self.__train_samples  = []
        self.__valid_samples  = []
        self.__model          = self._prepare_model()
        self.__train_history  = None

    def read_data(self):
        with open(self.__driving_log) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                self.__samples.append(line)
        print('Total log samples: ', len(self.__samples))

    def preprocess(self):
        self.__samples = shuffle(self.__samples)
        self.__train_samples, self.__valid_samples = train_test_split(self.__samples, test_size=self.__valid_split, random_state=42)

    def _prepare_model(self):
        row, col, ch = 160, 320, 3

        # Model Architecture
        model = Sequential()
        model.add(Cropping2D(cropping=((60,20), (0,0)), dim_ordering='tf', input_shape=(row, col, ch)))
        model.add(Lambda(lambda x: (x/255) - 0.5))
        model.add(SpatialDropout2D(0.2))
        # Layer 1: CONV
        model.add(Convolution2D(24, 5, 5))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(SpatialDropout2D(0.2))
        # Layer 2: CONV
        model.add(Convolution2D(36, 5, 5))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(SpatialDropout2D(0.2))
        # Layer 3: CONV
        model.add(Convolution2D(48, 3, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(SpatialDropout2D(0.2))
        # Layer 4: CONV
        model.add(Convolution2D(64, 3, 3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(SpatialDropout2D(0.2))

        model.add(Flatten())

        # Layer 5: FC
        model.add(Dense(100))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        # Layer 6: FC
        model.add(Dense(50))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        # Layer 7: FC
        model.add(Dense(10))
        model.add(Activation('relu'))

        model.add(Dense(1))

        model.compile(optimizer=Adam(self.__learning_rate), loss='mse')
        return model

    def train(self):
        train_generator = self._generator(self.__train_samples, self.__batch_size, self.__all_cameras)
        valid_generator = self._generator(self.__valid_samples, self.__batch_size, False)

        train_samples_count = len(self.__train_samples) * 3  if self.__all_cameras else len(self.__train_samples)
        valid_samples_count = len(self.__valid_samples)
        
        print('Total training samples:   ', train_samples_count)
        print('Total validation samples: ', valid_samples_count)

        self.__train_history = self.__model.fit_generator(
            generator         = train_generator,
            samples_per_epoch = train_samples_count,
            validation_data   = valid_generator,
            nb_val_samples    = valid_samples_count,
            nb_epoch          = epochs,
            verbose           = 1
        )

    def save_model(self):
        self.__model.save(self.__model_file)
        print('Model saved to ', self.__model_file)

    def visualize_loss(self):
        print(self.__train_history.history.keys())

        plt.plot(self.__train_history.history['loss'])
        plt.plot(self.__train_history.history['val_loss'])
        plt.title('model mean squared error loss')
        plt.ylabel('mean squared error loss')
        plt.xlabel('epoch')
        plt.legend(['training set', 'validation set'], loc='upper right')
        plt.show()

    def cleanup(self):
        backend.clear_session()

    def _generator(self, samples, batch_size, all_cameras=False):
        num_samples = len(samples)
        while 1: # Loop forever so the generator never terminates
            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset:offset+batch_size]

                images = []
                angles = []
                for batch_sample in batch_samples:
                    center_name  = self.__driving_images + batch_sample[0].split('/')[-1]
                    center_image = cv2.imread(center_name)
                    center_angle = float(batch_sample[3])
                    
                    images.append(center_image)
                    angles.append(center_angle)

                    # Use left and right camera                
                    if all_cameras:
                        left_name   = self.__driving_images + batch_sample[1].split('/')[-1]
                        right_name  = self.__driving_images + batch_sample[2].split('/')[-1]

                        left_image  = cv2.imread(left_name)
                        right_image = cv2.imread(right_name)
                        left_angle  = center_angle + 0.25
                        right_angle = center_angle - 0.25

                        images.append(left_image)
                        images.append(right_image)
                        angles.append(left_angle)
                        angles.append(right_angle)

                X_train = np.array(images)
                y_train = np.array(angles)

                # Flip to combat left turn bias
                flips = random.sample(range(X_train.shape[0]), int(X_train.shape[0]/2))
                X_train[flips] = X_train[flips, :, ::-1, :]
                y_train[flips] = -y_train[flips]

                yield (X_train, y_train)

    def visualize_data(self):    
        # visualize an image
        name = self.__driving_images + self.__train_samples[0][0].split('/')[-1]
        img = Image.open(name)
        w = img.size[0]
        h = img.size[1]
        img_crop = img.crop((0, 60, w, h-20))
        plt.imshow(img_crop)
        plt.show()


#__main__
driving_log      = './data/driving_log.csv'
driving_images   = './data/IMG/'
#driving_log      = './data1/driving_log.csv'
#driving_images   = './data1/IMG/'

validation_split = 0.2
batch_size       = 16
epochs           = 100
learning_rate    = 0.0001
all_cameras      = True
model_file       = 'model.h5'

driver = AutonomousDriver(driving_log, driving_images, validation_split, batch_size, epochs, learning_rate, all_cameras, model_file)
driver.read_data()
driver.preprocess()
driver.visualize_data()
driver.train()
driver.visualize_loss()
driver.save_model()
driver.cleanup()
