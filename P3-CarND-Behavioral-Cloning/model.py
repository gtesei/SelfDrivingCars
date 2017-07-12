from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, ELU, Dropout
from sklearn.model_selection import train_test_split
from keras.backend import tf as ktf
from keras.optimizers import Adam
from scipy.misc import imresize
from keras.callbacks import ModelCheckpoint
import matplotlib.image as mpimg
import cv2
import matplotlib.pyplot as plt
import numpy as np
from random import randint
import argparse

# Helper Functions
def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype = np.float64)
    random_bright = .5+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1[:,:,2][image1[:,:,2]>255]  = 255
    image1 = np.array(image1, dtype = np.uint8)
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def load_image(imagepath):
    im = mpimg.imread(imagepath, 1)
    # + crop
    return imresize(im[45:135, :], (66, 200, 3), interp='bilinear', mode=None)


def load_images(args,csv_file,path,verbose=True):
    car_images, steering_angles = [], []

    if verbose:
        print(">>> Loading images from ",csv_file,"...")

    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            steering_center = float(row[3])

            # create adjusted steering measurements for the side camera images
            correction = 0.2  # this is a parameter to tune
            steering_left = steering_center + correction
            steering_right = steering_center - correction

            # read in images from center, left and right cameras
            img_center = load_image(path + row[0].split('/')[-1])
            img_left = load_image(path + row[1].split('/')[-1])
            img_right = load_image(path + row[2].split('/')[-1])

            # add images and angles to data set
            car_images.extend([img_center, img_left, img_right])
            steering_angles.extend([steering_center, steering_left, steering_right])

            if args.flip_img:
                car_images.extend([np.fliplr(img_center), np.fliplr(img_left), np.fliplr(img_right)])
                steering_angles.extend([-steering_center, -steering_left, -steering_right])

            if args.rand_brightness:
                car_images.extend([augment_brightness_camera_images(img_center)
                                                                   , augment_brightness_camera_images(img_left)
                                                                   , augment_brightness_camera_images(img_right)])
                steering_angles.extend([steering_center, steering_left, steering_right])

    if verbose:
        print(">>> N. Samples:",len(car_images))

    return np.array(car_images), np.array(steering_angles)

# NVIDIA Model
def build_model(args,width=200, height=66, depth=3):
    """
    width: The width of our input images.
    height: The height of our input images.
    depth: The depth (i.e., number of channels) of our input images.

    NVIDIA model used
    Image normalization to avoid saturation and make gradients work better.
    Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
    Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
    Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    Drop out (0.5)
    Fully connected: neurons: 100, activation: ELU
    Fully connected: neurons: 50, activation: ELU
    Fully connected: neurons: 10, activation: ELU
    Fully connected: neurons: 1 (output)
    # the convolution layers are meant to handle feature engineering
    the fully connected layer for predicting the steering angle.
    dropout avoids overfitting
    ELU(Exponential linear unit) function takes care of the Vanishing gradient problem.
    """
    model = Sequential()
    model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=(height,width,depth)))
    model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Dropout(args.keep_prob))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.summary()

    return model


# Train and save model
def train_and_save_model(args):
    """
    Train and save model
    """

    # center lane driving
    X_train1, y_train1 = load_images(args,csv_file=args.data_dir+'/data_2/driving_log.csv', path=args.data_dir+'/data_2/IMG/')
    #  recovering from the left side and right sides of the road back to center
    X_train2, y_train2 = load_images(args,csv_file=args.data_dir+'/data_3/driving_log.csv', path=args.data_dir+'/data_3/IMG/')
    # driving counter-clockwise
    X_train3, y_train3 = load_images(args,csv_file=args.data_dir+'/data_4/driving_log.csv', path=args.data_dir+'/data_4/IMG/')

    # merge train sets
    X_train = np.concatenate((X_train1, X_train2, X_train3), axis=0)
    y_train = np.concatenate((y_train1, y_train2, y_train3), axis=0)

    # build model
    model = build_model(args)

    # train params
    adam = Adam(lr=args.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='mse')

    # train
    if args.vis_loss == 0:
        ### only training without visualizing loss function
        model.fit(X_train, y_train, validation_split=args.test_size, shuffle=True, nb_epoch=args.nb_epoch, batch_size=args.batch_size)
    else:
        ### plot the training and validation loss for each epoch
        history_object = model.fit(X_train, y_train, validation_split=args.test_size, shuffle=True, nb_epoch=args.nb_epoch, batch_size=args.batch_size , verbose=1)
        plt.plot(history_object.history['loss'])
        plt.plot(history_object.history['val_loss'])
        plt.title('model mean squared error loss')
        plt.ylabel('mean squared error loss')
        plt.xlabel('epoch')
        plt.legend(['training set', 'validation set'], loc='upper right')
        plt.show()

    # save
    print(">>> Saving model to ", args.out_mod," ...")
    model.save(args.out_mod)


# Main
def main():
    """
    Load train/validation data set and train the model
    """
    parser = argparse.ArgumentParser(description='Behavioral Cloning Project')
    parser.add_argument('-d', help='data directory', dest='data_dir', type=str, default='data')
    parser.add_argument('-t', help='test size fraction', dest='test_size', type=float, default=0.2)
    parser.add_argument('-k', help='drop out probability', dest='keep_prob', type=float, default=0.5)
    parser.add_argument('-n', help='number of epochs', dest='nb_epoch', type=int, default=3)
    parser.add_argument('-b', help='batch size', dest='batch_size', type=int, default=32)
    parser.add_argument('-l', help='learning rate', dest='learning_rate', type=float, default=1.0e-4)
    parser.add_argument('-v', help='visualize loss', dest='vis_loss', type=float, default=0)
    parser.add_argument('-o', help='output model', dest='out_mod', type=str, default='model.h5')
    parser.add_argument('-f', help='flip images', dest='flip_img', type=float, default=1)
    parser.add_argument('-r', help='rand brightness', dest='rand_brightness', type=float, default=1)
    args = parser.parse_args()

    # print parameters
    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    # load data
    train_and_save_model(args)


if __name__ == '__main__':
    main()
