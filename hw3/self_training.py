import pickle
import numpy as np
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils


data_path = './data/'
fn_label = data_path + 'all_label.p'
fn_unlabel = data_path + 'all_unlabel.p'
fn_test = data_path + 'test.p'

batch_size = 128
nb_classes = 10
nb_epoch = 1
nb_label_data = 5000
nb_unlabel_data = 45000
nb_test = 10000
data_augmentation = True
img_rows, img_cols = 32, 32
img_channels = 3





def load_label_data():
    label_data = pickle.load(open(fn_label, 'rb'))
    nb_class = len(label_data)  # 10
    nb_piece = len(label_data[0])   # 500
    y_train = []

    X_train = np.array(label_data).reshape(nb_label_data, 3, 32, 32)
    for i in range(nb_class):
        y_train += [[i] for _ in range(nb_piece)]
    del label_data  
    y_train = np.array(y_train)  

    """
    for i in range(nb_class):
        for j in range(nb_piece):
            y_train.append([i])
            X_train.append(np.array(label_data[i][j]).reshape(3, 32, 32))
            tmp_row = []
            tmp_img = []
            for m in range(img_rows * img_cols):
                RGB = [label_data[i][j][m], label_data[i][j][m+1024], label_data[i][j][m+2048]]
                tmp_row.append(RGB)
                if len(tmp_row) == img_cols:
                    tmp_img.append(tmp_row)
                    tmp_row = []    # reinitial
                if len(tmp_img) == img_rows:
                    X_train.append(tmp_img)
    X_train = np.array(X_train)
    """
    return (X_train, y_train)

    
def load_unlabel_data():
    unlabel_data = pickle.load(open(fn_unlabel, 'rb'))
    X_unlabel = np.array(unlabel_data).reshape(nb_unlabel_data, 3, 32, 32)
    del unlabel_data
    
    """
    for i in range(nb_piece):
        tmp_row = []
        tmp_img = []
        for m in range(img_rows * img_cols):
            RGB = [unlabel_data[i][m], unlabel_data[i][m+1024], unlabel_data[i][m+2048]]
            tmp_row.append(RGB)
            if len(tmp_row) == img_cols:
                tmp_img.append(tmp_row)
                tmp_row = []    # reinitial
            if len(tmp_img) == img_rows:
                X_unlabel.append(tmp_img)
    X_unlabel = np.array(X_unlabel)
    """
    return X_unlabel
    
def load_test_data():
    test_data = pickle.load(open(fn_test, 'rb'))
    X_test = np.array(test_data['data']).reshape(nb_test, 3, 32, 32)
    del test_data
    
    return X_test
    

    
def modify_label(model, X_train, Y_train, X_unlabel):
    label_list = []
    
    confidence = model.predict(X_unlabel).tolist()
    for img in confidence:
        label = img.index(max(img))
        label_list.append([label])
    bin_label = np_utils.to_categorical(label_list, nb_classes)
    print bin_label.shape
    print Y_train.shape
    Y_train = np.concatenate((Y_train, bin_label), axis=0)
    X_train = np.concatenate((X_train, X_unlabel), axis=0)
    
    del confidence
    del label_list
    del bin_label
    return (X_train, Y_train, X_unlabel)



    
if __name__ == '__main__':
    #(X_train, y_train), (X_test, y_test) = cifar10.load_data()
    (X_train, y_train) = load_label_data()
    X_unlabel = load_unlabel_data()
    #print X_train.shape, y_train.shape
    #print X_unlabel.shape

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    #Y_test = np_utils.to_categorical(y_test, nb_classes)

    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='same', dim_ordering = 'th', input_shape=X_train.shape[1:]))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(64, 3, 3))
    model.add(MaxPooling2D(pool_size=(2, 2)))



    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    # let's train the model using SGD + momentum (how original).
    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    X_train = X_train.astype('float32')
    #X_test = X_test.astype('float32')
    X_train /= 255
    #X_test /= 255

    
    while True:
        print('X_train shape:', X_train.shape)
        print(X_train.shape[0], 'label samples')
        print(X_unlabel.shape[0], 'unlabel samples')
        #print(X_test.shape[0], 'test samples')

        if not data_augmentation:
            print('Not using data augmentation.')
            model.fit(X_train, Y_train,
                      batch_size=batch_size,
                      nb_epoch=nb_epoch,
                      shuffle=True, validation_split=0.2)
        else:
            print('Using real-time data augmentation.')

            # this will do preprocessing and realtime data augmentation
            datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=True,  # randomly flip images
                vertical_flip=False)  # randomly flip images

            # compute quantities required for featurewise normalization
            # (std, mean, and principal components if ZCA whitening is applied)
            datagen.fit(X_train)

            # fit the model on the batches generated by datagen.flow()
            model.fit_generator(datagen.flow(X_train, Y_train,
                                batch_size=batch_size),
                                samples_per_epoch=X_train.shape[0],
                                nb_epoch=nb_epoch)
         #                       validation_data=(X_test, Y_test))

        (X_train, y_train, X_unlabel) = modify_label(model, X_train, y_train, X_unlabel)
        



