import pickle
import numpy as np
import atexit
import sys
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils


data_path = sys.argv[1]     #'./data/'
fn_label = data_path + '/all_label.p'
fn_unlabel = data_path + '/all_unlabel.p'
fn_test = data_path + '/test.p'
fn_model = sys.argv[2]      #'trained_model'

batch_size = 128
nb_classes = 10
nb_epoch = 60
nb_label_data = 5000
nb_unlabel_data = 45000
nb_test = 10000
img_rows, img_cols = 32, 32
img_channels = 3
confidence_thresh = 0.9


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

    return (X_train, y_train)

    
def load_unlabel_data():
    unlabel_data = pickle.load(open(fn_unlabel, 'rb'))
    X_unlabel = np.array(unlabel_data).reshape(nb_unlabel_data, 3, 32, 32)
    del unlabel_data
    
    return X_unlabel
        
def load_test_data():
    test_data = pickle.load(open(fn_test, 'rb'))
    X_test = np.array(test_data['data']).reshape(nb_test, 3, 32, 32)
    del test_data
    
    return X_test

def store_model(model):
    model.save(fn_model)
    
       
def read_model():
    loaded_model = load_model(fn_model)
    return loaded_model

    
def modify_label(model, X_train, Y_train, X_unlabel):
    label_list = []
    X_append_list = []    
    confidence = model.predict(X_unlabel).tolist()
    X_unlabel = X_unlabel.tolist()
    for img in confidence:
        img_idx = confidence.index(img)
        max_confidence = max(img)
        if max_confidence > confidence_thresh:
            label = img.index(max_confidence)
            label_list.append([label])
            X_append_list.append(X_unlabel[img_idx])
    for e in X_append_list:
        X_unlabel.remove(e)
    """
    label_list = model.predict_classes(X_unlabel)
    label_list = [[e] for e in label_list]
    """
    bin_label = np_utils.to_categorical(label_list, nb_classes)
    if len(X_append_list) != 0:
        Y_train = np.concatenate((Y_train, bin_label), axis=0)
        X_train = np.concatenate((X_train, np.array(X_append_list)), axis=0)
    
    del confidence
    del label_list
    del bin_label
    del X_append_list
    return (X_train, Y_train, np.array(X_unlabel))


def build_model():
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='valid', dim_ordering = 'th', input_shape=(3, 32, 32)))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    return model
    
  
    
if __name__ == '__main__':
    time = 0
    try:
        (X_train, y_train) = load_label_data()
        X_unlabel = load_unlabel_data()
        X_test = load_test_data()

        X_train = X_train.astype('float32') / 255.
        X_unlabel = X_unlabel.astype('float32') / 255.
        X_test = X_test.astype('float32') / 255.
        Y_train = np_utils.to_categorical(y_train, nb_classes)
        del y_train
        
        model = build_model()
        model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=90, verbose=1, shuffle=True)
        store_model(model)

        for time in range(3):    
            print '======Time:',time, '======='         
                        
            (X_train, Y_train, X_unlabel) = modify_label(model, X_train, Y_train, X_unlabel)
            (X_train, Y_train, X_test) = modify_label(model, X_train, Y_train, X_test)
            print('X_train shape:', X_train.shape)
            print('Y_train shape:', Y_train.shape)
            print('X_unlabel shape:', X_unlabel.shape)
            print('X_test shape:', X_test.shape)            
            model = build_model()
            model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=60, shuffle=True)
            store_model(model)
                         
                 
    except KeyboardInterrupt:   
        print '======Time:',time, '======='         
        store_model(model)
        



