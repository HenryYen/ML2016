import pickle
import numpy as np
import atexit
import sys
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model, model_from_json, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import regularizers


data_path = sys.argv[1]         #'./data/'
fn_label = data_path + '/all_label.p'
fn_unlabel = data_path + '/all_unlabel.p'
fn_test = data_path + '/test.p'
fn_model = sys.argv[2]      #'trained_model'

batch_size = 128
nb_classes = 10
nb_epoch = 100
nb_label_data = 5000
nb_unlabel_data = 45000
nb_test = 10000
img_rows, img_cols = 32, 32
img_channels = 3
confidence_thresh = 0.8
encode_dim = 256


def load_label_data():
    label_data = pickle.load(open(fn_label, 'rb'))
    nb_class = len(label_data)  # 10
    nb_piece = len(label_data[0])   # 500
    y_train = []

    X_train = np.array(label_data).reshape(nb_label_data, 3072)
    for i in range(nb_class):
        y_train += [[i] for _ in range(nb_piece)]
    del label_data  
    y_train = np.array(y_train)  

    return (X_train, y_train)

    
def load_unlabel_data():
    unlabel_data = pickle.load(open(fn_unlabel, 'rb'))
    X_unlabel = np.array(unlabel_data).reshape(nb_unlabel_data, 3072)
    del unlabel_data
    
    return X_unlabel
        
def load_test_data():
    test_data = pickle.load(open(fn_test, 'rb'))
    X_test = np.array(test_data['data']).reshape(nb_test, 3072)
    del test_data
    
    return X_test

def store_model(fn, model):
    model.save(fn)
    
       
def read_model(fn):
    loaded_model = load_model(fn)
    return loaded_model

    
def modify_label(model, encoder, X_train_code, Y_train, X_unlabel_code):
    label_list = []
    X_append_list = []    
    confidence = model.predict(X_unlabel_code).tolist()
    X_unlabel_code = X_unlabel_code.tolist()
    for img in confidence:
        img_idx = confidence.index(img)
        max_confidence = max(img)
        if max_confidence > confidence_thresh:
            label = img.index(max_confidence)
            label_list.append([label])
            X_append_list.append(X_unlabel_code[img_idx])
    for e in X_append_list:
        X_unlabel_code.remove(e)
    """
    label_list = model.predict_classes(X_unlabel)
    label_list = [[e] for e in label_list]
    """
    bin_label = np_utils.to_categorical(label_list, nb_classes)
    if len(X_append_list) != 0:
        Y_train = np.concatenate((Y_train, bin_label), axis=0)
        X_train_code = np.concatenate((X_train_code, np.array(X_append_list)), axis=0)
    
    del confidence
    del label_list
    del bin_label
    del X_append_list
    return (X_train_code, Y_train, np.array(X_unlabel_code))


def get_code(encoder, X_batch):
    return  encoder.predict(X_batch)
    
	
def build_autoencoder():
    input_img = Input(shape=(3072,))
    inner = Dense(768, activation='relu')(input_img)
    inner = Dense(512, activation='relu')(inner)
    encoded = Dense(encode_dim, activation='relu')(inner)
    
    inner = Dense(512, activation='relu')(encoded)
    inner = Dense(768, activation='relu')(inner)
    decoded = Dense(3072, activation='sigmoid')(inner)

    encoder = Model(input=input_img, output=encoded)
    autoencoder = Model(input=input_img, output=decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return (autoencoder, encoder)


def build_model():
    model = Sequential()
    model.add(Dense(512, input_dim=encode_dim, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    return model    
  
    
if __name__ == '__main__':
    time = 0
    try:
        (X_train, y_train) = load_label_data()        
        X_unlabel = load_unlabel_data()
        X_test = load_test_data()

        X_unlabel = np.concatenate((X_unlabel, X_test))
        del X_test
        Y_train = np_utils.to_categorical(y_train, nb_classes)        
        del y_train
        X_train = X_train.astype('float32') / 255.
        X_unlabel = X_unlabel.astype('float32') / 255.
        #X_test = X_test.astype('float32') / 255.
        
        auto_train = np.concatenate((X_train, X_unlabel))
        #auto_train = np.concatenate((auto_train, X_test))
        (autoencoder, encoder) = build_autoencoder()
        autoencoder.fit(auto_train, auto_train, batch_size=256, nb_epoch=20, shuffle=True)
        del auto_train        
        store_model('encoder', encoder)
                
        X_train_code = get_code(encoder, X_train)
        del X_train
        X_unlabel_code = get_code(encoder, X_unlabel)
        del X_unlabel 
        #X_test_code = get_code(encoder, X_test)
        #del X_test   
        model = build_model()
        model.fit(X_train_code, Y_train, batch_size=batch_size, nb_epoch=180, shuffle=True)
        store_model(fn_model, model)
                      
        
        for time in range(4):    
            print '======Time:',time, '=======' 
            
            (X_train_code, Y_train, X_unlabel_code) = modify_label(model, encoder, X_train_code, Y_train, X_unlabel_code)
            #(X_train_code, Y_train, X_test_code) = modify_label(model, encoder, X_train_code, Y_train, X_test_code)
            print('X_train_code shape:', X_train_code.shape)
            print('Y_train shape:', Y_train.shape)
            print('X_unlabel_code shape:', X_unlabel_code.shape)
            #print('X_test_code shape:', X_test_code.shape)
            model = build_model()
            model.fit(X_train_code, Y_train, batch_size=batch_size, nb_epoch=80, shuffle=True)
            store_model(fn_model, model)
        
				                
    except KeyboardInterrupt:
        print '\n======Time:',time, '=======' 
        store_model('encoder', encoder)
        store_model(fn_model, model)
        




