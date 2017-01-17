import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation , Dropout
from keras.optimizers import Adam
from keras.utils import np_utils
import preprocess as pp
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

path = '../data'
fn_train_data = path + '/train'
fn_test_data = path + '/test.in'
fn_output = 'out.csv'
fn_model = 'model'

batch_size = 256
nb_classes = 5
nb_epoch = 1
use_LSA = False
nb_feature = 41
nb_LSAcomponent = 20


def load_data(fn, begin, end):
    X_train = []
    y_train = []
    counter = 0
    duration =  end - begin
    
    with open(fn, 'r') as f:
        for idx, line in enumerate(f):
            if counter == duration:
                break
            if idx >= begin:                
                parts = pp.preprocess(line)
                X_train.append(parts[:41])
                y_train.append(int(parts[41]))            
                counter += 1
    return (np.array(X_train), np.array(y_train))
    

def load_test_data():
    X_test = []    
    with open(fn_test_data, 'r') as f:
        for line in f:
            X_test.append(pp.preprocess(line))
    return np.array(X_test)
    
    
def write_csv(fn, data):
    index = 1
    with open(fn, 'w'):
        with open(fn, 'a+') as f:
            f.write('id,label\n')
            for val in data:
                f.write(str(index) + "," + str(val) + '\n')
                index += 1
                

def build_model():     
    model = Sequential()
    model.add(Dense(output_dim=256, input_dim=nb_feature))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation("softmax"))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
    
    
if __name__ == '__main__':
    try:
        print '***load data...'
        (X_train, y_train) = load_data(fn_train_data, 0, 2400000)
        Y_train = np_utils.to_categorical(y_train, nb_classes)
        
        if use_LSA:
            svd = TruncatedSVD(nb_LSAcomponent)       
            normalizer = Normalizer(copy=False)
            lsa = make_pipeline(svd, normalizer)
            X_train = lsa.fit_transform(X_train)
        
        print '***begin to train...'
        model = build_model()
        model.fit(X_train, Y_train, nb_epoch=nb_epoch, batch_size=batch_size,  shuffle=True)
        model.save(fn_model)
        
        
        X_test = load_test_data()
        result = model.predict_classes(X_test)
        write_csv(fn_output, result)
    
    except KeyboardInterrupt:           
        model.save(fn_model)
        
        
        
