import numpy as np
import pickle
import sys
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils


nb_test = 10000
data_path = sys.argv[1]             #'./data/'
fn_test = data_path + '/test.p'
fn_model = sys.argv[2]              #'trained_model'
fn_output = sys.argv[3]             #'auto_output.csv'


def load_test_data():
    test_data = pickle.load(open(fn_test, 'rb'))
    X_test = np.array(test_data['data']).reshape(nb_test, 3072)
    del test_data
    
    return X_test
    
def get_code(encoder, X_batch):
    return  encoder.predict(X_batch)
    
def predict_test():
    encoder = read_model('encoder')
    model = read_model(fn_model)
    X_test = load_test_data()
    X_test_code = get_code(encoder, X_test)
    result = model.predict_classes(X_test_code)
    write_csv(fn_output, result)
    
    
def write_csv(fn, data):
    index = 0
    with open(fn, 'w'):
        with open(fn, 'a+') as f:
            f.write('ID,class\n')
            for val in data:
                f.write(str(index) + "," + str(val) + '\n')
                index += 1
                
def read_model(fn):
    loaded_model = load_model(fn)
    return loaded_model
    
if __name__ == '__main__':
    predict_test()
