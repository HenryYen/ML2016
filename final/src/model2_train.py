import numpy as np
import preprocess as pp
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.ensemble import RandomForestClassifier
import pickle as pk
from sklearn.externals import joblib
from sklearn import preprocessing

path = '../data'
fn_train_data = path + '/train'
fn_subtrain_data = path + '/subtrain'
fn_test_data = path + '/test.in'
fn_model = 'model'
fn_normalizer = 'normalizer'
fn_output = 'out.csv'

n_estimators = 200
isNormalize = False
use_LSA = False
nb_LSAcomponent = 35


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


def save_model(fn, model):
    output = open(fn, 'wb')
    pk.dump(model, output)    
          
def write_csv(fn, data):
    index = 1
    with open(fn, 'w'):
        with open(fn, 'a+') as f:
            f.write('id,label\n')
            for val in data:
                f.write(str(index) + "," + str(val) + '\n')
                index += 1
          
if __name__ == '__main__':
    try:
        print '***load data...'
        (X_train, y_train) = load_data(fn_subtrain_data, 0, 10000000)
                
        if isNormalize:
            normalizer = preprocessing.Normalizer().fit(X_train)
            X_train = normalizer.transform(X_train)
            save_model(fn_normalizer, normalizer)
            
        if use_LSA:
            svd = TruncatedSVD(nb_LSAcomponent)       
            normalizer = Normalizer(copy=False)
            lsa = make_pipeline(svd, normalizer)
            X_train = lsa.fit_transform(X_train)
               
        print 'X_train:', X_train.shape
        print 'y_train:', y_train.shape
        
        
        print '***begin to train...'
        model = RandomForestClassifier(n_estimators=n_estimators, n_jobs=2, verbose=3)
        model.fit(X_train, y_train)
        save_model(fn_model, model)
        
        
        X_test = load_test_data()
        if use_LSA:            
            X_test = lsa.transform(X_test)
        result = model.predict(X_test)
        write_csv(fn_output, result)
        
    
    except KeyboardInterrupt:  
        save_model(fn_model, model)
        
        
        
