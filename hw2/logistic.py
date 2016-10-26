import numpy as np
import pickle
import sys

fn_train_set = sys.argv[1]    # filename of train set
fn_model = sys.argv[2]
setting = {"learning_rate":0.1, "lambda":0., "epoch_no":2000}

def sigmoid(z):
    return 1. / (1. + np.exp(-z))
  
     
def gradient(train_data, setting):    
    learning_rate = setting["learning_rate"]
    lambda_rate = setting["lambda"]   #for regularization
    epoch_no = setting["epoch_no"]
    feature_size = 57
    train_mail_no = len(train_data)
    b = 0.
    w = np.array( [0.] * feature_size )
    ada_b = 0.
    ada_w = np.array( [0.] * feature_size )
        
    for epoch in range(epoch_no):        
        grad_b = 0.
        grad_w = np.array( [0.] * feature_size )
        for n in range(train_mail_no):
            y_hat = train_data[n][-1]
            x = np.array(train_data[n][1:-1])
            f = sigmoid(b + np.sum(w * x))
            grad_b += (y_hat - f) * (-1)          
            grad_w += (y_hat - f) * (-x) + 2 * lambda_rate * w
        ada_b += grad_b**2
        ada_w += grad_w**2   
        b = b - grad_b * learning_rate * (1./(ada_b**0.5)) 
        w = w - grad_w * learning_rate * (1./(ada_w**0.5))
    return (b, w)   
    
    
def read_csv(filename):
    data = []
    with open(filename, 'r', encoding='mac_roman') as f:
        for line in f:
            parts = line.split(',')
            data.append([float(e) for e in parts])
    return data
    
    
def write_pickle(fn_model, mylist):
    with open(fn_model+'.pkl', 'wb') as f:
        pickle.dump(mylist, f)
        
    
train_data = read_csv(fn_train_set)
(b, w) = gradient(train_data, setting)
write_pickle(fn_model, [b, w])
