import numpy as np
import pickle
import sys

fn_model = sys.argv[1]
fn_test_set  = sys.argv[2]      # filename of test set
fn_result = sys.argv[3]


def read_csv(filename):
    data = []
    with open(filename, 'r', encoding='mac_roman') as f:
        for line in f:
            parts = line.split(',')
            data.append([float(e) for e in parts])
    return data
    
    
def write_csv(filename, data):
    index = 1
    with open(filename, 'w'):
        with open(filename, 'a+') as f:
            f.write('id,label\n')
            for val in data:
                f.write(str(index) + "," + str(val) + '\n')
                index += 1
                
def read_pickle(fn_model):
    with open(fn_model+'.pkl', 'rb') as f:
        mylist = pickle.load(f)
    return mylist
            
def run_result(test_data, b, w):
    result = []
    for n in test_data:
        x = n[1:]
        f = b + np.sum(w * x)
        ans = 1 if f > 0.5 else 0
        result.append(ans)
    return result  
    
    
test_data = read_csv(fn_test_set)
(b, w) = read_pickle(fn_model)
result = run_result(test_data, b, w)
write_csv(fn_result, result)