import pickle

path = './data'
fn_train_data = 'train'

data = []
nb_smurf = 500
nb_normal = 500
nb_neptune  = 500

nb_total = 2000


with open(fn_train_data, 'r') as f:
    for i, line in enumerate(f):
        if i % 13 == 0:   
            parts = line.rstrip().split(',')
            if parts[41] == 'smurf.' and nb_smurf != 0:
                data.append(parts)
                nb_smurf -= 1
            elif parts[41] == 'normal.' and nb_normal != 0:
                data.append(parts)
                nb_normal -= 1
            elif parts[41] == 'neptune.' and nb_neptune != 0:
                data.append(parts)
                nb_neptune -= 1
            elif parts[41] != 'neptune.' and parts[41] != 'normal.' and parts[41] != 'smurf.' :         
                data.append(parts)
                        
            
with open('validset', 'w'):
    with open('validset', 'a+') as f:
        for e in data[:nb_total]:
            f.write(','.join(e)+'\n')
        
        
