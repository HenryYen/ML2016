import numpy as np    #Setting:   14feature + 5hour
import random

def gradient(train_data):    
    learning_rate = 1.
    lambda_rate = 1.   #for regularization
    b = random.uniform( -2, 2 )
    w = np.array([random.uniform( -2, 2 ) for _ in range(factor_size * model_houramount * model_power)])
    adagrad_b = 0.
    adagrad_w = np.array([0. for _ in range(factor_size * model_houramount * model_power)])
        
    for t in range(75001):        # 75000 epoch
        gradient_val_b = 0.
        gradient_val_w = np.array([0. for _ in range(factor_size * model_houramount * model_power)])
        train_hour = len(train_data[0])
        for i in range(train_hour - model_houramount):         
            data_each_hour = [entry[i] for entry in train_data] 
            for p in range(model_houramount - 1):            
                for q in range(factor_size):
                    data_each_hour.append(train_data[q][i+1+p])
            size_each_power = len(data_each_hour)
            for p in range(model_power - 1):            
                for q in range(size_each_power):
                    data_each_hour.append(data_each_hour[q]**(p + model_power))
            data_each_hour = np.array(data_each_hour)
            predict_pm25 = b + np.sum(w * data_each_hour)
            real_pm25 = train_data[0][i+model_houramount]
            gradient_val_b += 2 * (real_pm25 - predict_pm25) * (-1)          
            gradient_val_w += [2 * (real_pm25 - predict_pm25) * (-x) for x in data_each_hour]  + 2 * lambda_rate * w
        adagrad_b += gradient_val_b**2
        adagrad_w += gradient_val_w**2   
        b = b - gradient_val_b * learning_rate * (1./(adagrad_b**0.5)) 
        w = w - gradient_val_w * learning_rate * (1./(adagrad_w**0.5))   
    return (b, w)

    
def run_result(b, w):
    test_x = [0 for _ in range(factor_size * model_houramount)]
    result = []
    with open('test_X.csv', 'r') as f:
        counter = 0
        for line in f:
            counter += 1        
            parts = line.split(',')
            if parts[1] in factor_choice:
                for i in range(model_houramount):
                    data_inline = parts[11 - model_houramount + i].rstrip('\n')
                    value = 0 if data_inline == 'NR'  else float(data_inline)           
                    test_x[factor_choice.index(parts[1]) + i * factor_size] = value
            if counter % dimension_per_day == 0:
                size_each_power = len(test_x)
                for p in range(model_power - 1):           
                    for q in range(size_each_power):
                        test_x.append(test_x[q]**(p + model_power))
                predict_rlt = do_predict_model(b, w, np.array(test_x))
                result.append(predict_rlt)
                test_x = [0 for _ in range(factor_size * model_houramount)]
    return result
    
    
def write_result(result):
    filename = 'linear_regression.csv'
    with open(filename, 'w') as _f:
        with open(filename, 'a+') as outf:
            id_idx = 0
            outf.write("id,value\n")
            for value in result:
                content = "id_" + str(id_idx) + "," + str(value) + '\n'
                outf.write(content)
                id_idx += 1


def do_predict_model(b, w, x):
	return b + np.sum(w * x)
	
factor_choice = ["PM2.5", "CH4", "CO", "NMHC", "NO", "NO2", "NOx", "O3", "PM10", "RAINFALL",  "SO2", "THC", "WD_HR", "WS_HR"]  
train_dict = {key:[]  for key in factor_choice}
factor_idx = 2     
factor_size = len(factor_choice)  
dimension_per_day = 18
model_power = 1     
model_houramount = 5

with open('train.csv', 'r') as f:
    for line in f:
        parts = line.split(',')
        factor = parts[factor_idx]
        if factor in train_dict:
            for i in range(factor_idx + 1, len(parts)):
                data_inline = parts[i].rstrip('\n')
                value = 0 if data_inline == 'NR'  else float(data_inline) 
                train_dict[factor].append(value)
                
train_data = [np.array(train_dict[key]) for key in factor_choice]      
(b, w) = gradient(train_data)
result = run_result(b, w)
write_result(result) 