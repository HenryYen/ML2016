import numpy as np
import sys

def do_predict_model(b, w, x):
	return b + np.sum(w * x)
 
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
    filename = sys.argv[1]
    with open(filename, 'w') as _f:
        with open(filename, 'a+') as outf:
            id_idx = 0
            outf.write("id,value\n")
            for value in result:
                content = "id_" + str(id_idx) + "," + str(value) + '\n'
                outf.write(content)
                id_idx += 1
                

factor_choice =["PM2.5", "CH4", "CO", "NMHC", "NO", "NO2", "NOx", "O3", "PM10", "RAINFALL",  "SO2", "THC", "WD_HR", "WS_HR"]
factor_size = len(factor_choice)    
dimension_per_day = 18
model_power = 1     
model_houramount = 5


b = -1.36347642373
a = """-3.91635194e-02   8.13537791e-03   1.82640325e-02   6.69056877e-03
   1.98901186e-02  -4.73936636e-02   1.48292362e-02  -1.10784497e-02
   7.64237752e-04  -4.52541460e-02  -9.24887305e-03   1.58102709e-02
   2.19620367e-04  -2.66267172e-02   3.35868329e-01   1.88787561e-02
   3.20029638e-04   8.37637636e-04  -3.60244161e-02   9.52993921e-03
  -1.37466670e-02  -2.32980918e-02  -1.16844136e-02   2.85002839e-02
   2.50363433e-02   2.26329903e-02   1.77682653e-03   2.42558745e-02
  -4.18486476e-01   1.43957091e-02  -6.32955299e-03   8.82665633e-04
   3.48657571e-02  -6.31398260e-02  -2.60510390e-02  -2.22465674e-02
   1.45861095e-02   5.94985583e-04  -3.17869459e-02   1.60837600e-02
  -3.01624485e-03  -3.26187983e-02   2.51844457e-02   1.59253448e-02
   4.41338669e-02   1.70479125e-03   3.35832133e-02  -6.32959658e-02
   2.55225935e-02  -7.42582481e-03  -4.30207566e-03  -3.26578754e-02
   7.18403748e-02   1.47938278e-02  -2.93272692e-04  -6.89103281e-02
   9.10389899e-01   3.49874503e-02   8.92904512e-02   6.21238762e-03
  -6.84603046e-02   1.71486834e-01   1.18905076e-01   1.02829449e-01
   5.09043019e-02  -7.34424199e-02   1.39186869e-01   3.74473860e-02
  -3.16898256e-04  -1.84767708e-02"""
out = a.split("  ")
w = [float(i.rstrip('\n')) for i in out]

result = run_result(b, w)
write_result(result)

