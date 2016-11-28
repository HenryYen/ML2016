from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.cluster import KMeans
import sys
import numpy as np
import re


path = sys.argv[1]      #'./data'
fn_title = path + '/title_StackOverflow.txt'
fn_test = path + '/check_index.csv'
fn_doc = path + '/docs.txt'
fn_output = sys.argv[2]     #'out.csv'
nb_category = 30
nb_feature = 10000
nb_LSAcomponent = 20
use_idf = True
use_LSA = True


def load_data(fn):
    data = []
    with open(fn, 'r') as f:
        for line in f:
            data.append(line)
    return data
    

def load_test_data():
    data = []
    with open(fn_test, 'r') as f:
        f.readline()
        for line in f:   
            pair = line.split(',')       
            data.append([int(pair[1]), int(pair[2])])
    return data


def do_predict(label, data):
    index = 0
    result = []
    
    with open(fn_output, 'w'):
        with open(fn_output, 'a+') as f:
            f.write('ID,Ans\n')
            for e in data:
                if label[e[0]] == label[e[1]]:
                    val = 1
                else:
                    val = 0                
                result.append(val)
                f.write(str(index) + "," + str(val) + '\n')
                index += 1
    return result
    

def preprocess(data):
    for i in range(len(data)):
        parts = re.compile('\w+').findall(data[i])
        data[i]  = ' '.join([s.lower() for s in parts])
    return data


title_data = load_data(fn_title)
title_data = preprocess(title_data)
test_data = load_test_data()
#print "Title data:" ,len(title_data)
#print "Test data:" ,len(test_data)


vectorizer = TfidfVectorizer(max_df=0.5, max_features=nb_feature, min_df=2, stop_words='english', use_idf=use_idf)
X = vectorizer.fit_transform(title_data)
#print "(n_samples, n_features)", X.shape


if use_LSA:
    svd = TruncatedSVD(nb_LSAcomponent)       
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    X = lsa.fit_transform(X)


km = KMeans(n_clusters=nb_category, init='k-means++', max_iter=100, n_init=1)
label = km.fit_predict(X)
result = do_predict(label, test_data)



        
                
        
        
