# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 11:14:26 2022

@author: rodion kovalenko
"""
import sys
import os
sys.path.append("..")
os.chdir('..\\..\\src')

import numpy as np
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

from sklearn.naive_bayes import GaussianNB
import pandas as pd
import pickle
from webservice import sdg_classifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib as plt
from os import path
import pathlib

sdg_clf = sdg_classifier.Sgd_Classifier()


def divide_into_train_and_validation(df):
    df = df.reset_index()
    dataset_shuffled = df.sample(frac=1, random_state=1).reset_index()
    X = get_numpy_dataset(dataset_shuffled['paragraph'])
    y = dataset_shuffled['annotation'].values
    
    return train_test_split(X, y, test_size=0.2, train_size=0.8)


def get_numpy_dataset(X):
    data = []
    for sentence_vec in X:
        data.append(np.array(sentence_vec))
        
    return np.array(data).astype('float32')

current_abs_path = str(pathlib.Path(__file__).parent.resolve())
train_dataset = pd.read_json (current_abs_path + '\\..\\..\\data\\sdg_sbert_embedded.json')

sgd_classes = pd.read_json(current_abs_path + '\\..\\..\\data\\sdg_classes.json').values
sgd_classes = sgd_classes.reshape((-1, 1))

X = get_numpy_dataset(train_dataset['paragraph'])
y = train_dataset['annotation'].values

# for i, y_i in enumerate(y):
#     y[i] = [y[i]]        


# X_train = np.concatenate((X, X_train))
# y_train = np.concatenate((y, y_train))

X_train = X
y_train = y.ravel()

saved_model_dir = current_abs_path + '\\..\\..\\saved_model\\gnb.pickle'

path_exists = path.exists(saved_model_dir)
   
if path_exists:
     print("saved gnb model exists")
     f = open(saved_model_dir, 'rb')
     gnb = pickle.load(f)
     f.close()
else:
    gnb = GaussianNB()
    print("saved model cannot not found")  

clf = gnb


clf.fit(X, y)

print('model has been trained')

#save the model
f = open(saved_model_dir, 'wb')
pickle.dump(clf, f)
f.close()
print('gnb model is saved')

#make predictions
    
for i, embedding in enumerate(X_train):    
    embedding = embedding.reshape((1, -1)).astype('float32')
    
    probabilities = clf.predict_proba(embedding)     
    pred_class = clf.predict(embedding)    
    print('predicted class{}'.format(pred_class))
    
    cosine_similarities = sdg_clf.predict_by_gnb_and_cos_sim(embedding, encode=False, use_gnb=False)        
    print('cosine similiary predicted {}'.format(cosine_similarities))
    
    print('actual {}'.format(y[i]))
    print('----------------------------------------------')
    if i == 50:
        break
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

