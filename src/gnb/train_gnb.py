# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 11:14:26 2022

@author: rodion kovalenko
"""
import sys
sys.path.append("..")

import numpy as np
from sklearn.multiclass import OneVsRestClassifier

from sklearn.naive_bayes import GaussianNB
import pandas as pd
import pickle
from embedding import sbert
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib as plt
from os import path
import pathlib

def divide_into_train_and_validation(df):
    dataset_shuffled = df.sample(frac=1, random_state=1).reset_index()
    X = get_numpy_dataset(dataset_shuffled['paragraph'])
    y = dataset_shuffled['annotation'].values
    
    return train_test_split(X, y, test_size=0.2, train_size=0.8)


def get_numpy_dataset(X):
    data = []
    for sentence_vec in X:
        data.append(np.array(sentence_vec))
        
    return np.array(data).astype('float32')

train_dataset = pd.read_json (r'../../data/sdg_train_dataset.json')
sgd_classes = pd.read_json (r'../../data/sdg_classes.json').values
sgd_classes = np.append(sgd_classes,'0')
sgd_classes = sgd_classes.reshape((-1, 1))

train_dataset_corpus = pd.read_json (r'../../data/sdg_train_dataset_corpus.json')
X_train, X_test, y_train, y_test = divide_into_train_and_validation(train_dataset_corpus)

X = get_numpy_dataset(train_dataset['paragraph'])
y = train_dataset['annotation'].values

for i, y_i in enumerate(y):
    y[i] = [y[i]]

X_train = np.concatenate((X, X_train))
y_train = np.concatenate((y, y_train))


current_abs_path = str(pathlib.Path().resolve())
saved_model_path = '\\..\\gnb\\gnb.pickle'
saved_model_dir = current_abs_path + '/' + saved_model_path

path_exists = path.exists(saved_model_dir)
   
if path_exists:
     print("saved model exists")    
     f = open(saved_model_dir, 'rb')
     gnb = pickle.load(f)
     f.close()
else:
    gnb = GaussianNB()
    print("saved model cannot not found")  


clf = OneVsRestClassifier(gnb)


y_train = MultiLabelBinarizer().fit_transform(y_train)
y_test_binary_matrix = MultiLabelBinarizer().fit_transform(y_test)
y_test_classes = np.array(np.unique(list(plt.cbook.flatten(y_test))))
clf.fit(X_train, y_train)

print('model has been trained')

#save the model
f = open('gnb.pickle', 'wb')
pickle.dump(clf, f)
f.close()
print('gnb model is saved')

#make predictions

sbert = sbert.Sbert()


for i, embedding in enumerate(X_test):    
    embedding = embedding.reshape((1, -1)).astype('float32')
    
    # predictions = clf.predict(embedding)
    
    # cosine_similarities = sbert.calculate_similarity(embedding, X, y, encode_sentence=False)    
    # best_five_classes = list(cosine_similarities.items())[-5:]    
    # print(best_five_classes)
    
    predicted_classes = sgd_classes.reshape((-1, 1))[predictions[0] == 1]
    print('predicted')
    print(predicted_classes)
    print('actual')
    print(y_test_classes[y_test_binary_matrix[i] == 1])
    print('----------------------------------------------')

