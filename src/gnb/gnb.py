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
import matplotlib.pyplot as plt
from os import path
import pathlib
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

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

def predict_classes(model, embedding, i = 0, y = [], verbose=False, threshold_probability = 0.99):
     embedding = embedding.reshape((1, -1)).astype('float32')
     predicted_classes_x = []
     expected_classes_y = []
     
     predicted_classes_x = model.predict(embedding)[0]
     probability_pred = model.predict_proba(embedding)
            
     indexes_of_predicted = np.where(model.classes_ == predicted_classes_x)
       
     if indexes_of_predicted:
         probability = probability_pred[0][indexes_of_predicted[0][0]]
         # print('probability {}'.format(probability_pred[0][indexes_of_predicted[0][0]]))
         if probability and probability < 0.9980:
             predicted_classes_x = 0
     # print('predicted class{}'.format(predicted_classes_x))
            
     return predicted_classes_x, expected_classes_y

def evaluate(model, X, y, threshold_probability = 0.99, stop_limit = 30, verbose=False):
        predicted_classes_x = []
                
        for i, embedding in enumerate(X):
            predicted_classes, expected_classes = predict_classes(model, embedding, i=i)
            
            
            predicted_classes_x.append(predicted_classes)
                
            if stop_limit and i > stop_limit:
                break
        
        y = y[0:len(predicted_classes_x)]
        confusion_matrices = confusion_matrix(y, predicted_classes_x)
        
        
        all_classes = y.tolist()
        all_classes.extend(predicted_classes_x)
      
        # print(confusion_matrices)
        class_report = classification_report(
                y,
                predicted_classes_x,
                output_dict=False,
                target_names=np.unique(all_classes),
                zero_division = 1
                )
        
        print(class_report)


current_abs_path = str(pathlib.Path(__file__).parent.resolve())
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


train_dataset = pd.read_json('https://cdn-lfs.huggingface.co/repos/09/f7/09f7b510f403b83a0640b04897c848818e68e6baef798fb907ece47e3e6ac300/c74052a03a328e1cf5a4fea9469873bf2567900ddc7990626ed661e276dd6aca')
train_dataset_corpus = pd.read_json('https://cdn-lfs.huggingface.co/repos/09/f7/09f7b510f403b83a0640b04897c848818e68e6baef798fb907ece47e3e6ac300/ab739d82fd3d442aaa36151d0f4ff5bd1d89a5a0fae951a38ba0cb95cf6c7953')

X = get_numpy_dataset(train_dataset['paragraph'])
y = train_dataset['annotation'].values

X_train, X_test, y_train, y_test = divide_into_train_and_validation(train_dataset_corpus)     
        
for i, target in enumerate(y_train):
    if len(target) > 1 and isinstance(target, str) == False:
        y_train[i] = target[1]
        # for label in target:
        #    y_train = np.append(y_train, label)
        #    X_train = np.vstack([X_train, X_train[i]])
    else:
        y_train[i] = target[0]

for i, target in enumerate(y_test):
    if len(target) > 1 and isinstance(target, str) == False:
        y_test[i] = target[1]
        # for label in target:
        #   y_test = np.append(y_test, label)
        #   X_test = np.vstack([X_test, X_test[i]])
    else:
        y_test[i] = target[0]
    

class_counts_train = {}
class_counts_test = {}
targets = []
X_train_balanced = []
y_train_balanced = []
X_test_balanced = []
y_test_balanced = []
for i, target in enumerate(y_train):
    if target not in targets:
        targets.append(target)
        
    if class_counts_train.get(target) is  None:
        # print('train target {}'.format(target))
        class_counts_train[target] = 0
            
    if class_counts_train[target] < 2 or target == '0':
        class_counts_train[target] = class_counts_train[target] + 1
        X_train_balanced.append(X_train[i])
        y_train_balanced.append(target)

for i, target in enumerate(y_test):
    if class_counts_test.get(target) is None:
        # print('test target {}'.format(target))
        class_counts_test[target] = 0
       
    if class_counts_test[target] < 20 or target == '0':
        class_counts_test[target] = class_counts_test[target] + 1
        X_test_balanced.append(X_test[i])
        y_test_balanced.append(target)

X_train = np.concatenate((X, X_train_balanced))
y_train = np.concatenate((y, y_train_balanced))

X_train_balanced = np.array(X_train_balanced)
y_train_balanced = np.array(y_train_balanced)
X_test_balanced = np.array(X_test_balanced)
y_test_balanced = np.array(y_test_balanced)

# import seaborn as sns
# sns.boxplot(x=X_train[0])

clf.fit(X_train, y_train)

print('model has been trained')

#save the model
f = open(saved_model_dir, 'wb')
pickle.dump(clf, f)
f.close()
print('gnb model is saved')

print('EVALUATION OF TEST SET')
score = clf.score(X_test_balanced, y_test_balanced)
print("score of Naive Bayes algo is :" , score)

print('EVALUATION OF TRAINING SET')
score = clf.score(X_train, y_train)
print("score of Naive Bayes algo is :" , score)

# print('EVALUATION OF TEST SET')
# evaluate(clf, X_test, y_test, threshold_probability=0.99, verbose=False, stop_limit=None)
# print('EVALUATION OF TRAINING SET')
# evaluate(clf, X_train, y_train, threshold_probability=0.99, verbose=False, stop_limit=None)


print('EVALUATION OF TEST SET')
evaluate(clf, X_test_balanced, y_test_balanced, threshold_probability=0.99, verbose=False, stop_limit=None)
print('EVALUATION OF TRAINING SET')
evaluate(clf, X_train, y_train, threshold_probability=0.99, verbose=False, stop_limit=None)

#make predictions
    
# for i, embedding in enumerate(X_train):    
#     embedding = embedding.reshape((1, -1)).astype('float32')
    
#     probabilities = clf.predict_proba(embedding)     
#     pred_class = clf.predict(embedding)    
#     print('predicted class{}'.format(pred_class))
    
#     cosine_similarities = sdg_clf.predict_by_gnb_and_cos_sim(embedding, encode=False, use_gnb=True, recog_level = 1)        
#     print('cosine similiary predicted {}'.format(cosine_similarities))
    
#     print('actual {}'.format(y_train[i]))
#     print('----------------------------------------------')
#     if i == 50:
#         break
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

