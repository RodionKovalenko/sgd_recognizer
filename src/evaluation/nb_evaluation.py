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

from sklearn.naive_bayes import BernoulliNB, MultinomialNB
import pandas as pd
import pickle
from webservice import sdg_classifier
from sklearn.model_selection import train_test_split
from os import path
import pathlib
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer

sdg_clf = sdg_classifier.Sgd_Classifier()


def divide_into_train_and_validation(df):
    df = df.reset_index()
    dataset_shuffled = df.sample(frac=1, random_state=1).reset_index()
    X = dataset_shuffled['paragraph']
    y = dataset_shuffled['annotation'].values
    
    return train_test_split(X, y, test_size=0.2, train_size=0.8)


def evaluate(model, X, y, threshold_probability = 0.99, stop_limit = 30, verbose=False):
        predicted_classes_x = []
                
        for i, embedding in enumerate(X):
            predicted_classes = model.predict(count_vectorizer.transform([embedding]))
                        
            predicted_classes_x.append(predicted_classes[0])
                
            if stop_limit and i > stop_limit:
                break
        
        y = y[0:len(predicted_classes_x)]
        confusion_matrices = confusion_matrix(y, predicted_classes_x)
        
        
        all_classes = y.tolist()
        all_classes.extend(predicted_classes_x)
      
        print(confusion_matrices)
        class_report = classification_report(
                y,
                predicted_classes_x,
                output_dict=False,
                target_names=np.unique(all_classes),
                zero_division = 1
                )
        
        print(class_report)


current_abs_path = str(pathlib.Path(__file__).parent.resolve())
saved_model_dir = current_abs_path + '\\..\\..\\saved_model\\nb.pickle'

path_exists = path.exists(saved_model_dir)
   
if path_exists:
     print("saved gnb model exists")
     f = open(saved_model_dir, 'rb')
     gnb = pickle.load(f)
     f.close()
else:
    gnb = MultinomialNB()
    print("saved model cannot not found")  

clf = gnb


train_dataset = pd.read_json('https://cdn-lfs.huggingface.co/repos/09/f7/09f7b510f403b83a0640b04897c848818e68e6baef798fb907ece47e3e6ac300/7c3d6be1ae048ce11d312c6e99c3450f490c6ec5f8836c8742e743abeb46c33e')
train_dataset_corpus_raw = pd.read_json('https://cdn-lfs.huggingface.co/repos/09/f7/09f7b510f403b83a0640b04897c848818e68e6baef798fb907ece47e3e6ac300/ad856b4889d0a82cfd06866bdd81e4c82a006a733115ff79d5f2293f1e58e47c')

X_train, X_test, y_train, y_test = divide_into_train_and_validation(train_dataset_corpus_raw)     

X = train_dataset['paragraph']
y = train_dataset['annotation'].values

X_train = np.concatenate((X, X_train))
y_train = np.concatenate((y, y_train))

all_dataset = np.concatenate((X_train, X_test))
count_vectorizer = CountVectorizer()
count_vectorizer.fit(all_dataset)

X_train_one_hot = count_vectorizer.transform(X_train)
X_test_one_hot = count_vectorizer.transform(X_test)
      
for i, target in enumerate(y_train):
    if len(target) > 1 and isinstance(target, str) == False:
        y_train[i] = target[1]
        # for label in target:
        #     y_train = np.append(y_train, label)
        #     X_train_one_hot = np.vstack([X_train_one_hot, count_vectorizer.transform([X_train[i]])])
    else:
        y_train[i] = target[0]

for i, target in enumerate(y_test):
    if len(target) > 1 and isinstance(target, str) == False:
        y_test[i] = target[1]
        # for label in target:
        #   y_test = np.append(y_test, label)
        #   X_test_one_hot = np.vstack([X_test_one_hot, count_vectorizer.transform([X_test[i]])])
    else:
        y_test[i] = target[0]
    

clf = MultinomialNB()
clf.fit(X_train_one_hot, y_train)

print('model has been trained')
#save the model
f = open(saved_model_dir, 'wb')
pickle.dump(clf, f)
f.close()
print('gnb model is saved')

score = clf.score(X_test_one_hot, y_test)
print("score of Naive Bayes algo is :" , score)

score = clf.score(X_train_one_hot, y_train)
print("score of Naive Bayes algo is :" , score)

print('EVALUATION OF TEST SET')
evaluate(clf, X_test, y_test, threshold_probability=0.99, verbose=False, stop_limit=None)
print('EVALUATION OF TRAINING SET')
evaluate(clf, X_train, y_train, threshold_probability=0.99, verbose=False, stop_limit=None)


#make predictions
    
# for i, embedding in enumerate(X_train_one_hot):
    
#     probabilities = clf.predict_proba(embedding)     
#     pred_class = clf.predict(embedding)    
#     print('predicted class{}'.format(pred_class))
    
#     print('actual {}'.format(y_train[i]))
#     print('----------------------------------------------')
#     if i == 50:
#         break
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



