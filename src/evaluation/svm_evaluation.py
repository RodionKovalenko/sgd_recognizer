# -*- coding: utf-8 -*-
"""
Created on Sat May 21 15:47:10 2022

@author: rodion kovalenko
"""


import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def divide_into_train_and_validation(df):
    df = df.reset_index()
    dataset_shuffled = df.sample(frac=1, random_state=1).reset_index()
    X = dataset_shuffled['paragraph']
    y = dataset_shuffled['annotation'].values
    
    return train_test_split(X, y, test_size=923, train_size=2317)

def get_numpy_dataset(X):
    data = []
    for sentence_vec in X:
        data.append(np.array(sentence_vec))
        
    return np.array(data).astype('float32')


def evaluate(model, X, y, threshold_probability = 0.99, stop_limit = 30, verbose=False):
        predicted_classes_x = []
                
        for i, embedding in enumerate(X):
            predicted_classes = model.predict([embedding])
                        
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
        return class_report


train_dataset = pd.read_json('https://cdn-lfs.huggingface.co/repos/09/f7/09f7b510f403b83a0640b04897c848818e68e6baef798fb907ece47e3e6ac300/c74052a03a328e1cf5a4fea9469873bf2567900ddc7990626ed661e276dd6aca')
train_dataset_corpus = pd.read_json('https://cdn-lfs.huggingface.co/repos/09/f7/09f7b510f403b83a0640b04897c848818e68e6baef798fb907ece47e3e6ac300/ab739d82fd3d442aaa36151d0f4ff5bd1d89a5a0fae951a38ba0cb95cf6c7953')

X = get_numpy_dataset(train_dataset['paragraph'])
y = train_dataset['annotation'].values

X_train, X_test, y_train, y_test = divide_into_train_and_validation(train_dataset_corpus)     

X_train = list(X_train)
X_test = list(X_test)
        
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


y_train = np.reshape(y_train, (-1, 1))
y_test = np.reshape(y_test, (-1, 1))

# svm 
svm = LinearSVC(random_state=123)

# transforming into multilabel classifier
multilabel_classifier = MultiOutputClassifier(svm, n_jobs=-1)

# fitting the model
multilabel_classifier = multilabel_classifier.fit(X_train, y_train)

# Get predictions for test data
y_test_pred = multilabel_classifier.predict(X_test)

smv_score = multilabel_classifier.score(X_test, y_test)

print('svm F1-evaluation score is {}'.format(smv_score))

class_report_train = evaluate(multilabel_classifier, X_test, y_test, threshold_probability=0.99, verbose=False, stop_limit=None)

class_report_test = evaluate(multilabel_classifier, X_train, y_train, threshold_probability=0.99, verbose=False, stop_limit=None)



















