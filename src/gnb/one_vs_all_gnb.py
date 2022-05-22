# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 11:14:26 2022

@author: rodion kovalenko
"""

import numpy as np
from sklearn.multiclass import OneVsRestClassifier

from sklearn.naive_bayes import GaussianNB
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from os import path
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import classification_report

from pathlib import Path

class Gnb():
    def __init__(self):        
        current_abs_path = str(Path(__file__).resolve().parent.parent.parent)
        self.saved_model_dir = current_abs_path + '/saved_model/gnb.pickle'    
           
        if path.exists(self.saved_model_dir):
             print("saved gnb model exists")
             f = open(self.saved_model_dir, 'rb')
             self.model = pickle.load(f)
             f.close()
        else:
            self.model = OneVsRestClassifier(GaussianNB())
            print("saved model gnb cannot not found")
            
        self.saved_label_binarizer_dir = current_abs_path + '/saved_model/multi_label_binarizer.pickle'
        
        if path.exists(self.saved_label_binarizer_dir):
             print("saved multilabel_binarizer model exists")
             f = open(self.saved_label_binarizer_dir, 'rb')
             self.multilabel_binarizer = pickle.load(f)
             f.close()
        else:
            print("saved multilabel_binarizer cannot not found")        
            self.multilabel_binarizer = MultiLabelBinarizer()
            
            train_dataset = pd.read_json('https://cdn-lfs.huggingface.co/repos/09/f7/09f7b510f403b83a0640b04897c848818e68e6baef798fb907ece47e3e6ac300/c74052a03a328e1cf5a4fea9469873bf2567900ddc7990626ed661e276dd6aca')          
            train_dataset_corpus = pd.read_json('https://cdn-lfs.huggingface.co/repos/09/f7/09f7b510f403b83a0640b04897c848818e68e6baef798fb907ece47e3e6ac300/ab739d82fd3d442aaa36151d0f4ff5bd1d89a5a0fae951a38ba0cb95cf6c7953')
            X_train, X_test, y_train, y_test = self.divide_into_train_and_validation(train_dataset_corpus)
            
            X = self.get_numpy_dataset(train_dataset['paragraph'])
            y = train_dataset['annotation'].values
            
            for i, y_i in enumerate(y):
                y[i] = [y[i]]                
            
            X_train = np.concatenate((X, X_train))
            y_train = np.concatenate((y, y_train))
            
            self.multilabel_binarizer.fit_transform(y_train)
            print('multilabel binarizer is fitted')
            f = open(self.saved_label_binarizer_dir, 'wb')
            pickle.dump(self.multilabel_binarizer, f)
            f.close()
            print('multibinarizer model is saved')

    def divide_into_train_and_validation(self, df):
        df = df.reset_index()
        dataset_shuffled = df.sample(frac=1, random_state=1).reset_index()
        X = self.get_numpy_dataset(dataset_shuffled['paragraph'])
        y = dataset_shuffled['annotation'].values
        
        return train_test_split(X, y, test_size=0.2, train_size=0.8)
    
    
    def get_numpy_dataset(self, X):
        data = []
        for sentence_vec in X:
            data.append(np.array(sentence_vec))
            
        return np.array(data).astype('float32')
    
    def fit(self, X_train, y_train, fit_again = True, save_model = True):
        if fit_again:
            print('model is being trained')
            self.model.fit(X_train, y_train)
            print('model has been trained')        

        #save the model
        if save_model: 
            f = open(self.saved_model_dir, 'wb')
            pickle.dump(self.model, f)
            f.close()
            print('gnb model is saved')
            
            f = open(self.saved_label_binarizer_dir, 'wb')
            pickle.dump(self.multilabel_binarizer, f)
            f.close()
            print('multibinarizer model is saved')
     
        
    def predict_classes(self, embedding, i = 0, y = [], verbose=False, threshold_probability = 0.99):
         embedding = embedding.reshape((1, -1)).astype('float32')
         predicted_classes_x = []
         expected_classes_y = []
            
         # index_with_one = np.where(self.model.predict(embedding)[0] == 1)
         # class_probabilities = self.model.predict_proba(embedding)
         # predicted_classes = self.multilabel_binarizer.inverse_transform(self.model.predict(embedding))
         
         # predicted_classes_selected = {}
            
         # for index, value in enumerate(index_with_one):     
         #    if len(value):
         #        label_index = value[index]
           
         #        if class_probabilities[0][label_index] >= threshold_probability:
         #          predicted_classes_selected[predicted_classes[0][index]] = [class_probabilities[0][label_index]]
                     
         # if len(predicted_classes_selected) != 0:
         #    predicted_classes_x.append(list(predicted_classes_selected.copy().keys()))

         # if len(y) != 0:
         #    if isinstance(y[i][0], list):
         #        expected_classes_y.append(y[i][0])
         #    else:
         #        expected_classes_y.append([y[i][0]])
            
         # if verbose and len(y) > 0:
         #    print('predicted classes {}'.format(predicted_classes_selected)) 
         #    print('actual {}'.format(y[i]))
         #    print('----------------------------------------------')
         predicted_classes_x = self.model.predict(embedding)
         probability_pred = self.model.predict_proba(embedding)               
                  
         indexes_of_predicted = np.where(self.model.classes_ == predicted_classes_x)
           
         if indexes_of_predicted:
             probability = probability_pred[0][indexes_of_predicted[0][0]]
             print('probability {}'.format(probability_pred[0][indexes_of_predicted[0][0]]))
             if probability and probability < 1.0:
                 predicted_classes_x = [[]]
         # predicted_classes_x = self.multilabel_binarizer.inverse_transform(predicted_classes_x)
         print('predicted class{}'.format(predicted_classes_x))
        
         return predicted_classes_x, expected_classes_y
                
    def evaluate(self, X, y, threshold_probability = 0.99, stop_limit = 30, verbose=False):
        # print('EVALUATION OF TEST SET')
        predicted_classes_x = []
        expected_classes_y = []
        for i, embedding in enumerate(X):
            predicted_classes, expected_classes = self.predict_classes(embedding, y=y, i=i)
            
            if predicted_classes:
                predicted_classes_x.append(predicted_classes)
                expected_classes_y.append(expected_classes)
                
            if stop_limit and i > stop_limit:
                break
        
        multilabel_binarizer_x = MultiLabelBinarizer()       
        
        all_classes = np.array(np.array(predicted_classes_x).reshape((-1, 1)).tolist() 
                               + np.array(expected_classes_y).reshape((-1,1)).tolist()).ravel().reshape((-1, 1))
        
        if isinstance(all_classes[0][0], str):
            all_classes = all_classes.ravel()
        
        multilabel_binarizer_x.fit(all_classes)
      
        # multilabel_binarizer_x.fit(predicted_classes_x + expected_classes_y)
        expected_classes_y = multilabel_binarizer_x.transform(np.array(expected_classes_y).ravel())
        predicted_classes_x = multilabel_binarizer_x.transform(np.array(predicted_classes_x).ravel())
        
        confusion_matrices = multilabel_confusion_matrix(expected_classes_y, predicted_classes_x)
      
        
        print(confusion_matrices)
        class_report = classification_report(
                expected_classes_y,
                predicted_classes_x,
                output_dict=False,
                target_names=multilabel_binarizer_x.classes_,
                zero_division = 1
                )
        
        print(class_report)
        
        # for class_index, confusion_matrix in enumerate(confusion_matrices):
        #     tp_and_fn = confusion_matrix[1][1] + confusion_matrix[1][0]
        #     tp_and_fp = confusion_matrix[1][1] + confusion_matrix[0][1]
        #     tp = confusion_matrix[1][1]
        #     tp_and_tn = confusion_matrix[0][0] + confusion_matrix[1][1]
        #     tp = confusion_matrix[1][1]
        #     tn = confusion_matrix[0][0]
        #     fp = confusion_matrix[0][1]
        #     fn = confusion_matrix[1][0]
            
        #     denominator = ((tp + fp)* (tp + fn) * (tn + fp) * (tn + fn))
            
        #     if denominator == 0:
        #         denominator = 1
        #     mcc = ((tp*tn) - (fp*fn)) / (math.sqrt(denominator))
            
        #     precision = 0
        #     recall = 1
        #     accuracy = tp_and_tn / np.sum(confusion_matrix)
        #     f1_score = 0
            
        #     if tp_and_fp > 0:
        #         precision = tp / tp_and_fp
        #     if tp_and_fn > 0:
        #         recall = tp / tp_and_fn
            
        #     if precision > 0 or recall > 0:
        #         f1_score = 2 * precision * recall / (precision + recall)
        
        #     print(confusion_matrix)
        #     print('class {}'.format(multilabel_binarizer_x.classes_[class_index]))
        #     print('accuracy: {}'.format(accuracy))
        #     print('precision: {}'.format(precision))
        #     print('recall: {}'.format(recall))
        #     print('f1 score: {}'.format(f1_score))
        #     print('Matthews correlation coefficient: {}'.format(mcc))
        #     print('__________________________________________')
        
        
        
        
        

# gnb = Gnb()

# current_abs_path = str(pathlib.Path().resolve())
# train_dataset = pd.read_json('https://cdn-lfs.huggingface.co/repos/09/f7/09f7b510f403b83a0640b04897c848818e68e6baef798fb907ece47e3e6ac300/c74052a03a328e1cf5a4fea9469873bf2567900ddc7990626ed661e276dd6aca')        
# train_dataset_corpus = pd.read_json('https://cdn-lfs.huggingface.co/repos/09/f7/09f7b510f403b83a0640b04897c848818e68e6baef798fb907ece47e3e6ac300/ab739d82fd3d442aaa36151d0f4ff5bd1d89a5a0fae951a38ba0cb95cf6c7953')


# X_train, X_test, y_train, y_test = gnb.divide_into_train_and_validation(train_dataset_corpus)        

# X = gnb.get_numpy_dataset(train_dataset['paragraph'])
# y = train_dataset['annotation'].values

# for i, y_i in enumerate(y):
#     y[i] = [y[i]]    
    
# for i, y_i in enumerate(y_test):
#     y_test[i] = y_test[i]

# X_train = np.concatenate((X, X_train))
# y_train = np.concatenate((y, y_train))


# y = gnb.multilabel_binarizer.fit_transform(y_train)
# y_test_binarized = gnb.multilabel_binarizer.fit_transform(y_test)

# gnb.fit(X_train, y, save_model = True, fit_again=True)
# gnb.fit(X_test, y_test_binarized, save_model = True, fit_again=True)


# print('EVALUATION OF TEST SET')
# gnb.evaluate(X_test, y_test, threshold_probability=0.99, verbose=False, stop_limit=None)
# print('EVALUATION OF TRAINING SET')
# gnb.evaluate(X_train, y_train, threshold_probability=0.99, verbose=False, stop_limit=None)


    
    
    
    
    
    
    
    
    
    
    
    

