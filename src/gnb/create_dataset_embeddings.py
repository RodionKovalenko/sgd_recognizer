# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 12:19:29 2022

@author: rodion kovalenko
"""

import sys
sys.path.append("..")
import os
os.chdir('..\\..\\src')

import numpy as np
from nltk import tokenize
import pandas as pd
import io, json
import pathlib

from sdg import sdg_target_list
from sdg import sdg_list
from embedding import sbert_finetuned, sbert_generic_pretrained

sbert_finetuned = sbert_finetuned.Sbert()
sbert_generic = sbert_generic_pretrained.Sbert_generic_pretrained()

def get_concatencated_sbert_embedding(raw_text):
     embedding = (sbert_finetuned.embed(raw_text) + sbert_generic.embed(raw_text))/2.0
     return embedding
     

def generate_embedded_dataset(df, return_json = False):
    train_dataset = []
    
    for index, row in df.iterrows():
        goal_indices = row['annotation']
        target_list = []
        
        for goal_index in goal_indices:
            target_list.append(goal_index[0])
            for target in goal_index[1]:
                if target != '0' and target not in target_list:
                    target_list.append(target)
                
        if len(target_list) > 0:
            list_item = {'paragraph': get_concatencated_sbert_embedding(row['paragraph']), 'annotation': target_list}
            train_dataset.append(list_item)
        
    if return_json:
        return pd.Series(train_dataset).to_json(orient='values')
    
    return train_dataset

sgd_targets = np.array(sdg_target_list.targets)
sdg_goals = np.array(sdg_list.goals)
train_dataset = []
classes = []


# sdg goals
for i in sdg_goals:
    sentences = tokenize.sent_tokenize(i[1])
    
    embedding = get_concatencated_sbert_embedding(i[1])
    list_item = {'paragraph': embedding, 'annotation': i[0]}
    train_dataset.append(list_item)
    
    if i[0] not in classes:
        classes.append(i[0])
        
    for sentence in sentences:
        list_item = {'paragraph': get_concatencated_sbert_embedding(sentence), 'annotation': i[0]}
        train_dataset.append(list_item)
        
# sdg target
for goal_index, targets in enumerate(sgd_targets):
    goal = goal_index + 1
    for target in targets:
        list_item = {'paragraph': get_concatencated_sbert_embedding(target[1]), 'annotation': target[0]}
        
        if [goal, [target[0]]] not in classes:
            classes.append(target[0])
            
        train_dataset.append(list_item)

json_classes = pd.Series(classes).to_json(orient='values')
json_dataset = pd.Series(train_dataset).to_json(orient='values')

current_abs_path = str(pathlib.Path(__file__).parent.resolve())
with open(current_abs_path + '\\..\\..\\data\\sdg_sbert_embedded.json', 'w') as outfile:
    outfile.write(json_dataset)
    
with open(current_abs_path + '\\..\\..\\data\\sdg_classes.json', 'w') as outfile:
    outfile.write(json_classes)
    
        
# main corpus of training and validation dataset
    
df_set_a = pd.read_json(current_abs_path + '\\..\\..\\data\\dataset\\set-A.json')
df_set_b = pd.read_json(current_abs_path + '\\..\\..\\data\\dataset\\set-B.json')
df_set_cb = pd.read_json(current_abs_path + '\\..\\..\\data\\dataset\\set-CB.json')
df_set_cl = pd.read_json(current_abs_path + '\\..\\..\\data\\dataset\\set-CL.json')
df_set_cs = pd.read_json(current_abs_path + '\\..\\..\\data\\dataset\\set-CS.json')


# make sure indexes pair with number of rows

set_a = generate_embedded_dataset(df_set_a.reset_index())
set_b = generate_embedded_dataset(df_set_b.reset_index())
set_cb = generate_embedded_dataset(df_set_cb.reset_index())
set_cl = generate_embedded_dataset(df_set_cl.reset_index()) 
set_cs = generate_embedded_dataset(df_set_cs.reset_index())

training_dataset = np.concatenate((set_a, set_b, set_cb, set_cl, set_cs))
training_dataset_json = pd.Series(training_dataset).to_json(orient='values')

with open(current_abs_path + '\\..\\..\\data\\sdg_train_dataset_embedded_corpus.json', 'w') as outfile:
    outfile.write(training_dataset_json)
    
    
    
    
    
    
    
    
    
    
    
    