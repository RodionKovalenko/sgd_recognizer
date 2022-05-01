# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 12:19:29 2022

@author: rodion kovalenko
"""

import sys
sys.path.append("..")

import numpy as np
from nltk import tokenize
import pandas as pd
from sdg import sdg_target_list
from sdg import sdg_list
import matplotlib.pyplot as plt
import numpy as np


#embed training dataset into numeric vector-space

def generate_dataset(df, return_json = False):
    train_dataset = []
    
    for index, row in df.iterrows():
        goal_indices = row['annotation']
        target_list = []
        append_record = False
        
        for goal_index in goal_indices:
            target_list.append(goal_index[0])
            
            if goal_index[0] not in sdg_count_map.keys():
                sdg_count_map[goal_index[0]] = 0
            sdg_count_map[goal_index[0]] = sdg_count_map[goal_index[0]] + 1
            
        
            for target in goal_index[1]:
                 # if target != '0':
                 if target not in sdg_target_count_map.keys():
                    sdg_target_count_map[target] = 0 
                    
                 if target not in sdg_target_count_map_actual.keys():
                    sdg_target_count_map_actual[target] = 0                                     
              
                 sdg_target_count_map[target] = sdg_target_count_map[target]  + 1
                   
            
                 if sdg_target_count_map[target] < 1080:
                     sdg_target_count_map_actual[target] = sdg_target_count_map_actual[target]  + 1
                     append_record = True
              
                 
                 if target != '0' and target not in target_list:
                    target_list.append(target)
                    
                    # to flatten target
                #  if target == '0':
                #     list_item = {'paragraph': row['paragraph'], 'annotation': [goal_index[0]]}
                #     train_dataset.append(list_item)
                #  elif target not in target_list:
                #     list_item = {'paragraph': row['paragraph'], 'annotation': [target]}
                #     train_dataset.append(list_item)      
                # target_list.append(target)
                
            if len(target_list) > 0:
                list_item = {'paragraph': row['paragraph'], 'annotation': target_list}
                train_dataset.append(list_item)
                
            if append_record:
                cos_sim_training_dataset.append(list_item)
        
    if return_json:
        return pd.Series(train_dataset).to_json(orient='values')
    
    return train_dataset

def calculate_similarity_degree(target_1, target_2):
     match = 0.0
     n_common_classes_1 = len(np.where(np.isin(target_1, target_2) == True)[0])
     n_common_classes_2 = len(np.where(np.isin(target_2, target_1) == True)[0])
     n_common_classes = max(n_common_classes_1, n_common_classes_2)
     n_all = len(np.unique(target_1 + target_2).tolist())
     
     if len(target_1) == len(target_2) and (np.array(target_1) == np.array(target_2)).all():
         match = 1.0
     elif n_common_classes > 0:
        max_len = max(len(target_1), len(target_2))
        min_len = min(len(target_1), len(target_2))
        if n_common_classes == min_len:
            match = float(n_common_classes / max_len) + float(n_common_classes / n_all)
        else:
            match = float(n_common_classes / n_all)
        
     if match > 1:
        match = 1 / match
              
     return match

def create_cos_similarity_records(dataset):
    cos_similiarity_dataset = []
    for index_1, record_1 in enumerate(dataset):
        next_index = index_1 + 1
        if (len(dataset) - 1 >= next_index):
            for index_2, record_2 in enumerate(dataset[next_index:]):
                similarity_degree = calculate_similarity_degree(record_1['annotation'], record_2['annotation'])
                
                # if similarity_degree > 1:
                cos_similiarity_dataset.append([{'text_1': record_1['paragraph'],
                                                 'text_2': record_2['paragraph'],
                                                 'match' : similarity_degree}])
          
    
    return cos_similiarity_dataset

sgd_targets = np.array(sdg_target_list.targets)
sdg_goals = np.array(sdg_list.goals)
train_dataset = []
sdg_count_map = {}
sdg_target_count_map = {}
cos_sim_training_dataset = []
sdg_target_count_map_actual = {}

# sdg goals
for index, i in enumerate(sdg_goals):
    sentences = tokenize.sent_tokenize(i[1])
    list_item = {'paragraph': i[1], 'annotation': i[0]}
    train_dataset.append(list_item)    
    sdg_goal = str((index + 1))
       
    for sentence in sentences:
        list_item = {'paragraph': sentence, 'annotation': i[0]}
        list_item = {'paragraph': 'SDG ' + sdg_goal + ' ' + sentence, 'annotation': i[0]}
        list_item = {'paragraph': 'SDG-' + sdg_goal +  ' ' + sentence, 'annotation': i[0]}
        list_item = {'paragraph': 'sdg ' + sdg_goal +  ' ' + sentence, 'annotation': i[0]}
        list_item = {'paragraph': 'sdg-' + sdg_goal + ' ' +  sentence, 'annotation': i[0]}
        list_item = {'paragraph': 'sustainable development goal ' + sdg_goal + sentence, 'annotation': i[0]}
        train_dataset.append(list_item)
        
# sdg target
for goal_index, targets in enumerate(sgd_targets):
    goal = goal_index + 1
    for target in targets:
        list_item = {'paragraph': target[1], 'annotation': target[0]}
     
        train_dataset.append(list_item)

json_dataset = pd.Series(train_dataset).to_json(orient='values')

with open('../../data/sdg_train_dataset_raw.json', 'w') as outfile:
    outfile.write(json_dataset)
        
        
# main corpus of training and validation dataset
    

df_set_a = pd.read_json (r'../../data/dataset/set-A.json')
df_set_b = pd.read_json (r'../../data/dataset/set-B.json')
df_set_cb = pd.read_json (r'../../data/dataset/set-CB.json')
df_set_cl = pd.read_json (r'../../data/dataset/set-CL.json')
df_set_cs = pd.read_json (r'../../data/dataset/set-CS.json')


# make sure indexes pair with number of rows

set_a = generate_dataset(df_set_a.reset_index())
set_b = generate_dataset(df_set_b.reset_index())
set_cb = generate_dataset(df_set_cb.reset_index())
set_cl = generate_dataset(df_set_cl.reset_index()) 
set_cs = generate_dataset(df_set_cs.reset_index())

training_dataset = np.concatenate((set_a, set_b, set_cb, set_cl, set_cs))
training_dataset_json = pd.Series(training_dataset).to_json(orient='values')


cos_similarity_records = create_cos_similarity_records(cos_sim_training_dataset)

cos_sim_training_dataset_json = pd.Series(cos_similarity_records).to_json(orient='values')

with open('../../data/sdg_train_dataset_corpus_raw.json', 'w') as outfile:
    outfile.write(training_dataset_json)

with open('../../data/sdg_train_dataset_cos_sim_corpus_raw.json', 'w') as outfile:
    outfile.write(cos_sim_training_dataset_json)

def sort_dict(dictionary):
      sorted_dict_ = sorted(dictionary, key=dictionary.get) 
      sorted_dict = {}
        
      for w in sorted_dict_:
        sorted_dict[w] = dictionary[w]
        
      return sorted_dict
  
def convert_to_relative_values(dictionary):
     n_sum = sum(dictionary.values())
     normalized_dict = {}
        
     for w in dictionary:
        normalized_dict[w] = (dictionary[w] / n_sum)
        
     return normalized_dict
    
# sdg_count_map = sort_dict(sdg_count_map)
# sdg_target_count_map = sort_dict(sdg_target_count_map)
    

# sdg_count_map = convert_to_relative_values(sdg_count_map)
# sdg_target_count_map = convert_to_relative_values(sdg_target_count_map)
sdg_count_map = dict(sorted(sdg_count_map.items()))
sdg_target_count_map = dict(sorted(sdg_target_count_map.items()))
sdg_target_count_map_actual = dict(sorted(sdg_target_count_map_actual.items()))

goal_labels = []
target_labels = []
target_labels_actual = []

for i in sdg_count_map.keys():
    goal = str(int(i))
    goal_labels.append('SDG ' + goal)
    
for i in sdg_target_count_map.keys():
    target_labels.append(i)
    
for i in sdg_target_count_map_actual.keys():
    target_labels_actual.append(i)
    
plt.figure(figsize=(15, 10))
ax = plt.gca()
width= 0.5

# plt.bar(sdg_target_count_map.keys(), sdg_target_count_map.values(), width, color='g')
# plt.xticks(ticks=np.arange(0, len(target_labels), 1), labels=target_labels, rotation=90)
# ax.xaxis.label.set_size(150)


plt.bar(sdg_target_count_map_actual.keys(), sdg_target_count_map_actual.values(), width, color='g')
plt.xticks(ticks=np.arange(0, len(target_labels_actual), 1), labels=target_labels_actual, rotation=90)
ax.xaxis.label.set_size(150)

ax.set_ylim(0, 100)
# plt.savefig('sgd_goals_histrogram.png')


# plt.bar(sdg_count_map.keys(), sdg_count_map.values(), width, color='g')
# plt.xticks(ticks=np.arange(0, len(goal_labels), 1), labels=goal_labels, rotation=90)
# ax.xaxis.label.set_size(150)
# plt.savefig('sgd_histrogram.png')

plt.savefig('sgd_histrogram_test.png')
plt.show()

    
    
    