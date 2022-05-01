import numpy as np
import pandas as pd
from flask_restful import Resource
from nltk import tokenize

from src.embedding import sbert_finetuned, sbert_generic_pretrained
from src.sdg import sdg_target_list
from src.sdg import sdg_list

from src.gnb import one_vs_all_gnb

class Sgd_Classifier(Resource):
    def get_numpy_dataset(self, X):
        data = []
        for sentence_vec in X:
            data.append(np.array(sentence_vec).astype(np.float32))
            
        return np.array(data)

    def __init__(self, cos_sim_threshold = 0.6):
        self.sbert_generic = sbert_generic_pretrained.Sbert_generic_pretrained()
        self.sbert_finetuned = sbert_finetuned.Sbert()
        
        self.cos_sim_threshold = cos_sim_threshold
        train_embedded_dataset = pd.read_json('https://cdn-lfs.huggingface.co/repos/09/f7/09f7b510f403b83a0640b04897c848818e68e6baef798fb907ece47e3e6ac300/c74052a03a328e1cf5a4fea9469873bf2567900ddc7990626ed661e276dd6aca')
    
        
        self.sgd_embedded = self.get_numpy_dataset(train_embedded_dataset['paragraph'])
        self.sgd_targets = train_embedded_dataset['annotation']
        
        self.sdg_target_map = dict()
        sdg_target_goals = sdg_list.goals
        sdg_target_tuples = [target_tuple for array_item in np.array(sdg_target_list.targets) for target_tuple in array_item]
        
        for sdg_goal, sdg_text in sdg_target_goals:
            self.sdg_target_map.setdefault(sdg_goal, []).append(sdg_text)
            
        for sdg_target, sdg_text in sdg_target_tuples:
            self.sdg_target_map.setdefault(sdg_target, []).append(sdg_text)
            
        self.gnb = one_vs_all_gnb.Gnb()
            
    def get_concatencated_sbert_embedding(self, raw_text):
        embedding = np.add(self.sbert_finetuned.embed(raw_text), self.sbert_generic.embed(raw_text)) / 2.0
        return embedding
    
    def predict_by_gnb(self, raw_text, encode = True):
        embedding = raw_text
        
        if encode:
            embedding = self.get_concatencated_sbert_embedding(raw_text)
            
        predicted_classes, expected_classes = self.gnb.predict_classes(embedding.reshape((1, -1)))
        return predicted_classes

    def predict_by_cos_sim(self, raw_text, encode = True):
        
        embedding = raw_text
        
        if encode == True:
            embedding = self.get_concatencated_sbert_embedding(raw_text)
    
        cosine_similarities = self.sbert_finetuned.calculate_similarity(embedding, self.sgd_embedded, self.sgd_targets, encode_sentence = False)  
        return cosine_similarities
    
    def predict_by_gnb_and_cos_sim(self, raw_text, encode = True, threshold = None, recog_level = 1, use_gnb = True):
        sentences = [raw_text]
        gnb_pred_classes = []
        sentence_to_classes_map = {}       
        
        if threshold and float(threshold) <= 1.0:
            self.cos_sim_threshold = float(threshold)
        else:
            self.cos_sim_threshold = 0.6
        
        if encode and recog_level == 1:
            sentences = tokenize.sent_tokenize(raw_text)
            
        if  encode and raw_text and recog_level == 2:
            sentences = raw_text.split('\n')
            
        for sentence in sentences:
            cosine_similarity = self.predict_by_cos_sim(sentence, encode)
            
            if use_gnb:
                gnb_pred_classes = self.predict_by_gnb(sentence, encode)
                
            best_five_classes = list(cosine_similarity.items())[-3:]
            selected_classes = {}
           
            for index, value in enumerate(best_five_classes):
                if value[1] >= self.cos_sim_threshold:
                    if encode == False: 
                        sentence = str(value[0])
                
                    key = str(value[0])
                    value = self.sdg_target_map[str(value[0])][0] + ', similarity measure: ' + str(value[1])
                    
                    selected_classes[sentence] = {key: value}
            
          
              # if Gaussian Naive Bayes has no prediction or contains value '0' return 0
            if use_gnb and ('0' in gnb_pred_classes or 0 in gnb_pred_classes):
                selected_classes['naive bayes'] = {'result' : 0}
             
            if selected_classes:
                sentence_to_classes_map[sentence] = selected_classes
                
            
            print('predicted gnb {}'.format(gnb_pred_classes))
            if use_gnb and len(gnb_pred_classes) > 0 and any('0' in sublist for sublist in gnb_pred_classes) == False:                
                sdg_class_text = self.sdg_target_map[str(gnb_pred_classes[0][-1])][0]
                sdg_class = gnb_pred_classes[0][-1]

                print('sdg text {}'.format(sdg_class_text))
                print('sgd_class {}'.format(sdg_class))
                if sentence not in sentence_to_classes_map.keys():
                    sentence_to_classes_map[sentence] = {sdg_class: {sdg_class : sdg_class_text}}
        
        
        return sentence_to_classes_map
    
