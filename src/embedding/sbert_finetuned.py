# -*- coding: utf-8 -*-
"""
@author: rodion kovalenko
"""

from flask_restful import Resource
from sentence_transformers import SentenceTransformer, util
import numpy as np

class Sbert(Resource):
    def __init__(self):
        self.model = SentenceTransformer('Rodion/sbert_uno_sustainable_development_goals')
        print('finetuned sbert is downloaded')
      

    def calculate_match(self, sentences):
        # https://www.sbert.net/

        # embeddings = model.encode(sentences, convert_to_tensor=True)
        embeddings = self.model.encode(sentences, convert_to_tensor=True)

        return self.calculate_similarity_from_emb(embeddings[0], embeddings[1])
    
    def calculate_similarity_from_emb(self, sentence1_emb, sentence_2_emb):    
        # Compute cosine-similarities for each sentence with each other sentence
        cosine_scores = util.pytorch_cos_sim(sentence1_emb, sentence_2_emb)

        pairs = []
        for i in range(len(cosine_scores) - 1):
            for j in range(i+1, len(cosine_scores)):
                pairs.append({'index': [i, j], 'score': cosine_scores[i][j]})

        # # Sort scores in decreasing order
        pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)
        matching_score = {}    

        for i in range(len(cosine_scores)):
            matching_score = float("{:.4f}".format(cosine_scores[i][i]))

        return matching_score
    
    def embed(self, sentence):
        return np.array(self.model.encode(sentence, normalize_embeddings=False))
    
    def embed_array(self, sentences_array):
        embeddings = []
        
        for sentence in sentences_array:
            embeddings.append(self.embed(sentence))
        
        return np.array(embeddings)
    
    def calculate_similarity(self, sentence, target_sentences_emb, sgd_targets, encode_sentence = True):
        sim_scores = {}
        sentence_emb = sentence
        
        if encode_sentence:
            sentence_emb = self.model.encode(sentence)
        
        sentence_emb = sentence_emb.ravel()
        for i, target_sentence in enumerate(target_sentences_emb):
            # score-target map
            target_sentence = target_sentence.ravel()
            
            if type(sgd_targets[0]) is list:
                sim_scores[sgd_targets[i][0]] = self.calculate_similarity_from_emb(sentence_emb, target_sentence)
            else:
                sim_scores[sgd_targets[i]] = self.calculate_similarity_from_emb(sentence_emb, target_sentence)
        
        sorted_similarities_keys = sorted(sim_scores, key=sim_scores.get)  

        sorted_dict = {}
        
        for w in sorted_similarities_keys:
            sorted_dict[w] = sim_scores[w]
            
        return sorted_dict          
    





