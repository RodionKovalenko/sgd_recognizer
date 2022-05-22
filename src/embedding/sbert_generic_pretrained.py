# -*- coding: utf-8 -*-
"""
@author: rodion kovalenko
"""

from flask_restful import Resource
from sentence_transformers import SentenceTransformer
import numpy as np
from os import path
from pathlib import Path

class Sbert_generic_pretrained(Resource):
    def __init__(self):
        current_abs_path = str(Path(__file__).resolve().parent.parent.parent)
        self.saved_model_dir = current_abs_path + '/saved_model/sbert_generic_pretrained'
                
        path_exists = path.exists(self.saved_model_dir)
           
        if path_exists:
            print("saved sbert generic model exists")    
            self.model = SentenceTransformer(self.saved_model_dir)
        else:
            print('sbert model generic is downloaded')
            self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

        
    def embed(self, sentence):
        return np.array(self.model.encode(sentence, normalize_embeddings=False))
    
    def embed_array(self, sentences_array):
        embeddings = []
        
        for sentence in sentences_array:
            embeddings.append(self.embed(sentence))
        
        return np.array(embeddings)
    
    





