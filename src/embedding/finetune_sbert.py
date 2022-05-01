# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 23:15:48 2022

@author: rodion kovalenko
"""

from sentence_transformers import util,SentenceTransformer, models
from sentence_transformers import evaluation
from sentence_transformers import InputExample, losses
from torch.utils.data import DataLoader
import pandas as pd
from os import path
import pathlib

df = pd.read_json('https://cdn-lfs.huggingface.co/repos/09/f7/09f7b510f403b83a0640b04897c848818e68e6baef798fb907ece47e3e6ac300/7e55a4a7d2af7fe4edfbc4f7cd2f84256b352fbe96c2e6370efbb8ce553a4f2b')

sample = df.sample(n=20000, random_state=1)
#70 percent of all data for training set
sample_training = sample.sample(frac=0.7, random_state=1)
#30 percent of all data for evaluation data set
sample_evaluation = sample.drop(sample_training.index)

#Define the model. Either from scratch of by loading a pre-trained model
# word_embedding_model = SentenceTransformer('data/sbert_trained_model/')

current_abs_path = str(pathlib.Path(__file__).parent.resolve())

saved_model_path = '..\\..\\saved_model\\sbert_trained_sdg_sim_score'
#bert model to create sentence representations
word_embedding_model = models.Transformer('sentence-transformers/all-mpnet-base-v2')

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

#sbert model
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# retrieve the existing model or pull the defined if not exists

saved_model_dir = current_abs_path + '\\' + saved_model_path


print(saved_model_dir)

if path.exists(saved_model_dir):
    print('model was already saved')
    model = SentenceTransformer(saved_model_dir)
else:
    print('model was not saved')

#crete training and validation sets
train_num_labels = len(sample_training)

sample_training.reset_index()
print(sample_training)


def generate_input_target_dataset(dataset):
    dataset = dataset.reset_index()
    labels_evaluation = []
    evaluation_examples = []
    evaluation_text_1 = []
    evaluation_text_2 = []
    
    for sample_index, row in dataset.iterrows():
        match = row[0]['match']
        text_1 = row[0]['text_1']
        text_2 = row[0]['text_2']
            
        if  not text_1.isspace() and text_2 and not text_2.isspace():           
            evaluation_examples.append(InputExample(texts=[text_1, text_2], label=float(match)))
            
            evaluation_text_1.append(text_1)
            evaluation_text_2.append(text_2)
            labels_evaluation.append(float(match))    
    return evaluation_examples, labels_evaluation, evaluation_text_1, evaluation_text_2


train_examples, label, eval_1, eval_2 = generate_input_target_dataset(sample_training)
evaluation_examples, labels_evaluation, eval_text_1, eval_text_2 = generate_input_target_dataset(sample_evaluation)


#Define your train dataset, the dataloader and the train loss
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)
evaluator = evaluation.EmbeddingSimilarityEvaluator(eval_text_1, eval_text_2, labels_evaluation)

num_epochs = 1
warmup_steps = 2

#Tune the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=num_epochs,
          warmup_steps=warmup_steps,
          evaluator=evaluator,
          evaluation_steps=2,         
          output_path=saved_model_path)


print("EVALUATION FIRST 10 training records")
#evaluate model first on trained data set
sample_training = sample_training.reset_index()
sample_evaluation = sample_evaluation.reset_index()

for sample_index, row in sample_training[0:10].iterrows():
    embeddings = model.encode([eval_1[sample_index], eval_2[sample_index]])

    cosine_scores = util.pytorch_cos_sim(embeddings[0], embeddings[1])
    print(cosine_scores)
    print(label[sample_index])
    print()

print("EVALUATION FIRST 10 evalation records")
#evaluate model first on evaluation data set
for sample_index, row in sample_evaluation[0:10].iterrows():
    embeddings = model.encode([eval_text_1[sample_index], eval_text_2[sample_index]])
    
    cosine_scores = util.pytorch_cos_sim(embeddings[0], embeddings[1])
    print(cosine_scores)
    print(labels_evaluation[sample_index])
    print()