# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 16:40:22 2021

@author: zhou1281
"""


import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import sys

plt.rcParams['figure.figsize'] = [60, 36]
from adjustText import adjust_text
from transformers import BertTokenizer, BertModel, BertForMaskedLM,BertConfig,BertForPreTraining,load_tf_weights_in_bert

import logging
logging.basicConfig(level=logging.INFO)
# Load BERT.
#model = BertModel.from_pretrained('bert-large-uncased-whole-word-masking')
# Set the model to eval mode.

# This notebook assumes CPU execution. If you want to use GPUs, put the model on cuda and modify subsequent code blocks.
#model.to('cuda')
# Load tokenizer.


#load cancer bert customized vocab
tokenizer_cancer_bert = BertTokenizer.from_pretrained('./cancerbert_iter2/')
config_cancer_bert  = BertConfig.from_json_file('cancerbert_iter2/bert_config.json')
config_cancer_bert.output_hidden_states =True
model_cancer_bert = BertForPreTraining(config_cancer_bert)
model_cancer_bert = load_tf_weights_in_bert(model = model_cancer_bert,config = config_cancer_bert,
                                tf_checkpoint_path="cancerbert_iter2/bert_model.ckpt")

model_cancer_bert.eval()

# Get BERT's vocabulary embeddings.
wordembs = model_cancer_bert.get_input_embeddings()

print(model_cancer_bert.config.vocab_size)

# Convert the vocabulary embeddings to numpy.
allinds = np.arange(0,model_cancer_bert.config.vocab_size,1)
inputinds = torch.LongTensor(allinds)
bertwordembs = wordembs(inputinds).detach().numpy()
print(type(bertwordembs))
print(bertwordembs.shape)

def loadLines(filename):
    print("Loading lines from file", filename)
    f = open(filename,'r',encoding = "utf-8")
    lines = np.array([])
    for line in f:
        lines = np.append(lines, line.rstrip())
    print("Done. ", len(lines)," lines loaded!")
    return lines

bertwords = loadLines('./cancerbert_iter2/vocab.txt')

#Determine vocabulary to use for t-SNE/visualization. The indices are hard-coded based partially on inspection:
bert_char_indices_to_use = np.arange(0, 400, 1)
print(bert_char_indices_to_use.shape)
bert_char_indices_to_use2 = np.arange(1000, 1500, 1)
bert_char_indices_to_use= np.append(bert_char_indices_to_use, bert_char_indices_to_use2)
print(bert_char_indices_to_use.shape)
bert_voc_indices_to_plot = bert_char_indices_to_use
bert_voc_indices_to_use = bert_char_indices_to_use

print(len(bert_voc_indices_to_plot))
print(len(bert_voc_indices_to_use))

# print(bertwords[bert_voc_indices_to_use])

bert_voc_indices_to_use_tensor = torch.LongTensor(bert_voc_indices_to_use)
bert_word_embs_to_use = wordembs(bert_voc_indices_to_use_tensor).detach().numpy()


# Run t-SNE on the BERT vocabulary embeddings we selected:
mytsne_words = TSNE(n_components=2,early_exaggeration=12,verbose=2,metric='cosine',init='pca',n_iter=3500)
bert_word_embs_to_use_tsne = mytsne_words.fit_transform(bert_word_embs_to_use)

bert_words_to_plot = bertwords[bert_voc_indices_to_plot]
print(len(bert_words_to_plot))
# Plot the transformed BERT vocabulary embeddings:
fig = plt.figure() 
alltexts = list()
for i, txt in enumerate(bert_words_to_plot):
    plt.scatter(bert_word_embs_to_use_tsne[i,0], bert_word_embs_to_use_tsne[i,1], s=0)
    currtext = plt.text(bert_word_embs_to_use_tsne[i,0], bert_word_embs_to_use_tsne[i,1], txt, family='sans-serif')
    alltexts.append(currtext)
    

# Save the plot before adjusting.
plt.savefig('viz-bert-900_cancervoc.pdf', format='pdf')
print('now running adjust_text')
# Using autoalign often works better in my experience, but it can be very slow for this case, so it's false by default below:
#numiters = adjust_text(alltexts, autoalign=True, lim=50)
numiters = adjust_text(alltexts, autoalign=True, lim=50)
print('done adjust text, num iterations: ', numiters)
plt.savefig('viz-bert-900_cancervoc-adj50.pdf', format='pdf')

plt.show