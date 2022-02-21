# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 12:08:50 2021

@author: zhou1281
"""

from transformers import BertTokenizer, BertModel,BertConfig,BertForPreTraining,load_tf_weights_in_bert
import pandas as pd
import numpy as np
import nltk
import torch
from pytorch_pretrained_bert import BertTokenizer
import os

#current_directory = os.path.dirname(__file__)
#print(current_directory)
#load cancer bert customized vocab
tokenizer_cancer_bert = BertTokenizer.from_pretrained('./cancerbert_iter2/')
config_cancer_bert  = BertConfig.from_json_file('cancerbert_iter2/bert_config.json')
config_cancer_bert.output_hidden_states =True
model_cancer_bert = BertForPreTraining(config_cancer_bert)
model_cancer_bert = load_tf_weights_in_bert(model = model_cancer_bert,config = config_cancer_bert,
                                tf_checkpoint_path="cancerbert_iter2/bert_model.ckpt")

#load blue bert
tokenizer_blue_bert = BertTokenizer.from_pretrained('./blue_bert/')
config_blue_bert  = BertConfig.from_json_file('blue_bert/bert_config.json')
config_blue_bert.output_hidden_states =True
model_blue_bert = BertForPreTraining(config_blue_bert)
model_blue_bert = load_tf_weights_in_bert(model = model_blue_bert,config = config_blue_bert,
                                tf_checkpoint_path="blue_bert/model.ckpt-1000000")

#load cancer bert 900 vocab
tokenizer_c900_bert = BertTokenizer.from_pretrained('./997cust_vocab_50wsteps/')
config_c900_bert  = BertConfig.from_json_file('997cust_vocab_50wsteps/bert_config.json')
config_c900_bert.output_hidden_states =True
model_c900_bert = BertForPreTraining(config_c900_bert)
model_c900_bert = load_tf_weights_in_bert(model = model_c900_bert,config = config_c900_bert,
                                tf_checkpoint_path="997cust_vocab_50wsteps/bert_model.ckpt")



#load cancer bert origin vocab
tokenizer_bert_ori = BertTokenizer.from_pretrained('./bert_origin/')
config_bert_ori  = BertConfig.from_json_file('bert_origin/bert_config.json')
config_bert_ori.output_hidden_states =True
model_bert_ori = BertForPreTraining(config_bert_ori)
model_bert_ori = load_tf_weights_in_bert(model = model_bert_ori,config = config_bert_ori,
                                tf_checkpoint_path="bert_origin/bert_model.ckpt")

def bert_text_preparation(text, tokenizer):
    """Preparing the input for BERT
    
    Takes a string argument and performs
    pre-processing like adding special tokens,
    tokenization, tokens to ids, and tokens to
    segment ids. All tokens are mapped to seg-
    ment id = 1.
    
    Args:
        text (str): Text to be converted
        tokenizer (obj): Tokenizer object
            to convert text into BERT-re-
            adable tokens and ids
        
    Returns:
        list: List of BERT-readable tokens
        obj: Torch tensor with token ids
        obj: Torch tensor segment ids
    
    
    """
    marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1]*len(indexed_tokens)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    return tokenized_text, tokens_tensor, segments_tensors


def get_bert_embeddings(tokens_tensor, segments_tensors, model):
    """Get embeddings from an embedding model
    
    Args:
        tokens_tensor (obj): Torch tensor size [n_tokens]
            with token ids for each token in text
        segments_tensors (obj): Torch tensor size [n_tokens]
            with segment ids for each token in text
        model (obj): Embedding model to generate embeddings
            from token and segment ids
    
    Returns:
        list: List of list of floats of size
            [n_tokens, n_embedding_dimensions]
            containing embeddings for each token
    
    """
    
    # Gradient calculation id disabled
    # Model is in inference mode
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        # Removing the first hidden state
        # The first state is the input state
        hidden_states = outputs[2][1:]

    # Getting embeddings from the final BERT layer
    token_embeddings = hidden_states[-1]
    # Collapsing the tensor into 1-dimension
    token_embeddings = torch.squeeze(token_embeddings, dim=0)
    # Converting torchtensors to lists
    list_token_embeddings = [token_embed.tolist() for token_embed in token_embeddings]

    return list_token_embeddings


def target_word_embeddings(model, tokenizer,texts):
    target_word_embeddings = []
    for text in texts:
        tokenized_text, tokens_tensor, segments_tensors = bert_text_preparation(text, tokenizer)
        list_token_embeddings = get_bert_embeddings(tokens_tensor, segments_tensors, model)
        tokenzied_word = tokenizer.tokenize(text)
        word_index = []
        for i in tokenzied_word:
            word_index.append(tokenized_text.index(i))      
        word_embedding = []
        for i in word_index:
            word_embedding.append(list_token_embeddings[i])
        nums = len(word_embedding)
        word_embedding = [sum(i) for i in zip(*word_embedding)]
        word_embedding = [i/nums for i in word_embedding]
        target_word_embeddings.append(word_embedding)
    return target_word_embeddings

# with open('all_labeled_phrases.txt','r') as f:
#     lines= f.readlines()
#     lines = [i.strip() for i in lines]
    
with open('all_labeled_tokens.txt','r') as f:
    lines= f.readlines()
    lines = [i.strip() for i in lines]    
#labeled_tokens  = target_word_embeddings(model_cancer_bert, tokenizer_cancer_bert,lines)

#labeled_tokens  = target_word_embeddings(model_c900_bert, tokenizer_c900_bert,lines)
labeled_tokens  = target_word_embeddings(model_bert_ori, tokenizer_bert_ori,lines)
#labeled_tokens  = target_word_embeddings(model_cancer_bert, tokenizer_cancer_bert,lines)




# # Get BERT's vocabulary embeddings.
# wordembs = model_cancer_bert.get_input_embeddings()
# print(wordembs.shape())

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import sys


from adjustText import adjust_text


# # print(bert_char_indices_to_use.shape)
# # bert_char_indices_to_use2 = np.arange(1000, 1500, 1)
# # bert_char_indices_to_use= np.append(bert_char_indices_to_use, bert_char_indices_to_use2)
# # print(bert_char_indices_to_use.shape)
# # bert_voc_indices_to_plot = bert_char_indices_to_use
# # bert_voc_indices_to_use = bert_char_indices_to_use

# # print(len(bert_voc_indices_to_plot))
# # print(len(bert_voc_indices_to_use))

# # print(bertwords[bert_voc_indices_to_use])

# # bert_voc_indices_to_use_tensor = torch.LongTensor(bert_voc_indices_to_use)
# # bert_word_embs_to_use = wordembs(bert_voc_indices_to_use_tensor).detach().numpy()
# labeled_tokens = np.array(labeled_tokens)
# print(labeled_tokens.shape)

# Run t-SNE on the BERT vocabulary embeddings we selected:
mytsne_words = TSNE(n_components=2,early_exaggeration=12,verbose=2,metric='cosine',init='pca',n_iter=3500)
bert_word_embs_to_use_tsne = mytsne_words.fit_transform(labeled_tokens)


def loadLines(filename):
    print("Loading lines from file", filename)
    f = open(filename,'r',encoding = "utf-8")
    lines = np.array([])
    for line in f:
        lines = np.append(lines, line.rstrip())
    print("Done. ", len(lines)," lines loaded!")
    return lines
bertwords = loadLines('all_labeled_tokens.txt')


bert_char_indices_to_use = np.arange(0, 426, 1)
bert_voc_indices_to_plot= bert_char_indices_to_use
bert_words_to_plot = bertwords[bert_voc_indices_to_plot]
print(len(bert_words_to_plot))
# Plot the transformed BERT vocabulary embeddings:
plt.rcParams['figure.figsize'] = [70, 42]
fig = plt.figure() 
alltexts = list()
for i, txt in enumerate(bert_words_to_plot):
    plt.scatter(bert_word_embs_to_use_tsne[i,0], bert_word_embs_to_use_tsne[i,1], s=0)
    currtext = plt.text(bert_word_embs_to_use_tsne[i,0], bert_word_embs_to_use_tsne[i,1], txt, family='sans-serif')
    alltexts.append(currtext)
    

# Save the plot before adjusting.
# plt.savefig('viz-bert-phrase_cancervoc.pdf', format='pdf')
# print('now running adjust_text')
# # Using autoalign often works better in my experience, but it can be very slow for this case, so it's false by default below:
# #numiters = adjust_text(alltexts, autoalign=True, lim=50)
adjust_text(alltexts, autoalign=True, lim=50)
#print('done adjust text, num iterations: ', numiters)
plt.savefig('viz-bert_ori_voc-adj502.pdf', format='pdf')
plt.show