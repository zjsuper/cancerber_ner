# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 16:34:42 2021

@author: zhou1281
"""

from transformers import BertTokenizer, BertModel,BertConfig,BertForPreTraining,load_tf_weights_in_bert
import pandas as pd
import numpy as np
import nltk
import torch
from pytorch_pretrained_bert import BertTokenizer

#load blue bert
tokenizer_blue_bert = BertTokenizer.from_pretrained('./blue_bert/')
config_blue_bert  = BertConfig.from_json_file('blue_bert/bert_config.json')
config_blue_bert.output_hidden_states =True
model_blue_bert = BertForPreTraining(config_blue_bert)
model_blue_bert = load_tf_weights_in_bert(model = model_blue_bert,config = config_blue_bert,
                                tf_checkpoint_path="blue_bert/model.ckpt-1000000")

#load cancer bert customized vocab
tokenizer_cancer_bert = BertTokenizer.from_pretrained('./cancerbert_iter2/')
config_cancer_bert  = BertConfig.from_json_file('cancerbert_iter2/bert_config.json')
config_cancer_bert.output_hidden_states =True
model_cancer_bert = BertForPreTraining(config_cancer_bert)
model_cancer_bert = load_tf_weights_in_bert(model = model_cancer_bert,config = config_cancer_bert,
                                tf_checkpoint_path="cancerbert_iter2/bert_model.ckpt")

#load cancer bert origin vocab
tokenizer_cancer_bert_ori = BertTokenizer.from_pretrained('./cancerbert_based_on_origin_vocab/')
config_cancer_bert_ori  = BertConfig.from_json_file('cancerbert_based_on_origin_vocab/bert_config.json')
config_cancer_bert_ori.output_hidden_states =True
model_cancer_bert_ori = BertForPreTraining(config_cancer_bert_ori)
model_cancer_bert_ori = load_tf_weights_in_bert(model = model_cancer_bert_ori,config = config_cancer_bert_ori,
                                tf_checkpoint_path="cancerbert_based_on_origin_vocab/bert_model.ckpt")


df = pd.read_csv('word_pairs/UMNSRS_relatedness.csv')

t1 = df.Term1.tolist()
t1 = [i.lower() for i in t1]
t2 = df.Term2.tolist()
t2 = [i.lower() for i in t2]

# with open("cancerbert_iter2/vocab.txt","r",encoding = "utf-8") as f:
#     cust_lines = f.readlines()
# with open("cancerbert_based_on_origin_vocab/vocab.txt","r",encoding = "utf-8") as f:
#     ori_vocab = f.readlines()
# cust_lines = [i.strip() for i in lines]
# ori_vocab= [i.strip() for i in lines]
# a = 0
# for i in t2:
#     if i in cust_lines:
#         print(i)
#         a +=1
# b = 0
# for i in t2:
#     if i in ori_vocab:
#         print(i)
#         b +=1        

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



# Getting embeddings for the target
# word in all given contexts
# target_word_embeddings = []

#tokenized_text, tokens_tensor, segments_tensors = bert_text_preparation('the patient is her2 positive', tokenizer_blue_bert)
# tokenized_text, tokens_tensor, segments_tensors = bert_text_preparation("bank", tokenizer)
# list_token_embeddings = get_bert_embeddings(tokens_tensor, segments_tensors, model)
# word_index = tokenized_text.index('bank')
# word_embedding = list_token_embeddings[word_index]
# for text in texts:
#     tokenized_text, tokens_tensor, segments_tensors = bert_text_preparation(text, tokenizer_bert_origin)
#     list_token_embeddings = get_bert_embeddings(tokens_tensor, segments_tensors, model_bert_origin)
    
#     # Find the position 'her2' in list of tokens
#     tokenzied_word = tokenizer_bert_origin.tokenize('dcis')
    
#     word_index = []
#     for i in tokenzied_word:
#         word_index.append(tokenized_text.index(i))
#     # Get the embedding for bank
    
#     word_embedding = []
#     for i in word_index:
#         word_embedding.append(list_token_embeddings[i])
#     nums = len(word_embedding)
#     word_embedding = [sum(i) for i in zip(*word_embedding)]
#     word_embedding = [i/nums for i in word_embedding]
    

#     target_word_embeddings.append(word_embedding)
    
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



text_stage = ['mx','m0','iiic','iiia','pt1b','pt3a', 'pt3c','pn2']


stage  = target_word_embeddings(model_cancer_bert, tokenizer_cancer_bert,text_stage)
t1_cancerbert_embed = target_word_embeddings(model_cancer_bert, tokenizer_cancer_bert,t1)
t2_cancerbert_embed = target_word_embeddings(model_cancer_bert, tokenizer_cancer_bert,t2)

t1_cancerbert_ori_embed = target_word_embeddings(model_cancer_bert_ori, tokenizer_cancer_bert_ori,t1)
t2_cancerbert_ori_embed = target_word_embeddings(model_cancer_bert_ori, tokenizer_cancer_bert_ori,t2)

t1_bluebert_embed = target_word_embeddings(model_blue_bert, tokenizer_blue_bert,t1)
t2_bluebert_embed = target_word_embeddings(model_blue_bert, tokenizer_blue_bert,t2)

from scipy.spatial.distance import cosine
from numpy import linalg as LA

print(LA.norm(t1_cancerbert_embed[1]))
list_of_distances = []
dot = np.dot(t1_cancerbert_embed[1], t1_cancerbert_embed[1])
for i in range(len(t2_cancerbert_embed)):
        # the larger value means close
    #print(cosine(t1_cancerbert_embed[i], t2_cancerbert_embed[i]))
    cos_dist = 1 - cosine(t1_cancerbert_embed[i], t2_cancerbert_embed[i])
    list_of_distances.append(cos_dist)
gold_score =     df.Mean.tolist()
r = np.corrcoef(list_of_distances,df.Mean.tolist())


def pearson_corr(emb1,emb2):
    list_of_distances = []
    for i in range(len(emb1)):
            # the larger value means close
        #print(cosine(t1_cancerbert_embed[i], t2_cancerbert_embed[i]))
        cos = np.dot(emb1[i],emb2[i])/(LA.norm(emb1[i])*LA.norm(emb2[i]))
        list_of_distances.append(cos)
    gold_score = df.Mean.tolist()
    r = np.corrcoef(list_of_distances,df.Mean.tolist())
    print(r)
    return r
r = pearson_corr(t1_cancerbert_embed,t2_cancerbert_embed)
r_ori = pearson_corr(t1_cancerbert_ori_embed,t2_cancerbert_ori_embed)
r_blue = pearson_corr(t1_bluebert_embed,t2_bluebert_embed)