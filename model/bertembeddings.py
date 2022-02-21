# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 21:07:52 2021

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


#load bert origin
tokenizer_bert_origin = BertTokenizer.from_pretrained('./bert_origin/')
config_bert_origin  = BertConfig.from_json_file('bert_origin/bert_config.json')
config_bert_origin.output_hidden_states =True
model_bert_origin = BertForPreTraining(config_bert_origin)
model_bert_origin = load_tf_weights_in_bert(model = model_bert_origin,config = config_bert_origin,
                                tf_checkpoint_path="bert_origin/bert_model.ckpt")

#load biobert
tokenizer_biobert = BertTokenizer.from_pretrained('./biobert_v1_pubmed_pmc/')
config_biobert  = BertConfig.from_json_file('biobert_v1_pubmed_pmc/bert_config.json')
config_biobert.output_hidden_states =True
model_biobert = BertForPreTraining(config_biobert)
model_biobert = load_tf_weights_in_bert(model = model_biobert,config = config_biobert,
                                tf_checkpoint_path="biobert_v1_pubmed_pmc/biobert_model.ckpt")


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


texts_her2 = ["the patient is her2 positive",
         #"her hair is very long",
         "the patient's her2 gene was overexpressed",
         "evidence of amplification of her2 gene was detected"]

texts_lcis = ["lcis is located at 3 cm from nearest margin",
         #"her hair is very long",
         "margins are uninvolved by pleomorphic lcis",
         "right breast showed lcis"]

texts_dcis = ["there was evidence of angiolymphatic invasion as well as extensive dcis with high-grade necrosis",
          #"her hair is very long",
          "possible dcis near anterosuperior margin",
          "cancer diagnosis information diagnosis ductal carcinoma in situ (dcis) of left breast",
          "the re-excision showed residual dcis with a clear margin"]

texts_er = ["the patient is er positive",
         #"her hair is very long",
         "malignant neoplasm of upper-outer quadrant of right breast in female, er negative"]

texts_lcis = ["lcis is located at 3 cm from nearest margin",
         #"her hair is very long",
         "margins are uninvolved by pleomorphic lcis",
         "right breast showed lcis"]

text_receptors = ['her2','er','pr','estrogen receptor', 'progesterone receptor']
text_htype = ['dcis','ductal carcinoma in situ','lcis','lobular carcinoma in situ',
              'invasive carcinoma', 'squamous cell carcinoma','adenocarcinomas',
              'metastatic carcinoma']

# text_stage = ['t4','mx',' n0','m0','pta','ptnm','pt2','iiic','iiia','t4','n2','pt1b','ypt',
#               'pt3a', 'pt3c','pn2']

text_stage = ['mx','m0','iiic','iiia','pt1b','pt3a', 'pt3c','pn2']


receptor  = target_word_embeddings(model_blue_bert, tokenizer_blue_bert,text_receptors)

htype  = target_word_embeddings(model_bert_origin, tokenizer_bert_origin,text_htype)
stage  = target_word_embeddings(model_bert_origin, tokenizer_bert_origin,text_stage)

er  = target_word_embeddings(model_bert_origin, tokenizer_bert_origin,'estrogen receptor',['estrogen receptor'])
pr =  target_word_embeddings(model_bert_origin, tokenizer_bert_origin,'progesterone receptor',['progesterone receptor'])   
dcis  = target_word_embeddings(model_cancer_bert_ori, tokenizer_cancer_bert_ori,'dcis',texts_dcis)
lcis  = target_word_embeddings(model_cancer_bert, tokenizer_cancer_bert,'lcis',texts_lcis)    

pt  = target_word_embeddings(model_bert_origin, tokenizer_bert_origin,'pt2n1a',['pt2n1a'])
pn  = target_word_embeddings(model_bert_origin, tokenizer_bert_origin,'estrogen receptor',['estrogen receptor'])
pm =  target_word_embeddings(model_bert_origin, tokenizer_bert_origin,'progesterone receptor',['progesterone receptor'])   
from scipy.spatial.distance import cosine
# combined = [her2,er,pr]

# for i in combined:
#     for k in combined:
#         print( 1 - cosine(i, k))
# Calculating the distance between the
# embeddings of 'her2' in all the
# given contexts of the word

list_of_distances = []
for text1, embed1 in zip(text_stage, stage):
    for text2, embed2 in zip(text_stage , stage):
        # the larger value means close
        if text1 != text2:
            cos_dist = 1 - cosine(embed1, embed2)
            list_of_distances.append([text1, text2, cos_dist])

distances_df = pd.DataFrame(list_of_distances, columns=['text1', 'text2', 'distance'])

distances_df.to_csv('stage_distance_model_bert_origin.csv',index=False)

