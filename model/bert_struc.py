# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 15:24:14 2021

@author: zhou1281
"""


from transformers import BertTokenizer, BertModel,BertConfig,BertForPreTraining,load_tf_weights_in_bert
import pandas as pd
import numpy as np
import nltk
import torch
from pytorch_pretrained_bert import BertTokenizer

#load cancer bert customized vocab
tokenizer_cancer_bert = BertTokenizer.from_pretrained('./cancerbert_iter2/')
config_cancer_bert  = BertConfig.from_json_file('cancerbert_iter2/bert_config.json')
config_cancer_bert.output_hidden_states =True
model_cancer_bert = BertForPreTraining(config_cancer_bert)
model_cancer_bert = load_tf_weights_in_bert(model = model_cancer_bert,config = config_cancer_bert,
                                tf_checkpoint_path="cancerbert_iter2/bert_model.ckpt")



import sys

#sys.stdout = open("bert_structure.txt","w")
print(model_cancer_bert)
#sys.stdout.closed()