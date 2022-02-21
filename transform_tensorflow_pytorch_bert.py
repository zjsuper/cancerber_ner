# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 13:02:02 2020

@author: zhou1281
"""

transformers-cli convert --model_type bert --tf_checkpoint \BERT_BASE_DIR\bert_model.ckpt --config \BERT_BASE_DIR\bert_config.json --pytorch_dump_output BERT_BASE_DIR\pytorch_model.bin



transformers-cli convert --model_type bert --tf_checkpoint bert_model.ckpt --config bert_config.json --pytorch_dump_output pytorch_model.bin