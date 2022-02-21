# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 15:06:44 2021

@author: zhou1281
"""



import numpy as np
import scipy.sparse as ss
import pandas as pd
import csv
import statistics
import sys
from collections import Counter 
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
import re, string, timeit
import random

# (1) read file, in list
#spliy by tab
#separate by space

with open('train.txt') as f , open('train1.txt','wt') as out_put:
    lines = f.readlines()
    for i in lines:
        if i == "\n":
            out_put.write('')
            out_put.write('\n')
        else:
            line = i.split('\t')
            out_put.write(line[0]+' '+line[1])
            
with open('dev.txt') as f , open('dev1.txt','wt') as out_put:
    lines = f.readlines()
    for i in lines:
        if i == "\n":
            out_put.write('')
            out_put.write('\n')
        else:
            line = i.split('\t')
            out_put.write(line[0]+' '+line[1])
            
with open('test.txt') as f , open('test1.txt','wt') as out_put:
    lines = f.readlines()
    for i in lines:
        if i == "\n":
            out_put.write('')
            out_put.write('\n')
        else:
            line = i.split('\t')
            out_put.write(line[0]+' '+line[1])