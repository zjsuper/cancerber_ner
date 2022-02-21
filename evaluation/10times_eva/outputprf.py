# -*- coding: utf-8 -*-
"""
Created on Wed May 26 22:47:18 2021

@author: zhou1281
"""


# import torch
# import torch.nn as nn
# import torch.optim as optim
import numpy as np
import argparse
#from sklearn.metrics import f1_score
import pandas as pd

with open('./label_test_14.txt','r') as f:
    lines = f.readlines()

token  = []
gold = []
test = []


uniq_b = []

for i in range(len(lines)):
    k = lines[i].split('\t')
    k = [m.strip() for m in k]
    if k[1] != 'O' and k[1][0] =='B':
        uniq_b.append(k[1])
uniq_b = list(set(uniq_b))

print(uniq_b)

uniq_b2 = []

for i in range(len(lines)):
    k = lines[i].split('\t')
    k = [m.strip() for m in k]
    if k[1] != 'O' and k[1][0] =='B' and  k[1] !='B-laterality' and  k[1] !='B-site' and  k[1] !='B-htype':
        uniq_b2.append(k[1])
uniq_b2 = list(set(uniq_b2))

print(uniq_b2)

dic = {i:[] for i in uniq_b}

def gold_test_index(lines,var):
    gold = []
    test = []
    token = []
    for i in range(len(lines)):
        k = lines[i].split('\t')
        k = [m.strip() for m in k]
        #token.append(k[0])
    #if k[1] != 'O' and k[1] !='B-laterality' and k[1] !='B-site' and k[1] !='B-grade':
    #if k[1] != 'O' and k[1] !='B-laterality' :
        if k[1] == var:
            whole_start = i 
            #whole_test = k[2]
            for j in range(i+1,len(lines)):
                if lines[j] != '\n':
                    aa = lines[j].split('\t')
                    aa = [n.strip() for n in aa]
                    ss = aa[1]
                    if ss =='I'+var[1:]:
                        pass
                    else:
                        whole_end = j-1
                        break
                else:
                    break
            gold.append([whole_start,whole_end])
            token.append([whole_start,whole_end])
        if k[2] == var:
            whole_start_test = i 
            #whole_test = k[2]
            for m in range(i+1,len(lines)):
                if lines[m] != '\n':
                    aa = lines[m].split('\t')
                    aa = [n.strip() for n in aa]
                    ss = aa[2]
                    if ss =='I'+var[1:]:
                        pass
                    else:
                        whole_end_test = m-1
                        break
                else:
                    break            
            test.append([whole_start_test,whole_end_test])
    return gold,test



def p_r_f_strict(gold,test):
    g_num = len(gold)
    true_num = 0
    test_num = len(test)
    for i in range(len(gold)):
        if gold[i] in test:
            true_num+= 1
    recall = true_num/g_num
    if test_num>0:
        precision = true_num/test_num
    else:
        precision = 0
    if recall == 0 and precision == 0:
        f1= 0
    else:
        f1 = 2*precision*recall/(precision+recall)
    return precision, recall, f1

def performance_strict_output():
    precision = []
    recall = []
    f1_strict = []
    variables = []
    gold_num = []
    for v in uniq_b:
        gold_count = 0 
        g,t = gold_test_index(lines,v)
        for go in g:
            if go != 'O':
                gold_count += 1
        gold_num.append(gold_count)
        
        p,r,f = p_r_f_strict(g,t)
        precision.append(p)
        recall.append(r)
        variables.append(v)
        f1_strict.append(f)
    df = pd.DataFrame(columns = ['entity','precision','recall','f1_strict','gold_num'])
    df['entity'] = variables
    df['precision'] = precision
    df['recall'] = recall
    df['f1_strict'] = f1_strict
    df['gold_num'] = gold_num
    df.to_csv('strict_bert_origin4.csv', index = False)
    print('Strict p,r,f', v,p,r,f)


performance_strict_output()
#print('Overall f1_strict: ',sum(f1_strict)/len(f1_strict))
    
    
    
def p_r_f_lenient(gold,test):
    g_num = len(gold)
    true_num = 0
    test_num = len(test)
    for i in range(len(gold)):
        for t in test:
            if (gold[i][0]<= t[0] and t[0] <= gold[i][1]) or (gold[i][0]<= t[1] and t[1] <= gold[i][1]) or (t[0] <= gold[i][0] and gold[i][1]<= t[1]):
                true_num += 1
                break
    recall = true_num/g_num
    if test_num>0:
        precision = true_num/test_num
    else:
        precision = 0
    if recall == 0 and precision == 0:
        f1= 0
    else:
        f1 = 2*precision*recall/(precision+recall)
    return precision, recall, f1

def performance_lenient_output():
    precision = []
    recall = []
    f1_lenient = []
    variables = []
    gold_num = []
    for v in uniq_b:
        gold_count = 0
        g,t = gold_test_index(lines,v)
        for go in g:
            if go != 'O':
                gold_count += 1
        gold_num.append(gold_count)        
        p,r,f = p_r_f_lenient(g,t)
        precision.append(p)
        recall.append(r)
        variables.append(v)
        f1_lenient.append(f)
    df = pd.DataFrame(columns = ['entity','precision','recall','f1_lenient','gold_num'])
    df['entity'] = variables
    df['precision'] = precision
    df['recall'] = recall
    df['f1_lenient'] = f1_lenient
    df['gold_num'] = gold_num
    df.to_csv('lenient_bert_origin4.csv', index = False)
    #print('Strict p,r,f', v,p,r,f)
    
performance_lenient_output()