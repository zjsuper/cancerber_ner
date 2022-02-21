# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 09:16:32 2021

@author: zhou1281
"""

#########emsemble methods



import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from sklearn.metrics import f1_score
import pandas as pd

with open('./result_dir_cancerbert5/label_test.txt','r') as f:
    lines1 = f.readlines()

with open('./result_dir_bluebert3/label_test.txt','r') as f:
    lines2 = f.readlines()
    
with open('./result_dir_biobert_pretrain_output_disc_3/label_test.txt','r') as f:
    lines3 = f.readlines()

with open('./result_dir_biobert_v1_pubmed_pmc_3/label_test.txt','r') as f:
    lines4 = f.readlines()
token  = []
gold = []
test = []

        
def most_frequent(l):
    return max(set(l),key = l.count)

#ll = ['a','a','b','b','c','d']
#
#a = most_frequent(ll)

uniq_b1 = []

for i in range(len(lines1)):
    k = lines1[i].split('\t')
    k = [m.strip() for m in k]
    if k[1] != 'O' and k[1][0] =='B':
        uniq_b1.append(k[1])
uniq_b1 = list(set(uniq_b1))

uniq_b2 = []

for i in range(len(lines2)):
    k = lines1[i].split('\t')
    k = [m.strip() for m in k]
    if k[1] != 'O' and k[1][0] =='B':
        uniq_b2.append(k[1])
uniq_b2 = list(set(uniq_b2))

uniq_b3 = []

for i in range(len(lines3)):
    k = lines1[i].split('\t')
    k = [m.strip() for m in k]
    if k[1] != 'O' and k[1][0] =='B' and  k[1] !='B-laterality' and  k[1] !='B-site' and  k[1] !='B-htype':
        uniq_b3.append(k[1])
uniq_b3 = list(set(uniq_b3))

print(uniq_b3)

uniq_b4 = []

for i in range(len(lines4)):
    k = lines4[i].split('\t')
    k = [m.strip() for m in k]
    if k[1] != 'O' and k[1][0] =='B':
        uniq_b4.append(k[1])
uniq_b4 = list(set(uniq_b4))

#dic = {i:[] for i in uniq_b}

def gold_test_index(lines,var):
    gold = []
    test = []
    token = []
    goldlist = []
    testlist = []
    for i in range(len(lines)):
        k = lines[i].split('\t')
        k = [m.strip() for m in k]
        #token.append(k[0])
    #if k[1] != 'O' and k[1] !='B-laterality' and k[1] !='B-site' and k[1] !='B-grade':
    #if k[1] != 'O' and k[1] !='B-laterality' :
        goldlist.append(k[1])
        testlist.append(k[2])
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
#            gold_token = []
#            test_token = []
#            for i,j in zip(gold,test):
#                gold_token.append(goldlist[i[0]:i[1]+1])
#                test_token.append(testlist[j[0]:j[1]+1])
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
    for v in uniq_b1:
        g1,t1 = gold_test_index(lines1,v)
        g2,t2 = gold_test_index(lines2,v)
        g3,t3 = gold_test_index(lines3,v)
        g4,t4 = gold_test_index(lines4,v)
        print(g1,g2)
        assert g1==g2,'g1 != g2'
        assert g1==g3,'g1 != g3'
        assert g1==g4,'g1 != g4'
        t = []
        for i in range(len(t1)):
            t.append(most_frequent([t1[i],t2[i],t3[i],t4[i]]))
        p,r,f = p_r_f_strict(g1,t)
        precision.append(p)
        recall.append(r)
        variables.append(v)
        f1_strict.append(f)
    df = pd.DataFrame(columns = ['entity','precision','recall','f1_strict'])
    df['entity'] = variables
    df['precision'] = precision
    df['recall'] = recall
    df['f1_strict'] = f1_strict
    df.to_csv('strict_emsemble1.csv', index = False)
    print('Strict p,r,f', v,p,r,f)
    
    
performance_strict_output()