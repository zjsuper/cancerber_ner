# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 20:31:08 2021

@author: zhou1281
"""

import numpy as np
import scipy.stats as st
import argparse
from sklearn.metrics import f1_score
import pandas as pd

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

uniqb = ['B-receptor', 'B-htype', 'B-stage-value', 'B-grade-value', 
          'B-laterality-value', 'B-laterality', 'B-site', 'B-htype-value', 
          'B-grade', 'B-size-value', 'B-stage', 'B-site-value', 'B-size', 
          'B-receptor-status']

def p_r_f_strict(gold,test):
    g_num = len(gold)
    true_num = 0
    test_num = len(test)
    for i in range(len(gold)):
        if gold[i] in test:
            true_num+= 1
    if g_num == 0:
        recall = 0
    else:
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

def calculate_performance(file_names):
    #df = pd.DataFrame(columns = ['vars','precision','pci','recall','rci','F1','fci'])
    df = pd.DataFrame(columns = ['vars','F1','fci_0.05','fci_0.1','gold_num','micro','macro'])
    dic = {i:[] for i in uniqb}
    dicnum = {i:[] for i in uniqb}
    for fi in file_names:
        with open(fi,'r') as f:
            lines = f.readlines()
        for var in uniqb:
            g,t = gold_test_index(lines,var)
            dicnum[var].append(len(g))
            p,r,f = p_r_f_strict(g,t)
            dic[var].append(f)
    varlist = []
    f1 = []
    fci005 = []
    fci01 = []
    total_num = []
    for k,v in dic.items():
        varlist.append(k)
        f1.append(np.mean(v))
        fci005.append(st.t.interval(0.95,len(v)-1,loc = np.mean(v),scale = st.sem(v)))
        fci01.append(st.t.interval(0.90,len(v)-1,loc = np.mean(v),scale = st.sem(v)))
        total_num.append(sum(dicnum[k]))
    df['vars'] = varlist
    df['F1'] = f1
    df['fci_0.05'] = fci005
    df['fci_0.1'] = fci01
    df['gold_num'] = total_num
    df['micro'] = sum([a*b for a,b in zip(f1,total_num)])/sum(total_num)
    df['macro'] = sum(f1)/len(f1)
    
    return df
    
def calculate_performance_lenient(file_names):
    #df = pd.DataFrame(columns = ['vars','precision','pci','recall','rci','F1','fci'])
    df = pd.DataFrame(columns = ['vars','F1','fci_0.05','fci_0.1','gold_num','micro','macro'])
    dic = {i:[] for i in uniqb}
    dicnum = {i:[] for i in uniqb}
    for fi in file_names:
        with open(fi,'r') as f:
            lines = f.readlines()
        for var in uniqb:
            g,t = gold_test_index(lines,var)
            dicnum[var].append(len(g))
            p,r,f = p_r_f_lenient(g,t)
            dic[var].append(f)
    varlist = []
    f1 = []
    fci005 = []
    fci01 = []
    total_num = []
    for k,v in dic.items():
        varlist.append(k)
        f1.append(np.mean(v))
        fci005.append(st.t.interval(0.95,len(v)-1,loc = np.mean(v),scale = st.sem(v)))
        fci01.append(st.t.interval(0.90,len(v)-1,loc = np.mean(v),scale = st.sem(v)))
        total_num.append(sum(dicnum[k]))
    df['vars'] = varlist
    df['F1'] = f1
    df['fci_0.05'] = fci005
    df['fci_0.1'] = fci01
    df['gold_num'] = total_num
    df['micro'] = sum([a*b for a,b in zip(f1,total_num)])/sum(total_num)
    df['macro'] = sum(f1)/len(f1)
    
    return df

file_names = ['label_test_0.txt','label_test_1.txt',
              'label_test_2.txt','label_test_3.txt','label_test_4.txt',
              'label_test_5.txt','label_test_6.txt',
              'label_test_7.txt','label_test_8.txt','label_test_9.txt']
df = calculate_performance_lenient(file_names)
dfs = calculate_performance(file_names)
df.to_csv('cancerbert_cust_vocab_lenient.csv',index = False)
dfs.to_csv('cancerbert_cust_vocab_strict.csv',index = False)