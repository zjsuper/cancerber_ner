# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 12:02:25 2021

@author: zhou1281
"""

with open('whole.txt','r') as f:
    lines = f.readlines()

lines = [i for i in lines if i!='\n']
token  = []
gold = []
test = []


uniq_b = []

for i in range(len(lines)):
    
    k = lines[i].split(' ')
    k = [m.strip() for m in k]
    if k[1][0] == 'B':   
        uniq_b.append(k[1])
            
            
uniq_b = list(set(uniq_b))

print(uniq_b)

import pandas as pd

gold_num = []
variables = []
for v in uniq_b:
    variables.append(v)
    gold_count = 0 
    for i in range(len(lines)):
        k = lines[i].split(' ')
        k = [m.strip() for m in k]
        if k[1] == v:   
            gold_count+= 1
            
    gold_num.append(gold_count)
        

df = pd.DataFrame(columns = ['entity','gold_num'])
df['entity'] = variables
df['gold_num'] = gold_num 

df.to_csv('annatation_number.csv',index = False)