# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 11:52:10 2021

@author: zhou1281
"""
with open('train.txt','r') as f:
    lines1 = f.readlines()
    
with open('dev.txt','r') as f:
    lines2 = f.readlines()

with open('test.txt','r') as f:
    lines3 = f.readlines()
lines = lines1+lines2+lines3
lines = [i for i in lines if i!='\n']
token  = []
gold = []
test = []


uniq_b = []

for i in range(len(lines)):
    
    k = lines[i].split(' ')
    k = [m.strip() for m in k]
    if k[1][0] != 'O':   
        uniq_b.append(k[1])
            
            
uniq_b = list(set(uniq_b))

print(uniq_b)

# import pandas as pd

# gold_num = []
# variables = []
# for v in uniq_b:
#     variables.append(v)
#     gold_count = 0 
#     for i in range(len(lines)):
#         k = lines[i].split(' ')
#         k = [m.strip() for m in k]
#         if k[1] == v:   
#             gold_count+= 1
            
#     gold_num.append(gold_count)
        

# df = pd.DataFrame(columns = ['entity','gold_num'])
# df['entity'] = variables
# df['gold_num'] = gold_num 

# df.to_csv('annatation_number.csv',index = False)