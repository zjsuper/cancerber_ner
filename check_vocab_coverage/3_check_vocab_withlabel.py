# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 12:09:13 2021

@author: zhou1281
"""

##check vocab with labels

vocab = []
count = 0
with open('train.txt') as f:
    lines = f.readlines()
    for i in lines:
        count += 1
        if i == "\n":
            pass
        else:
            line = i.split(' ')
            if len(line) != 2:
                print(line,count)
            else:
                if len(line[1])<2:
                    print(line,count)
                else:
                    if line[1].strip() != 'O':
                        vocab.append(line[0].strip())


count = 0
with open('dev.txt') as f:
    lines = f.readlines()
    for i in lines:
        count += 1
        if i == "\n":
            pass
        else:
            line = i.split(' ')
            if len(line) != 2:
                print(line,count)
            else:
                if len(line[1])<2:
                    print(line,count)
                else:
                    if line[1].strip() != 'O':
                        vocab.append(line[0].strip())
                        
count = 0
with open('test.txt') as f:
    lines = f.readlines()
    for i in lines:
        count += 1
        if i == "\n":
            pass
        else:
            line = i.split(' ')
            if len(line) != 2:
                print(line,count)
            else:
                if len(line[1])<2:
                    print(line,count)
                else:
                    if line[1].strip() != 'O':
                        vocab.append(line[0].strip())
                        
                        
            
vocab = list(set(vocab))  


v997 = []
with open('vocab_997.txt',encoding = "UTF-8") as f:
    lines = f.readlines()
    for i in lines:
        if i == "\n":
            pass
        else:
            v997.append(i.strip())
v397 = []
with open('vocab_397.txt',encoding = "UTF-8") as f:
    lines = f.readlines()
    for i in lines:
        if i == "\n":
            pass
        else:
            v397.append(i.strip())



vori = []
with open('vocab_ori.txt',encoding = "UTF-8") as f:
    lines = f.readlines()
    for i in lines:
        if i == "\n":
            pass
        else:
            vori.append(i.strip())

c997 = 0
cori=0
c397=0
for i in vocab:
    if i in v997:
        c997+=1
    if i in v397:
        c397+=1
    if i in vori:
        cori+=1


from collections import defaultdict
dic_vocab= defaultdict(list)

count = 0
with open('train.txt') as f:
    lines = f.readlines()
    for i in lines:
        count += 1
        if i == "\n":
            pass
        else:
            line = i.split(' ')
            if len(line) != 2:
                print(line,count)
            else:
                if len(line[1])<2:
                    print(line,count)
                else:
                    if line[1].strip() == 'I-size-value' or line[1].strip() == 'B-size-value' or line[1].strip() == 'B-size':
                        if line[0].strip() not in dic_vocab['size']:
                            dic_vocab['size'].append(line[0].strip())
                    elif line[1].strip() == 'I-receptor' or line[1].strip() == 'B-receptor':
                        if line[0].strip() not in dic_vocab['receptor_type']:
                            dic_vocab['receptor_type'].append(line[0].strip())                        
                        
                    elif line[1].strip() == 'I-receptor-value' or line[1].strip() == 'B-receptor-value':
                        if line[0].strip() not in dic_vocab['receptor_status']:
                            dic_vocab['receptor_status'].append(line[0].strip())
                    elif line[1].strip() == 'B-htype-value' or line[1].strip() == 'I-htype-value' or line[1].strip() == 'B-htype':
                        if line[0].strip() not in dic_vocab['B-htype']:
                            dic_vocab['B-htype'].append(line[0].strip())                            
                    elif line[1].strip() == 'B-laterality-value' or line[1].strip() == 'B-laterality':
                        if line[0].strip() not in dic_vocab['B-laterality']:
                            dic_vocab['B-laterality'].append(line[0].strip())                                
                    elif line[1].strip() == 'B-stage-value' or line[1].strip() == 'B-stage':
                        if line[0].strip() not in dic_vocab['B-stage']:
                            dic_vocab['B-stage'].append(line[0].strip())          
                    elif line[1].strip() == 'B-grade-value' or line[1].strip() == 'B-grade':
                        if line[0].strip() not in dic_vocab['B-grade']:
                            dic_vocab['B-grade'].append(line[0].strip())                                 
                    elif line[1].strip() == 'I-site-value' or line[1].strip() == 'B-site-value' or line[1].strip() == 'B-site':
                        if line[0].strip() not in dic_vocab['site']:
                            dic_vocab['site'].append(line[0].strip())              
count = 0
with open('dev.txt') as f:
    lines = f.readlines()
    for i in lines:
        count += 1
        if i == "\n":
            pass
        else:
            line = i.split(' ')
            if len(line) != 2:
                print(line,count)
            else:
                if len(line[1])<2:
                    print(line,count)
                else:
                    if line[1].strip() == 'I-size-value' or line[1].strip() == 'B-size-value' or line[1].strip() == 'B-size':
                        if line[0].strip() not in dic_vocab['size']:
                            dic_vocab['size'].append(line[0].strip())
                    elif line[1].strip() == 'I-receptor' or line[1].strip() == 'B-receptor':
                        if line[0].strip() not in dic_vocab['receptor_type']:
                            dic_vocab['receptor_type'].append(line[0].strip())                        
                        
                    elif line[1].strip() == 'I-receptor-value' or line[1].strip() == 'B-receptor-value':
                        if line[0].strip() not in dic_vocab['receptor_status']:
                            dic_vocab['receptor_status'].append(line[0].strip())
                    elif line[1].strip() == 'B-htype-value' or line[1].strip() == 'I-htype-value' or line[1].strip() == 'B-htype':
                        if line[0].strip() not in dic_vocab['B-htype']:
                            dic_vocab['B-htype'].append(line[0].strip())                            
                    elif line[1].strip() == 'B-laterality-value' or line[1].strip() == 'B-laterality':
                        if line[0].strip() not in dic_vocab['B-laterality']:
                            dic_vocab['B-laterality'].append(line[0].strip())                                
                    elif line[1].strip() == 'B-stage-value' or line[1].strip() == 'B-stage':
                        if line[0].strip() not in dic_vocab['B-stage']:
                            dic_vocab['B-stage'].append(line[0].strip())          
                    elif line[1].strip() == 'B-grade-value' or line[1].strip() == 'B-grade':
                        if line[0].strip() not in dic_vocab['B-grade']:
                            dic_vocab['B-grade'].append(line[0].strip())                                 
                    elif line[1].strip() == 'I-site-value' or line[1].strip() == 'B-site-value' or line[1].strip() == 'B-site':
                        if line[0].strip() not in dic_vocab['site']:
                            dic_vocab['site'].append(line[0].strip())   
                        
count = 0
with open('test.txt') as f:
    lines = f.readlines()
    for i in lines:
        count += 1
        if i == "\n":
            pass
        else:
            line = i.split(' ')
            if len(line) != 2:
                print(line,count)
            else:
                if len(line[1])<2:
                    print(line,count)
                else:
                    if line[1].strip() == 'I-size-value' or line[1].strip() == 'B-size-value' or line[1].strip() == 'B-size':
                        if line[0].strip() not in dic_vocab['size']:
                            dic_vocab['size'].append(line[0].strip())
                    elif line[1].strip() == 'I-receptor' or line[1].strip() == 'B-receptor':
                        if line[0].strip() not in dic_vocab['receptor_type']:
                            dic_vocab['receptor_type'].append(line[0].strip())                        
                        
                    elif line[1].strip() == 'B-receptor-status':
                        if line[0].strip() not in dic_vocab['receptor_status']:
                            dic_vocab['receptor_status'].append(line[0].strip())
                    elif line[1].strip() == 'B-htype-value' or line[1].strip() == 'I-htype-value' or line[1].strip() == 'B-htype':
                        if line[0].strip() not in dic_vocab['B-htype']:
                            dic_vocab['B-htype'].append(line[0].strip())                            
                    elif line[1].strip() == 'B-laterality-value' or line[1].strip() == 'B-laterality':
                        if line[0].strip() not in dic_vocab['B-laterality']:
                            dic_vocab['B-laterality'].append(line[0].strip())                                
                    elif line[1].strip() == 'B-stage-value' or line[1].strip() == 'B-stage':
                        if line[0].strip() not in dic_vocab['B-stage']:
                            dic_vocab['B-stage'].append(line[0].strip())          
                    elif line[1].strip() == 'B-grade-value' or line[1].strip() == 'B-grade':
                        if line[0].strip() not in dic_vocab['B-grade']:
                            dic_vocab['B-grade'].append(line[0].strip())                                 
                    elif line[1].strip() == 'I-site-value' or line[1].strip() == 'B-site-value' or line[1].strip() == 'B-site':
                        if line[0].strip() not in dic_vocab['site']:
                            dic_vocab['site'].append(line[0].strip())   

                            
def count(dic_vocab,vocab):
    for k,v in dic_vocab.items():
        t = 0
    #print(k,len(v))
        for vo in vocab:
            if  vo in v:
                t+=1
        print(k,t)
    
count(dic_vocab,v397)
    
for k,v in dic_vocab.items():
    print(k,len(v))
