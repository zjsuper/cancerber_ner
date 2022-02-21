# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 12:09:13 2021

@author: zhou1281
"""

##check vocab with labels

vocab = []
count = 0
label = []
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
                        label.append(line[1].strip())
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
                        label.append(line[1].strip())
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
                        label.append(line[1].strip())
                        vocab.append(line[0].strip())
                        
                        
label=   list(set(label))    
vocab = list(set(vocab))  

print(len(vocab))

# with open('all_labeled_tokens.txt','w') as f:
#     for v in vocab:
#         f.write(v)
#         f.write('\n')
    
    
phrase = []   

from collections import defaultdict
dic= defaultdict(list)

with open('train.txt') as f:
    lines = f.readlines()
    i = 0
    while i < len(lines):
        print(i)
        if lines[i] == "\n":
            i+=1
            pass
        else:
            line =lines[i].split(' ')
            if len(line) != 2:
                i+=1
                print(line,i)
            else:
                if len(line)<2:
                    i+=1
                    print(line,i)
                else:
                    if line[1].strip()[0] == 'B':
                        label = line[1].strip()
                        temp = line[0].strip()
                        for j in range(i+1,len(lines)):
                            linej =lines[j].split(' ')
                            #print(linej)
                            if linej[0] == "\n":
                                phrase.append(temp)  
                                dic[label].append(temp)
                                i = j
                                break
                            elif linej[1].strip()[0] == 'I':
                                temp =temp+ ' '+linej[0].strip()
                            else:
                                phrase.append(temp)  
                                dic[label].append(temp)    
                                i = j
                                break
                    else:
                        i+=1
                        pass


with open('dev.txt') as f:
    lines = f.readlines()
    i = 0
    while i < len(lines):
        print(i)
        if lines[i] == "\n":
            i+=1
            pass
        else:
            line =lines[i].split(' ')
            if len(line) != 2:
                i+=1
                print(line,i)
            else:
                if len(line)<2:
                    i+=1
                    print(line,i)
                else:
                    if line[1].strip()[0] == 'B':
                        label = line[1].strip()
                        temp = line[0].strip()
                        for j in range(i+1,len(lines)):
                            linej =lines[j].split(' ')
                            #print(linej)
                            if linej[0] == "\n":
                                phrase.append(temp)  
                                dic[label].append(temp)
                                i = j
                                break
                            elif linej[1].strip()[0] == 'I':
                                temp =temp+ ' '+linej[0].strip()
                            else:
                                phrase.append(temp)  
                                dic[label].append(temp)    
                                i = j
                                break
                    else:
                        i+=1
                        pass
                    
with open('test.txt') as f:
    lines = f.readlines()
    i = 0
    while i < len(lines):
        print(i)
        if lines[i] == "\n":
            i+=1
            pass
        else:
            line =lines[i].split(' ')
            if len(line) != 2:
                i+=1
                print(line,i)
            else:
                if len(line)<2:
                    i+=1
                    print(line,i)
                else:
                    if line[1].strip()[0] == 'B':
                        label = line[1].strip()
                        temp = line[0].strip()
                        for j in range(i+1,len(lines)):
                            linej =lines[j].split(' ')
                            #print(linej)
                            if linej[0] == "\n":
                                phrase.append(temp)  
                                dic[label].append(temp)
                                i = j
                                break
                            elif linej[1].strip()[0] == 'I':
                                temp =temp+ ' '+linej[0].strip()
                            else:
                                phrase.append(temp)  
                                dic[label].append(temp)    
                                i = j
                                break
                    else:
                        i+=1
                        pass
                    
phrase_uniq = list(set(phrase))
print(len(phrase_uniq))

# with open('all_labeled_phrases.txt','w') as f:
#     for v in phrase_uniq:
#         f.write(v)
#         f.write('\n')


dic_vocab= defaultdict(list)

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
                            
                            
for k,v in dic_vocab.items():
    print(k,len(v))