# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 15:20:04 2021

@author: zhou1281
"""

##check label integrety
labels = []
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
                    labels.append(line[1].strip())
            
labels = list(set(labels))  


labels = []
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
                    labels.append(line[1].strip())
            
labels = list(set(labels))  


labels = []
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
                    labels.append(line[1].strip())
            
labels = list(set(labels))            
#with open('dev.txt') as f , open('dev1.txt','wt') as out_put:
#    lines = f.readlines()
#    for i in lines:
#        if i == "\n":
#            out_put.write('')
#            out_put.write('\n')
#        else:
#            line = i.split('\t')
#            out_put.write(line[0]+' '+line[1])
#            
#with open('test.txt') as f , open('test1.txt','wt') as out_put:
#    lines = f.readlines()
#    for i in lines:
#        if i == "\n":
#            out_put.write('')
#            out_put.write('\n')
#        else:
#            line = i.split('\t')
#            out_put.write(line[0]+' '+line[1])