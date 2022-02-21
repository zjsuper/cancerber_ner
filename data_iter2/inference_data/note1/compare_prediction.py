# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 09:43:45 2021

@author: zhou1281
"""
cc = 0
with open('label_test1.txt','r') as f:
    lines = f.readlines()
    print(len(lines))


c = 0
with open('test.txt','r') as f:
    lines = f.readlines()
    print(len(lines))
    for i in range(len(lines)):
        temp = lines[i].strip()
        temp = temp.split(' ')
        if len(temp)>128:
            c += 1


print(c)
