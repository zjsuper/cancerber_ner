#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Author:zhoukaiyin
"""

import pandas as pd

#df1= pd.read_csv('lenient_cancerbert1.csv')
#df2= pd.read_csv('lenient_cancerbert2.csv')
#df3= pd.read_csv('lenient_cancerbert3.csv')
#
#df4  = pd.concat([df1,df2,df3],ignore_index = True)
#stats = df4.groupby(['entity'])['precision','recall','f1_lenient'].agg(['mean','count','std'])
#stats.to_csv('cancer_bluebert.csv')


df1= pd.read_csv('strict_biobert_disc1.csv')
df2= pd.read_csv('strict_biobert_disc2.csv')
df3= pd.read_csv('strict_biobert_disc3.csv')

df4  = pd.concat([df1,df2,df3],ignore_index = True)
stats = df4.groupby(['entity'])['precision','recall','f1_strict'].agg(['mean','count','std'])
stats.to_csv('biobert_disc_strict.csv')