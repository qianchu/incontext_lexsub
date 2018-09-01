
# coding: utf-8

# In[5]:


import numpy
import six
import sys
import traceback
import re

from chainer import cuda
from context2vec.common.context_models import Toks
from context2vec.common.model_reader import ModelReader
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.stats import spearmanr


# In[11]:


model_param_file = '../models/context2vec/model_dir/MODEL-wiki.params.10'
n_result = 10  # number of search result to show
gpu = -1 # todo: make this work with gpu

if gpu >= 0:
         cuda.check_cuda_available()
         cuda.get_device(gpu).use()    

xp = cuda.cupy if gpu >= 0 else numpy

model_reader = ModelReader(model_param_file)
w = model_reader.w
word2index = model_reader.word2index
index2word = model_reader.index2word
model = model_reader.model


# In[12]:


#similarity simlex
predicts=[]
golds=[]
line_num=0
with open ('simlex') as f:
    for line in f:
        if line_num==0:
            line_num+=1
            continue
        
        line=line.split('\t')
        try:
            predict = cosine_similarity([w[word2index[line[0]]]],[w[word2index[line[1]]]])
        except KeyError as e:
            print (e)
            #predict = cosine_similarity([context_dependent([line[0]])],[context_dependent([line[1]])])
        
        print (line[3],predict[0][0])

        predicts.append(predict[0][0])

        golds.append(line[3])

        line_num+=1   
          
print ('simlex sim is {0}'.format(spearmanr(predicts,golds)))


# In[13]:


#MEN
predicts=[]
golds=[]
line_num=0
with open ('MEN_dataset_lemma_form_full','r') as f:
    for line in f:
        
    #         if line_num==0:
#             line_num+=1
#             continue
        line=line.split()
        
        try:
            
            predict = cosine_similarity([w[word2index[line[0].split('-')[0]]]],[w[word2index[line[1].split('-')[0]]]])
        except KeyError as e:
            print (e)
#             print (line[0].split('-')[0])
#             predict = cosine_similarity([context_dependent([line[0].split('-')[0]])],[context_dependent([line[1].split('-')[0]])])
        
        print (line[2],predict[0][0])

        predicts.append(predict[0][0])

        golds.append(float(line[2]))

        line_num+=1   
          
print ('MEN sim is {0}'.format(spearmanr(predicts,golds)))

