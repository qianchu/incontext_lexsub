
# coding: utf-8

# In[1]:


from allennlp.modules.elmo import Elmo, batch_to_ids
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.stats import spearmanr


# In[2]:


options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

elmo = Elmo(options_file, weight_file, 1, dropout=0)


# In[3]:


# def context_independent_sim(x,y):
# # use batch_to_ids to convert sentences to character ids
#     sentences = [[x], [y]]
#     character_ids = batch_to_ids(sentences)
#     embeddings = elmo(character_ids)
#     x1=embeddings['elmo_representations'][0][0].detach().numpy()
#     x2=embeddings['elmo_representations'][0][1].detach().numpy()
# #     print (x1)
#     return cosine_similarity(x1,x2)
    
# allennlp.modules.scalar_mix.ScalarMix??


# def context_independent(x):
#     sentences=[[x]]
#     character_ids = batch_to_ids(sentences)
#     embeddings = elmo(character_ids)
#     x1=embeddings['elmo_representations'][0][0].detach().numpy()
#     return x1


def context_independent_batch(words,batchsize):
    words_matrix=[]
    for i in range(int(len(words)/batchsize)+1):
        print (i)
        start=i*batchsize
        end=(i+1)*batchsize
        if end > len(words):
            end=len(words)
        sentences=[[w] for w in words[start:end]]
        character_ids = batch_to_ids(sentences)
        embeddings = elmo(character_ids)
        x_ls=embeddings['elmo_representations'][0].detach().numpy()
        words_matrix.append(x_ls)
        
    words_matrix_all=np.concatenate(words_matrix,axis=0)
#     words_matrix=words_matrix.reshape(words_matrix.shape[0],words_matrix.shape[2])
    return words_matrix_all
    
    
def context_dependent(x,index=0):
    sentences=[x]
    character_ids = batch_to_ids(sentences)
    embeddings = elmo(character_ids)
    print (embeddings['elmo_representations'][0].shape)
    x1=embeddings['elmo_representations'][0][0][index].detach().numpy()
    return x1

def nearest_neighbour(x,vocab,index2word,n_result,index=0):
   
    x1=context_dependent(x,index)
    similarity=cosine_similarity([x1],vocab)[0]
    count=0
    for i in (-similarity).argsort():
                    if np.isnan(similarity[i]):
                        continue
                    print('{0}: {1}'.format(str(index2word[i]), str(similarity[i])))
                    count += 1
#                     top_words_i.append(i)
#                     top_words.append(index2word[i])
#                     similarity_scores.append(similarity[i])
                    if count == n_result:
                        break
    return 


# In[ ]:


#vocab

word2index={}
index2word={}
vocab_matrix=[]
words=[]
index=0
with open('test_vocab_bnc') as f:
    for line in f:
        word=line.split()[2]
#         vocab_matrix.append(context_independent(word))
        index2word[index]=word
        word2index[word]=index
        words.append(word)
        index+=1
        
    



words_matrix=context_independent_batch(words,1000)
len(words_matrix)


# In[ ]:


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
            predict = cosine_similarity(words_matrix[word2index[line[0]]],words_matrix[word2index[line[1]]])
        except KeyError as e:
            predict = cosine_similarity([context_dependent([line[0]])],[context_dependent([line[1]])])
        
        print (line[3],predict[0][0])

        predicts.append(predict[0][0])

        golds.append(line[3])

        line_num+=1   
          
print ('simlex sim is {0}'.format(spearmanr(predicts,golds)))


# In[ ]:


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
            
            predict = cosine_similarity(words_matrix[word2index[line[0].split('-')[0]]],words_matrix[word2index[line[1].split('-')[0]]])
        except KeyError as e:
#             print (line[0].split('-')[0])
            predict = cosine_similarity([context_dependent([line[0].split('-')[0]])],[context_dependent([line[1].split('-')[0]])])
        
        print (line[2],predict[0][0])

        predicts.append(predict[0][0])

        golds.append(float(line[2]))

        line_num+=1   
          
print ('MEN sim is {0}'.format(spearmanr(predicts,golds)))


# In[ ]:


# nearest neighbour
#words_matrix2=deepcopy(words_matrix)
# nearest_neighbour(['old'],words_matrix,index2word,30,index=0)
# cosine_similarity([words_matrix[0][1]],words_matrix[0])

