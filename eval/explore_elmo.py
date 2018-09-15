
# coding: utf-8

# In[5]:


from allennlp.modules.elmo import Elmo, batch_to_ids
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.stats import spearmanr
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import safe_sparse_dot


# In[2]:


options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

elmo = Elmo(options_file, weight_file, 1, dropout=0)


# In[25]:


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



def cosine_similarity2(X, Y=None, dense_output=True):
    """Compute cosine similarity between samples in X and Y.

    Cosine similarity, or the cosine kernel, computes similarity as the
    normalized dot product of X and Y:

        K(X, Y) = <X, Y> / (||X||*||Y||)

    On L2-normalized data, this function is equivalent to linear_kernel.

    Read more in the :ref:`User Guide <cosine_similarity>`.

    Parameters
    ----------
    X : ndarray or sparse array, shape: (n_samples_X, n_features)
        Input data.

    Y : ndarray or sparse array, shape: (n_samples_Y, n_features)
        Input data. If ``None``, the output will be the pairwise
        similarities between all samples in ``X``.

    dense_output : boolean (optional), default True
        Whether to return dense output even when the input is sparse. If
        ``False``, the output is sparse if both input arrays are sparse.

        .. versionadded:: 0.17
           parameter ``dense_output`` for dense output.

    Returns
    -------
    kernel matrix : array
        An array with shape (n_samples_X, n_samples_Y).
    """
    # to avoid recursive import

#     X, Y = check_pairwise_arrays(X, Y)
    X=np.array(X)
    Y=np.array(Y)
    X_normalized = X / np.sqrt((X * X).sum())
    if X is Y:
        Y_normalized = X_normalized
    else:
        Y_normalized = Y / np.sqrt((Y * Y).sum())
    
    print (np.linalg.norm(X_normalized),np.linalg.norm(Y_normalized[0]))
    K = safe_sparse_dot(X_normalized, Y_normalized.T, dense_output=dense_output)

    return K

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
#     vocab = normalize(vocab)
    s = np.sqrt((vocab * vocab).sum(1))
    s[s==0.] = 1.
    vocab /= s.reshape((s.shape[0], 1))
    
    x1=context_dependent(x,index)
#     x1=normalize([x1])[0]
    x1 = x1 / np.sqrt((x1 * x1).sum())

#     x1=x1[0]
#     print(np.linalg.norm(x1))
#     similarity=cosine_similarity2([x1],vocab)[0]
    similarity=vocab.dot(x1)
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


# In[6]:


#vocab
try:
    words_matrix=np.load('./lexsub_en/vocab2elmo.npy')
    vocab2index=np.load('./lexsub_en/vocab2index.npy')
    index2word={}
    word2index={}
    for vocab_index in vocab2index:
        word2index[vocab_index[0]]=int(vocab_index[1])
        index2word[int(vocab_index[1])]=vocab_index[0]
except:
    word2index={}
    index2word={}
    vocab_matrix=[]
    words=[]
    index=0
    with open('../corpora/1-billion-vocab') as f:
        for line in f:
            word=line.split()[0]
    #         vocab_matrix.append(context_independent(word))
            index2word[int(index)]=word
            word2index[word]=int(index)
            words.append(word)
            index+=1





    words_matrix=context_independent_batch(words[:10000],1000)
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


# In[29]:


get_ipython().run_line_magic('pinfo2', 'cosine_similarity')


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


# In[26]:


# nearest neighbour
sent='during the siege , george robertson had appointed shuja-ul-mulk , who was a bright boy only 12 years old and the youngest surviving son of aman-ul-mulk , as the ruler of chitral .'
words_lst=[w for w in sent.split()]
index=words_lst.index('bright')
print (words_lst,index)
nearest_neighbour(words_lst,words_matrix,index2word,30,index)
# cosine_similarity([words_matrix[0][1]],words_matrix[0])

