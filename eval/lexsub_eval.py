
# coding: utf-8

# In[63]:


from allennlp.modules.elmo import Elmo, batch_to_ids
import sys
import numpy as np

get_ipython().run_line_magic('pinfo2', 'batch_to_ids')


# In[64]:


def read_gold_candidates(gold_candis_f):
    ## read in substitutes apart from MWE
    target_w2candidates={}
    with open (gold_candis_f) as f:
        for line in f:
            line=line.strip()
            target_w,candidates=line.split('::')
            candidates=[candi for candi in candidates.split(';') if ' ' not in candi]
            target_w2candidates[target_w]=candidates
    return target_w2candidates


def read_eval_data(data_f):
    sents=[]
    pos_lst=[]
    target_ws=[]
    with open (data_f,encoding='utf-8',errors='replace') as f:
        try:
            for line in f:
                line=line.strip().split('\t')
                target_ws.append(line[0])
                pos_lst.append(int(line[2]))
                sent_w_lst=line[3].split()
                if sent_w_lst[int(line[2])]!=line[0].split('.')[0]:
                    print ('warning: position {2}, target_w:{0}:{1}'.format(line[0],sent_w_lst[int(line[2])],line[2]))
                sents.append(sent_w_lst)
        except UnicodeDecodeError as e:
            print (line)
            print (e)
    return sents,pos_lst,target_ws



##incontext representation
def context_dependent(x,index=0):
    '''
    eg. x=['apple']
    '''
    sentences=[x]
    character_ids = batch_to_ids(sentences)
    embeddings = elmo(character_ids)
    print (embeddings['elmo_representations'][0].shape)
    x1=embeddings['elmo_representations'][0][0][index].detach().numpy()
    return x1

def context_dependent_batch(sent_lst,pos_lst,batchsize):
    '''
    sent_lst=[['there','is','an','apple','.'],['there','is','an','apple','.']]
    pos_lst=[2,3] the position list
    '''
    character_ids_all = batch_to_ids(sent_lst)
    context_reps=[]
    for i in range(int(len(sent_lst)/batchsize)+1):
        print (i)
        start=i*batchsize
        end=(i+1)*batchsize
        if start>=len(sent_lst):
            break
        if end > len(sent_lst):
            end=len(sent_lst)
        sentences=sent_lst[start:end]
        pos=pos_lst[start:end]
        character_ids=character_ids_all[start:end]
        embeddings = elmo(character_ids)
        context_reps.append(embeddings['elmo_representations'][0].detach().numpy()[np.arange(len(sentences)),np.array(pos)])
    context_reps=np.concatenate(context_reps,axis=0)
    return context_reps
        
## substitute ranking


# In[3]:


if __name__=="__main__":
    
    #elmo parameters
    elmo_options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    elmo_weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

    if sys.argv[0]=='/usr/local/lib/python3.6/site-packages/ipykernel_launcher.py':
        model='elmo'
        data_f='/home/incontext_lexsub/datasets/lst_test.preprocessed'
        gold_candis_f='/home/incontext_lexsub/datasets/lst.gold.candidates'
        
    else:
        model=sys.argv[1]
        data_f=sys.argv[2]
        gold_candis_f=sys.argv[3]
    
    
   
   


# In[4]:


# 1. read in model
if model=='elmo':
    elmo = Elmo(elmo_options_file, elmo_weight_file, 1, dropout=0)
else:
    raise NotImplementedError


# In[65]:


# 2. read in gold substitutes
target_w2candidates=read_gold_candidates(gold_candis_f)
# 3. evaluate on the data:
sents,pos_lst,target_ws=read_eval_data(data_f)

# 3.1. context representations
contexts_rep=context_dependent_batch(sents,pos_lst,100)


# In[66]:


contexts_rep.shape


# In[31]:


contexts_rep[np.arange(4),np.array([0,1,0,1])]


# In[20]:


contexts_rep[1][0]

