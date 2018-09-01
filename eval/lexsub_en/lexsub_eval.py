
# coding: utf-8

# In[1]:


from allennlp.modules.elmo import Elmo, batch_to_ids
import sys
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# In[2]:



####read evaluation data
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
                target_ws.append(line[0]+' '+line[1])
                pos_lst.append(int(line[2]))
                sent_w_lst=line[3].split()
                if sent_w_lst[int(line[2])]!=line[0].split('.')[0]:
                    print ('warning: position {2}, target_w:{0}:{1}'.format(line[0],sent_w_lst[int(line[2])],line[2]))
                sents.append(sent_w_lst)
        except UnicodeDecodeError as e:
            print (line)
            print (e)
    return sents,pos_lst,target_ws

###### elmo word representation
def w2elmo_type_batch(w_lst,batchsize):
    '''
    w_lst=['apple','orange']
    '''
    candidates_all=[]
    candi2index={}
    index=0
    for candi in w_lst:
            if candi not in candidates_all:
                candidates_all.append([candi])
                candi2index[candi]=index
                index+=1

    candis2elmo=w2elmo_token_batch(candidates_all,batchsize,[0]*len(candidates_all))
    return candis2elmo,candi2index,candidates_all

def w2elmo_token_batch(sent_lst,batchsize,pos_lst=None):
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
        character_ids=character_ids_all[start:end]
        embeddings = elmo(character_ids)
        
        if pos_lst!=None:
            pos=pos_lst[start:end]
            context_reps.append(embeddings['elmo_representations'][0].detach().numpy()[np.arange(len(sentences)),np.array(pos)])
        else:
            context_reps.append(embeddings['elmo_representations'][0].detach().numpy())

    context_reps=np.concatenate(context_reps,axis=0)
    return context_reps
        
###### candidates ranking and ouput
def candidate_ranking_out(output_f,words2elmo_token,target_ws,target_w2candidates,candi2index,candis2elmo):
      with open (output_f,'w') as f:
        for i in range(len(words2elmo_token)):
            w2elmo_token=words2elmo_token[i]
            target_w=target_ws[i]
            try:
                candidates=target_w2candidates['.'.join(target_w.split()[0].split('.')[:2])]
                candis_cos=[]
                for candi in candidates:
                        candi_type=candis2elmo[candi2index[candi]]
                        cos=cosine_similarity([w2elmo_token],[candi_type])[0][0]
                        candis_cos.append('{0} {1}'.format(candi,cos))

                candis_cos=sorted(candis_cos,key=lambda x:x.split()[1],reverse=True)
                candis_cos='\t'.join(candis_cos)
                out_line='RANKED\t{0}\t{1}\n'.format(target_w,candis_cos)
                f.write(out_line)
            except KeyError as e:
                print (e)
                continue


# In[3]:


if __name__=="__main__":
    
    #elmo parameters
    elmo_options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    elmo_weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

    if sys.argv[0]=='/usr/local/lib/python3.6/site-packages/ipykernel_launcher.py':
        model='elmo'
        data_f='/home/incontext_lexsub/eval/lexsub_en/datasets/lst_all.preprocessed'
        gold_candis_f='/home/incontext_lexsub/eval/lexsub_en/datasets/lst.gold.candidates'
        
        
    else:
        print ('python lexsub_eval.py [model] [data_f] [gold_candis_f]')
        model=sys.argv[1]
        data_f=sys.argv[2]
        gold_candis_f=sys.argv[3]
    
    output_f=data_f.split('/')[-1]+'.ranked'
    
   
    # 1. read in model
    if model=='elmo':
        elmo = Elmo(elmo_options_file, elmo_weight_file, 1, dropout=0)
    else:
        raise NotImplementedError


    # 2. read in gold substitutes
    target_w2candidates=read_gold_candidates(gold_candis_f)
    # 3. evaluate on the data:
    sents,pos_lst,target_ws=read_eval_data(data_f)

    # 3.1. candi2elmo type word representation
    candidates_all=[w for lst in target_w2candidates.values() for w in lst]
    candis2elmo,candi2index,candidates_all=w2elmo_type_batch(candidates_all,1000)


    # 3.2. token level context representations
    words2elmo_token=w2elmo_token_batch(sents,100,pos_lst)

    # 3.3. candidate ranking
    candidate_ranking_out(output_f,words2elmo_token,target_ws,target_w2candidates,candi2index,candis2elmo)
#     with open (output_f,'w') as f:
#         for i in range(len(words2elmo_token)):
#             w2elmo_token=words2elmo_token[i]
#             target_w=target_ws[i]
#             candidates=target_w2candidates[target_w.split()[0]]
#             candis_cos=[]
#             for candi in candidates:
#                 candi_type=candis2elmo[candi2index[candi]]
#                 cos=cosine_similarity([w2elmo_token],[candi_type])[0][0]
#                 candis_cos.append('{0} {1}'.format(candi,cos))
#             candis_cos=sorted(candis_cos,key=lambda x:x.split()[1],reverse=True)
#             candis_cos='\t'.join(candis_cos)
#             out_line='RANKED\t{0}\t{1}\n'.format(target_w,candis_cos)
#             f.write(out_line)

