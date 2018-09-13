
# coding: utf-8

# In[1]:


from allennlp.modules.elmo import Elmo, batch_to_ids
import sys
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
from jcs.data.pos import to_wordnet_pos,from_lst_pos
from jcs.jcs_io import vec_to_str
from jcs.jcs_io import vec_to_str_generated
from sklearn.preprocessing import normalize


# In[31]:



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
                pos_lst.append(int(line[TARGET_ID]))
                sent_w_lst=line[3].split()
                if sent_w_lst[int(line[TARGET_ID])]!=line[0].split('.')[0]:
                    print ('warning: position {2}, target_w:{0}:{1}'.format(line[0],sent_w_lst[int(line[2])],line[2]))
                sents.append(sent_w_lst)
        except UnicodeDecodeError as e:
            print (line)
            print (e)
    return sents,pos_lst,target_ws


def load_vocab(vocab_f):
    vocab=[]
    with open (vocab_f) as f:
        for line in f:
            word=line.split()[0]
            if generated_word_re.match(word) != None:
                vocab.append(word)
    return vocab
            
###### elmo word representation
def w2elmo_type_batch(w_lst,batchsize):
    '''
    w_lst=['apple','orange']
    '''
    candidates_all=[]
    candi2index={}
    index=0
    for candi in w_lst:
#             if candi not in candidates_all:
                candidates_all.append([candi])
                candi2index[candi]=index
                index+=1
    print ('preprocessed word list.')
    candis2elmo=w2elmo_token_batch(candidates_all,batchsize,[0]*len(candidates_all))
    return candis2elmo,candi2index

def w2elmo_token_batch(sent_lst,batchsize,pos_lst=None):
    '''
    sent_lst=[['there','is','an','apple','.'],['there','is','an','apple','.']]
    pos_lst=[2,3] the position list
    '''
   
    character_ids_all = batch_to_ids(sent_lst)
    print ('done batch_to_ids')
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
        

###### candidates ranking and generation
def add_inference_result(token, weight, filtered_results, candidates_found):
        candidates_found.add(token)
        best_last_weight = filtered_results[token] if token in filtered_results else None
        if best_last_weight == None or weight > best_last_weight:
            filtered_results[token] = weight
def filter_inferred(result_vec, candidates, pos):

    filtered_results = {}
    candidates_found = set()

    if result_vec != None:
        for word, weight in result_vec:
            wn_pos = to_wordnet_pos[from_lst_pos[pos]]
            try:
                lemma = WordNetLemmatizer().lemmatize(word, wn_pos)
            except UnicodeDecodeError as e:
                print (word,e)
                continue
            if lemma in candidates:
                add_inference_result(lemma, weight, filtered_results, candidates_found)
            if lemma.title() in candidates:
                add_inference_result(lemma.title(), weight, filtered_results, candidates_found)
            if word in candidates: # there are some few cases where the candidates are not lemmatized
                add_inference_result(word, weight, filtered_results, candidates_found)                    
            if word.title() in candidates: # there are some few cases where the candidates are not lemmatized
                add_inference_result(word.title(), weight, filtered_results, candidates_found)

    # assign negative weights for candidates with no score
    # they will appear last sorted according to their unigram count        
#        candidates_left = candidates - candidates_found
#        for candidate in candidates_left:            
#            count = self.w2counts[candidate] if candidate in self.w2counts else 1
#            score = -1 - (1.0/count) # between (-1,-2] 
#            filtered_results[candidate] = score   

    return filtered_results

def generate_inferred(result_vec, target_word, target_lemma, pos):
#     print (target_word,target_lemma)
    generated_results = {}
    min_weight = None
    if result_vec is not None:
        for word, weight in result_vec:
            if generated_word_re.match(word) != None: # make sure this is not junk
                wn_pos = to_wordnet_pos[from_lst_pos[pos]]
                lemma = WordNetLemmatizer().lemmatize(word, wn_pos)
                if word != target_word and lemma != target_lemma:
                    if lemma in generated_results:
                        weight = max(weight, generated_results[lemma])
                    generated_results[lemma] = weight
                    if min_weight is None:
                        min_weight = weight
                    else:
                        min_weight = min(min_weight, weight)

    if min_weight is None:
        min_weight = 0.0
    i = 0.0                
    for lemma in default_generated_results:
        if len(generated_results) >= len(default_generated_results):
            break;
        i -= 1.0
        generated_results[lemma] = min_weight + i
    return generated_results


def candidate_ranking_out(data_f,words2elmo_token,target_ws,sents,position_lst,target_w2candidates,w2index,w2elmo,vocab_all):
    output_f_rank=open(data_f.split('/')[-1]+'.'+model+'vocab80000.ranked','w')
    output_f_oot=open(data_f.split('/')[-1]+'.'+model+'vocab80000.oot','w')
    output_f_best=open(data_f.split('/')[-1]+'.'+model+'vocab80000.best','w')
    print ('normalizing vectors')
    w2elmo=normalize(w2elmo)
    words2elmo_token=normalize(words2elmo_token)
    sim_matrix=w2elmo.dot(words2elmo_token.T).T

#     w2elmo = w2elmo / np.sqrt((w2elmo * w2elmo).sum())
#     words2elmo_token=words2elmo_token / np.sqrt((words2elmo_token * words2elmo_token).sum())
    print ('normalizing completed')
    for i in range(len(words2elmo_token)):
        if i%100==0 and i>=100:
            print (i)
        w2elmo_token=words2elmo_token[i]
        target_w_out=target_ws[i]
        target_w=target_w_out.split()[0]
        pos=target_w.split('.')[-1]
        target_w_lemma=target_w.split('.')[0]
        #similarity matrix
#             similarity=(w2elmo.dot(w2elmo_token)+1.0)/2
        similarity=(sim_matrix[i]+1.0)/2
        result_vec = sorted(zip(vocab_all, similarity), reverse=True, key=lambda x: x[1])
        try:
            candidates=target_w2candidates['.'.join(target_w.split('.')[:2])]
        except KeyError as e:
            print ('target w does not occur in gold candidates list: {0}'.format(e))
            continue

        #ranked result
        filtered_results=filter_inferred(result_vec, candidates, pos)
#             candis_cos=sorted(filtered_results.items(),key=lambda x:x[1],reverse=True)
#             candis_cos='\t'.join([res[0]+' '+str(res[1]) for res in candis_cos])
#             out_line='RANKED\t{0}\t{1}\n'.format(target_w,candis_cos)
#             output_f_rank.write(out_line)
        output_f_rank.write("RANKED\t" + target_w_out + "\t" + vec_to_str(filtered_results.items(), len(filtered_results))+"\n")

        #generate result
        generated_results = generate_inferred(result_vec, sents[i][position_lst[i]], target_w_lemma, pos)
        output_f_oot.write(target_w_out+" ::: " + vec_to_str_generated(generated_results.items(), 10)+"\n")
        output_f_best.write(target_w_out + " :: " + vec_to_str_generated(generated_results.items(), 1)+"\n")


    output_f_rank.close()
    output_f_best.close()
    output_f_oot.close()

            #oot result
            
#             out_line='RANKED\t{0}\t{1}\n'.format(target_w,candis_cos)
#             else:

#                 candis_cos=[]
#                 for candi in candidates:
#                         candi_type=candis2elmo[candi2index[candi]]
#                         cos=cosine_similarity([w2elmo_token],[candi_type])[0][0]
#                         candis_cos.append('{0} {1}'.format(candi,cos))

#                 candis_cos=sorted(candis_cos,key=lambda x:x.split()[1],reverse=True)
#                 candis_cos='\t'.join(candis_cos)
            
            


# In[3]:


if __name__=="__main__":
    TARGET_ID=2
    generated_word_re = re.compile('^[a-zA-Z]+$')
    default_generated_results = ['time', 'people', 'information', 'work', 'first', 'like', 'year', 'make', 'day', 'service']


# In[23]:


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



   
# 1. read in model
if model=='elmo':
    elmo = Elmo(elmo_options_file, elmo_weight_file, 1, dropout=0)
else:
    raise NotImplementedError


# 2. read in gold substitutes
target_w2candidates=read_gold_candidates(gold_candis_f)
# 3. evaluate on the data:
sents,pos_lst,target_ws=read_eval_data(data_f)

# 3.1. candi2elmo + vocab type word representation
print ('loading vocab')

print ('elmo word representations')
try:
    candis2elmo=np.load('./vocab2elmo_80000.npy')
    candi2index_tup=np.load('./vocab2index_80000.npy')
    vocab_with_candis=[]
    candi2index={}
    for candi_pair in candi2index_tup:
        vocab_with_candis.append(candi_pair[0])
        candi2index[candi_pair[0]]=int(candi_pair[1])

except FileNotFoundError:
    candidates_all=[w for lst in target_w2candidates.values() for w in lst]
    vocab=load_vocab('../../corpora/1-billion-vocab')[:80000]
    vocab_with_candis=list(set(candidates_all+vocab))
    candis2elmo,candi2index=w2elmo_type_batch(vocab_with_candis,1000)
    np.save('vocab2elmo_80000',candis2elmo)
    np.save('vocab2index_80000',np.array(list(candi2index.items())))


# In[34]:


# 3.2. token level context representations
words2elmo_token=w2elmo_token_batch(sents,100,pos_lst)

   


# In[35]:


# 3.3. candidate ranking

candidate_ranking_out(data_f,words2elmo_token,target_ws,sents,pos_lst,target_w2candidates,candi2index,candis2elmo,vocab_with_candis)

