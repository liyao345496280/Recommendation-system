import json
import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from utils import padded_tensor


class CRSDataset(Dataset):
    def __init__(
            self, dataset, split, tokenizer, debug=False,
            max_length=None, entity_max_length=None,
            prompt_tokenizer=None, prompt_max_length=None, bert_tokenizer=None
    ):
        super(CRSDataset, self).__init__()
        self.debug = debug
        self.tokenizer = tokenizer
        self.prompt_tokenizer = prompt_tokenizer
        self.bert_tokenizer = bert_tokenizer

        self.max_length = max_length  # 这里是加上输入序列和生成序列一共的长度，这就是验证/测试的时候的停止条件，按照代码来说，这里写错了 应该为None与max_new_tokens冲突
        if self.max_length is None:
            self.max_length = self.tokenizer.model_max_length

        self.prompt_max_length = prompt_max_length
        if self.prompt_max_length is None:
            self.prompt_max_length = self.prompt_tokenizer.model_max_length
        self.prompt_max_length -= 1

        self.entity_max_length = entity_max_length
        if self.entity_max_length is None:
            self.entity_max_length = self.tokenizer.model_max_length

        dataset_dir = os.path.join('data', dataset)
        data_file = os.path.join(dataset_dir, f'{split}_data_processed.jsonl')
        self.data = dict()

        #self.data['some_Discrete_Coherent4context']={}#也就是所有有rec的对话，但是我不希望再一个对话中重复
        #self.prepare_data_some_Discrete_Coherent4context(data_file)

        self.data['all_Discrete_Coherent4context']={}#也就是所有有rec的对话
        self.prepare_data_all_Discrete_Coherent4context(data_file)

        self.data['origin'] = []
        self.prepare_data(data_file)
        #
        # self.data['Context_Discrete_Relevance4context_short'] = {}
        # self.prepare_context_Discrete_Relevance4context_short(data_file)
        #
        # self.data['Context_Discrete_Relevance4context_long'] = {}
        # self.prepare_context_Discrete_Relevance4context_long(data_file)
        #
        # self.data['Context_Discrete_Utility4context_short'] = {}
        # self.prepare_context_Discrete_Utility4context_short(data_file)
        #
        # self.data['Context_Discrete_Utility4context_long'] = {}
        # self.prepare_context_Discrete_Utility4context_long(data_file)
        #
        self.data['add'] = []
        self.prepare_data_add(data_file,split)
    #一个上下文对应了1个pos，3个neg1，3个neg2
    def prepare_data_some_Discrete_Coherent4context(self, data_file,negative_number=3,bert_true=True):
        tmp=self.data['all_Discrete_Coherent4context']
        self.data['some_Discrete_Coherent4context']

    #一个上下文对应了1个pos，3个neg1，3个neg2
    def prepare_data_all_Discrete_Coherent4context(self, data_file,negative_number=3,bert_true=True):
        with open(data_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if self.debug:
                lines = lines[:1024]
            '''
            记录当前对话的id，哪一个对话
            '''
            impid=0
            current_utt_index=0
            l=0
            for line in tqdm(lines):
                l+=1
                dialog = json.loads(line)
                if len(dialog['rec']) == 0:#特别关键，这里只保留有推荐的电影对话作为resp，也就是label
                    continue
                prompt_context = ''
                for i, utt in enumerate(dialog['context']):
                    if i % 2 == 0:  # 偶数是user说话(除开0)
                        prompt_context += 'User: '
                    else:  # 奇数是system说话(除开0)，也就是默认都是System说第一句话
                        prompt_context += 'System: '
                    prompt_context += utt
                    if bert_true==True:
                        prompt_context+=' '+self.bert_tokenizer.sep_token+' '
                    else:
                        prompt_context += ' '+self.prompt_tokenizer.sep_token+' '  # '</s>'，该字符是roberta的结尾符
                resp = dialog['resp']

                '''
                context
                '''
                prompt_context_ids = self.bert_tokenizer.convert_tokens_to_ids(self.bert_tokenizer.tokenize(prompt_context))
                prompt_context_ids = prompt_context_ids[-self.prompt_max_length:]  # 最多取199
                prompt_context_ids.insert(0, self.bert_tokenizer.cls_token_id)  # 初始位置插入CLS对应id，实际上是<s>字符

                '''
                resp
                '''
                prompt_resp_ids=self.bert_tokenizer.convert_tokens_to_ids(self.bert_tokenizer.tokenize(resp))
                prompt_resp_ids=prompt_resp_ids[-self.prompt_max_length:]
                prompt_resp_ids.insert(0, self.bert_tokenizer.cls_token_id)


                if l==30000:
                    print("")
                '''
                template
                '''
                max_len_his=150
                if len(prompt_context.split(' '))>max_len_his:
                    p_context = ' '.join(prompt_context.split(' ')[-max_len_his:])
                    tmp=p_context.split(' ')
                    if tmp[0]!='User:' or  tmp[0]!='System:':
                        #next_role
                        if 'User:' not in tmp:
                            print(" ")
                        tmp.insert(0, '... ')
                        # if tmp.index('User:') < tmp.index('System:'):
                        #     tmp.insert(0,'System: ... ')
                        # else:
                        #     tmp.insert(0, 'User: ... ')
                    prompt_context=' '.join(tmp)


                template = "<dialogue_history> is [MASK] to <user_response>"
                base_sentence = template.replace("<dialogue_history>", prompt_context)

                max_len_resp = 50
                for _ in dialog['rec']:#若resp中有多个推荐的电影，则每一个推荐的电影对应的历史内容相同。
                    impid += 1
                    '''
                    a positive simple:level1
                    '''

                    if len(resp.split(' ')) > max_len_resp:
                        resp = ' '.join(resp.split(' ')[-max_len_resp:])
                        resp+=' ... '+resp

                    sentence = base_sentence.replace("<user_response>", resp)
                    self.data['all_Discrete_Coherent4context'][impid]=[]
                    self.data['all_Discrete_Coherent4context'][impid].append({'sentence': sentence, 'target': 1,'imp': impid})

                    '''n=3, a negative simple:level2 (有推荐的，但不相关)'''
                    obs_idxs = []
                    neg_resp = []
                    num_options = len(lines)
                    while len(obs_idxs)<negative_number:
                        obs_i = int(np.random.randint(0, num_options))
                        tmp=json.loads(lines[obs_i])
                        tmp_resp=tmp['resp']
                        tmp_rec =tmp['rec']
                        if obs_i !=current_utt_index and obs_i not in obs_idxs and len(tmp_rec)!=0:
                            if len(tmp_resp.split(' ')) > max_len_resp:
                                tmp_resp = ' '.join(tmp_resp.split(' ')[-max_len_resp:])
                                tmp_resp += ' ... ' + tmp_resp
                            neg_resp.append(tmp_resp)
                            obs_idxs.append(obs_i)
                    for K in neg_resp:
                        sentence = base_sentence.replace("<user_response>", K)
                        self.data['all_Discrete_Coherent4context'][impid].append({'sentence': sentence, 'target': 0,'impd': impid})

                    '''n=3, a negative simple:level3 (没有推荐的)'''
                    obs_idxs = []
                    neg_resp = []
                    num_options = len(lines)
                    while len(obs_idxs)<negative_number:
                        obs_i = int(np.random.randint(0, num_options))
                        tmp=json.loads(lines[obs_i])
                        tmp_resp=tmp['resp']
                        tmp_rec =tmp['rec']
                        if obs_i !=current_utt_index and obs_i not in obs_idxs and len(tmp_rec)==0:
                            if len(tmp_resp.split(' ')) > max_len_resp:
                                tmp_resp = ' '.join(tmp_resp.split(' ')[-max_len_resp:])
                                tmp_resp += ' ... ' + tmp_resp
                            neg_resp.append(tmp_resp)
                            obs_idxs.append(obs_i)
                    for K in neg_resp:
                        sentence = base_sentence.replace("<user_response>", K)
                        self.data['all_Discrete_Coherent4context'][impid].append({'sentence': sentence, 'target': 0,'impd': impid})

                '''
                data['prompt']:['System',':',word,...word,'</s>','User',':',word,...,word,'</s>','System',....,''</s>','User',....,'</s>'] 用roberta编码
                data['entity']:[历史实体] 
                data['rec']:dialog['resp']中的实体

                值得注意的是： data['context']和data['prompt']是dialog['context']+dialog['resp']的组合，而data['rec']是dialog['resp']中命名实体短语对应的实体id。
                '''
                current_utt_index+=1
    def prepare_data(self, data_file):
        with open(data_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if self.debug:
                lines = lines[:1024]

            impid = 0
            for line in tqdm(lines):

                dialog = json.loads(line)
                if len(dialog['rec']) == 0:  # 特别关键，这里只保留有推荐的电影对话作为resp，也就是label
                    continue
                # if len(dialog['context']) == 1 and dialog['context'][0] == '':
                #     continue

                context = ''
                prompt_context = ''
                for i, utt in enumerate(dialog['context']):
                    if utt == '':
                        continue
                    if i % 2 == 0:  # 偶数是user说话(除开0)
                        context += 'User: '
                        prompt_context += 'User: '
                    else:  # 奇数是system说话(除开0)，也就是默认都是System说第一句话
                        context += 'System: '
                        prompt_context += 'System: '
                    context += utt
                    context += self.tokenizer.eos_token  # '<|endoftext|>'，该字符是GPT2的结尾符
                    prompt_context += utt
                    prompt_context += self.prompt_tokenizer.sep_token  # '</s>'，该字符是roberta的结尾符
                if i % 2 == 0:  # i为偶数则是user最近一次的说话，回复则为system；相似的i为奇数则是system最近一次的说话，回复则为user
                    resp = 'System: '
                else:
                    resp = 'User: '  # 特别注意：rep也可能是User的回复
                resp += dialog['resp']
                '''
                这里将rsp添加到context后面是因为GPT2是自回归模型
                '''
                context += resp + self.tokenizer.eos_token  # '<|endoftext|>'
                prompt_context += resp + self.prompt_tokenizer.sep_token  # '</s>'

                context_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(context))
                context_ids = context_ids[-self.max_length:]  # 最多取200

                prompt_ids = self.prompt_tokenizer.convert_tokens_to_ids(self.prompt_tokenizer.tokenize(prompt_context))
                prompt_ids = prompt_ids[-self.prompt_max_length:]  # 最多取199
                prompt_ids.insert(0, self.prompt_tokenizer.cls_token_id)  # 初始位置插入CLS对应id，实际上是<s>字符

                '''
                data['context']:['System',':',word,...word,'<|endoftext|>','User',':',word,...,word,'<|endoftext|>','System',....,'<|endoftext|>','User',....,'<|endoftext|>'] 用gpt编码
                data['prompt']:['System',':',word,...word,'</s>','User',':',word,...,word,'</s>','System',....,''</s>','User',....,'</s>'] 用roberta编码
                data['entity']:[历史实体] 
                data['rec']:dialog['resp']中的实体

                值得注意的是： data['context']和data['prompt']是dialog['context']+dialog['resp']的组合，而data['rec']是dialog['resp']中命名实体短语对应的实体id。
                '''

                for rec in dialog['rec']:  # 若resp中有多个推荐的电影，则每一个推荐的电影对应的历史内容相同。
                    impid += 1
                    data = {
                        'context': context_ids,  # 包含了resp
                        'prompt': prompt_ids,  # 包含了resp
                        'entity': dialog['entity'][-self.entity_max_length:],
                        'rec': rec,
                        'impid': impid
                    }
                    self.data['origin'].append(data)
    def prepare_data_add(self, data_file, split):
        with open(data_file, 'r', encoding='utf-8') as f:
            # pos_index=0
            # neg_r_index=0
            for x,data in enumerate(self.data['origin']):
                context = data['context']
                prompt = data['prompt']
                entity = data['entity']
                rec = data['rec']
                impid = data['impid']

                all_data = {
                    'context': context,  # 包含了resp
                    'prompt': prompt,  # 包含了resp
                    'entity': entity,
                    'rec': rec,
                    'impid': impid,
                    'Discrete_Coherent4context': self.data['all_Discrete_Coherent4context'][impid],  # 1
                }
                self.data['add'].append(all_data)
            if split=='train':
                self.data['add'] = [self.data['add'][int(np.random.randint(0, len(self.data['add'])))] for _ in range(int(len(self.data['add'])/10.0))]




    def __getitem__(self, ind):
        return self.data['add'][ind]

    def __len__(self):
        return len(self.data['add'])


class CRSDataCollator:
    def __init__(
            self, tokenizer, device, pad_entity_id, debug=False,
            max_length=None, entity_max_length=None,
            prompt_tokenizer=None, prompt_max_length=None,
            use_amp=False, bert_tokenizer=None
    ):
        self.debug = debug
        self.device = device
        self.tokenizer = tokenizer
        self.prompt_tokenizer = prompt_tokenizer
        self.bert_tokenizer = bert_tokenizer

        self.padding = 'max_length' if self.debug else True
        self.pad_to_multiple_of = 8 if use_amp else None

        self.max_length = max_length
        if self.max_length is None:
            self.max_length = self.tokenizer.model_max_length

        self.prompt_max_length = prompt_max_length
        if self.prompt_max_length is None:
            self.prompt_max_length = self.prompt_tokenizer.model_max_length

        self.pad_entity_id = pad_entity_id
        self.entity_max_length = entity_max_length
        if self.entity_max_length is None:
            self.entity_max_length = self.tokenizer.model_max_length

    def sentence2token(self,name,sentences,target):
        prompt={}
        sentences = [x['sentence'] for x in sentences]
        # mask_index=sentences.index('[MASK]')
        # pre_sentences=sentences[0:mask_index+1][-200:]
        # after_sentnces=sentences[mask_index+1:]
        # sentences=pre_sentences+after_sentnces
        encode_dict = self.bert_tokenizer.batch_encode_plus(
            sentences,
            add_special_tokens=True,
            padding='max_length',
            max_length=300,
            truncation=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        prompt[name+'_input_ids'] = encode_dict['input_ids']
        prompt[name+'_attention_mask'] = encode_dict['attention_mask']
        prompt[name+'_target'] = torch.LongTensor(target)
        return prompt

    def __call__(self, data_batch):
        input_all_batch = {}
        '''
        discrete_coherent
        '''
        discrete_coherent_sentences = []
        discrete_coherent_target = []

        for data in data_batch:

            '''
            discrete_coherent
            '''
            discrete_coherent_sentences.append(data['Discrete_Coherent4context'][0])
            discrete_coherent_target.append(1)
            for K in data['Discrete_Coherent4context'][1:]:
                discrete_coherent_sentences.append(K)
                discrete_coherent_target.append(0)



        '''
        discrete_coherent
        '''
        input_all_batch['discrete_coherent']=self.sentence2token('discrete_coherent',discrete_coherent_sentences,discrete_coherent_target)

        return input_all_batch


if __name__ == '__main__':
    from dataset_dbpedia import DBpedia
    from config import gpt2_special_tokens_dict
    from pprint import pprint

    debug = True
    device = torch.device('cpu')
    dataset = 'inspired'

    kg = DBpedia(dataset, debug=debug).get_entity_kg_info()

    model_name_or_path = '../utils/tokenizer/dialogpt-small'
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.add_special_tokens(gpt2_special_tokens_dict)
    prompt_tokenizer = AutoTokenizer.from_pretrained('../utils/tokenizer/roberta-base')

    dataset = CRSDataset(
        dataset=dataset, split='test', tokenizer=tokenizer, debug=debug,
        prompt_tokenizer=prompt_tokenizer
    )
    for i in range(len(dataset)):
        if i == 3:
            break
        data = dataset[i]
        print(data)
        print(tokenizer.decode(data['context']))
        print(prompt_tokenizer.decode(data['prompt']))
        print()

    data_collator = CRSDataCollator(
        tokenizer=tokenizer, device=device, pad_entity_id=kg['pad_entity_id'],
        prompt_tokenizer=prompt_tokenizer
    )
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=data_collator,
    )

    input_max_len = 0
    entity_max_len = 0
    for batch in tqdm(dataloader):
        if debug:
            pprint(batch)
            print(tokenizer.decode(batch['context']['input_ids'][1]))
            print(prompt_tokenizer.decode(batch['prompt']['input_ids'][1]))
            exit()

        input_max_len = max(input_max_len, batch['context']['input_ids'].shape[1])
        entity_max_len = max(entity_max_len, batch['entity'].shape[1])

    print(input_max_len)
    print(entity_max_len)
    # redial: (1024, 31), (688, 29), (585, 19) -> (1024, 31)
    # inspired: (1024, 30), (902, 23), (945, 32) -> (1024, 32)
