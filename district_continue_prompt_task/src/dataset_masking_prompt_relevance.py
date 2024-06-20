import json
import os
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from utils import padded_tensor

seed=1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

class CRSDataset(Dataset):
    def __init__(
            self, dataset, split, tokenizer, debug=False,
            max_length=None, entity_max_length=None,
            prompt_tokenizer=None, prompt_max_length=None, bert_tokenizer=None,key_name=None,special_template=None,conti_tokens=None
    ):
        super(CRSDataset, self).__init__()
        self.debug = debug
        self.tokenizer = tokenizer
        self.prompt_tokenizer = prompt_tokenizer
        self.bert_tokenizer = bert_tokenizer
        self.key_name = key_name
        self.special_template=special_template
        self.conti_tokens=conti_tokens

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

        # self.data['all_Discrete_Coherent4context']={}#也就是所有有rec的对话
        # self.prepare_data_all_Discrete_Coherent4context(data_file)
        # self.data['Context_Discrete_Utility4context_short'] = {}
        # self.prepare_context_Discrete_Utility4context_short(data_file)
        #
        # self.data['Context_Discrete_Utility4context_long'] = {}
        # self.prepare_context_Discrete_Utility4context_long(data_file)
        # self.data['origin'] = []
        # self.prepare_data(data_file)
        if split=='train':
            negative_number=3
        else:
            negative_number = 3
        self.data['add_nopair'] = []

        Discrete_4context_prefix = 'discrete_' + self.key_name
        Discrete_4context_short = Discrete_4context_prefix + '4context_short'
        Discrete_4context_long = Discrete_4context_prefix + '4context_long'

        Discrete_4context_short_temp=Discrete_4context_prefix + 'short_temp'
        Discrete_4context_long_temp = Discrete_4context_prefix + 'long_temp'


        self.data[Discrete_4context_short] = {}
        self.prepare_context_Discrete_4context_short(Discrete_4context_short,data_file, negative_number=negative_number)

        self.data[Discrete_4context_long] = {}
        self.prepare_context_Discrete_4context_long(Discrete_4context_long,data_file, negative_number=negative_number)

        self.data[Discrete_4context_short_temp] = []
        tmp_len=len(self.data[Discrete_4context_short])
        for x in range(tmp_len):
            for j in range(negative_number+1):
                self.data[Discrete_4context_short_temp].append(self.data[Discrete_4context_short][x+1][j])

        self.data[Discrete_4context_long_temp] = []
        tmp_len=len(self.data[Discrete_4context_long])
        for x in range(tmp_len):
            for j in range(negative_number+1):
                self.data[Discrete_4context_long_temp].append(self.data[Discrete_4context_long][x+1][j])
        self.data['add_nopair']=[]
        for x in range(len(self.data[Discrete_4context_short])):
            self.data['add_nopair'].append([self.data[Discrete_4context_short_temp][x],self.data[Discrete_4context_long_temp][x]])

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
            for line in tqdm(lines):
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
                        prompt_context+=self.bert_tokenizer.sep_token
                    else:
                        prompt_context += self.prompt_tokenizer.sep_token  # '</s>'，该字符是roberta的结尾符

                if i % 2 == 0:  # i为偶数则是user最近一次的说话，回复则为system；相似的i为奇数则是system最近一次的说话，回复则为user
                    resp = 'System: '
                else:
                    resp = 'User: '  # 特别注意：rep也可能是User的回复
                resp += dialog['resp']

                '''
                template
                '''
                template = "<dialogue_history> is [MASK] to <user_response>"
                base_sentence = template.replace("<dialogue_history>", prompt_context)

                for _ in dialog['rec']:#若resp中有多个推荐的电影，则每一个推荐的电影对应的历史内容相同。
                    impid += 1
                    '''
                    a positive simple:level1
                    '''
                    sentence = base_sentence.replace("<user_response>", resp)
                    self.data['all_Discrete_Coherent4context'][impid]=[]
                    self.data['all_Discrete_Coherent4context'][impid].append({'sentence': sentence, 'target': 1,'impd': impid})

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
    def prepare_context_Discrete_4context_short(self, Discrete_4context_short,data_file, negative_number=2, bert_true=True):
        with open(data_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if self.debug:
                lines = lines[:1024]
            '''
            记录当前对话的id，哪一个对话
            '''
            impid = 0
            current_utt_index = 0
            for line in tqdm(lines):
                dialog = json.loads(line)
                if len(dialog['rec']) == 0:  # 特别关键，这里只保留有推荐的电影对话作为resp，也就是label
                    continue
                prompt_context = ''
                if (len(dialog['context'])-1)% 2 == 0:
                    prompt_context += 'User: '
                else:
                    prompt_context += 'System: '
                prompt_context+=dialog['context'][-1]
                if bert_true == True:
                    #prompt_context += self.bert_tokenizer.sep_token
                    prompt_context += "[ns]"
                else:
                    prompt_context += self.prompt_tokenizer.sep_token  # '</s>'，该字符是roberta的结尾符
                role=''
                if (len(dialog['context'])-1) % 2 == 0:  # i为偶数则是user最近一次的说话，回复则为system；相似的i为奇数则是system最近一次的说话，回复则为user
                    resp = 'System: '
                    role+= 'System: '
                else:
                    resp = 'User: '
                    role+='User: '# 特别注意：rep也可能是User的回复
                dialog['resp']+="[ns]"
                resp += dialog['resp']

                #任务性描述
                template="[TASK] Predicting the correlation between short-term dialogue history and current responses. "
                template=template+"[SEP] The short-term dialogue history <dialogue_history> is [MASK] to the current response [SEP] <user_response> ."
                tmp=template.split(" ")

                #模板
                index_mask=tmp.index("[MASK]")
                tmp[index_mask]=''.join(self.conti_tokens[1])+" "+tmp[index_mask]
                tmp[index_mask] = tmp[index_mask] +" "+''.join(self.conti_tokens[0])
                template = " ".join(tmp)
                #[TASK] Predicting the correlation between short-term dialogue history and current responses. [SEP] the short-term dialogue history <dialogue_history> is [P1][P2][P3][P4][P5][P6][P7][P8][P9][P10][MASK][Q1][Q2][Q3][Q4][Q5][Q6][Q7][Q8][Q9][Q10] to the current response [SEP] <user_response>
                prompt_context_ids = self.bert_tokenizer.encode(prompt_context, add_special_tokens=False)[-self.prompt_max_length:]
                prompt_context = self.bert_tokenizer.decode(prompt_context_ids)
                base_sentence = template.replace("<dialogue_history>", prompt_context)

                for _ in dialog['rec']:  # 若resp中有多个推荐的电影，则每一个推荐的电影对应的历史内容相同。
                    impid += 1
                    '''
                    a positive simple
                    '''
                    resp_ids = self.bert_tokenizer.encode(resp, add_special_tokens=False)[-self.prompt_max_length:]
                    resp = self.bert_tokenizer.decode(resp_ids)
                    sentence = base_sentence.replace("<user_response>", resp)
                    self.data[Discrete_4context_short][impid] = []
                    self.data[Discrete_4context_short][impid].append(
                        {'sentence': sentence, 'target': 1, 'impd': impid})

                    '''n=3, a negative simple'''
                    obs_idxs = []
                    neg_resp = []
                    num_options = len(lines)
                    while len(obs_idxs) < negative_number:
                        obs_i = int(np.random.randint(0, num_options))#102
                        if obs_i != current_utt_index and obs_i not in obs_idxs:
                            resp_ids = self.bert_tokenizer.encode(json.loads(lines[obs_i])['resp']+"[ns]", add_special_tokens=False)[-self.prompt_max_length:]
                            resp = role+self.bert_tokenizer.decode(resp_ids)
                            neg_resp.append(resp)
                            obs_idxs.append(obs_i)
                    for K in neg_resp:
                        sentence = base_sentence.replace("<user_response>", K)
                        self.data[Discrete_4context_short][impid].append(
                            {'sentence': sentence, 'target': 0, 'impd': impid})
                current_utt_index += 1
            print("short:")
            print(self.data[Discrete_4context_short][impid])
    def prepare_context_Discrete_4context_long(self,Discrete_4context_long, data_file, negative_number=2, bert_true=True):
        with open(data_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if self.debug:
                lines = lines[:1024]
            '''
            记录当前对话的id，哪一个对话
            '''
            impid = 0
            current_utt_index = 0
            for line in tqdm(lines):
                dialog = json.loads(line)
                if len(dialog['rec']) == 0:  # 特别关键，这里只保留有推荐的电影对话作为resp，也就是label
                    continue
                prompt_context = ''
                for i, utt in enumerate(dialog['context']):
                    if i % 2 == 0:  # 偶数是user说话(除开0)
                        prompt_context += 'User: '
                    else:  # 奇数是system说话(除开0)，也就是默认都是System说第一句话
                        prompt_context += 'System: '
                    prompt_context += utt
                    if bert_true == True:
                        prompt_context += "[ns]"
                    else:
                        prompt_context += self.prompt_tokenizer.sep_token  # '</s>'，该字符是roberta的结尾符
                role = ''
                if i % 2 == 0:  # i为偶数则是user最近一次的说话，回复则为system；相似的i为奇数则是system最近一次的说话，回复则为user
                    resp = 'System: '
                    role += 'System: '
                else:
                    resp = 'User: '  # 特别注意：rep也可能是User的回复
                    role += 'User: '
                dialog['resp']+="[ns]"
                resp += dialog['resp']

                #任务性描述
                template="[TASK] Predicting the correlation between long-term dialogue history and current responses. "
                template=template+"[SEP] The long-term dialogue history <dialogue_history> is [MASK] to the current response [SEP] <user_response> ."
                tmp=template.split(" ")

                #模板
                index_mask=tmp.index("[MASK]")
                tmp[index_mask]=''.join(self.conti_tokens[0])+" "+tmp[index_mask]
                tmp[index_mask] = tmp[index_mask] +" "+''.join(self.conti_tokens[1])
                template = " ".join(tmp)

                prompt_context_ids = self.bert_tokenizer.encode(prompt_context, add_special_tokens=False)[-self.prompt_max_length:]
                prompt_context = self.bert_tokenizer.decode(prompt_context_ids)
                base_sentence = template.replace("<dialogue_history>", prompt_context)

                for _ in dialog['rec']:  # 若resp中有多个推荐的电影，则每一个推荐的电影对应的历史内容相同。
                    impid += 1
                    '''
                    a positive simple
                    '''
                    resp_ids = self.bert_tokenizer.encode(resp, add_special_tokens=False)[-self.prompt_max_length:]
                    resp = self.bert_tokenizer.decode(resp_ids)
                    sentence = base_sentence.replace("<user_response>", resp)
                    self.data[Discrete_4context_long][impid] = []
                    self.data[Discrete_4context_long][impid].append(
                        {'sentence': sentence, 'target': 1, 'impd': impid})

                    '''n=3, a negative simple'''
                    obs_idxs = []
                    neg_resp = []
                    num_options = len(lines)
                    while len(obs_idxs) < negative_number:
                        obs_i = int(np.random.randint(0, num_options))#63
                        if obs_i != current_utt_index and obs_i not in obs_idxs:
                            resp_ids = self.bert_tokenizer.encode(json.loads(lines[obs_i])['resp']+"[ns]", add_special_tokens=False)[-self.prompt_max_length:]
                            resp = role+self.bert_tokenizer.decode(resp_ids)
                            neg_resp.append(json.loads(lines[obs_i])['resp'])
                            obs_idxs.append(obs_i)
                    for K in neg_resp:
                        sentence = base_sentence.replace("<user_response>", K)
                        self.data[Discrete_4context_long][impid].append(
                            {'sentence': sentence, 'target': 0, 'impd': impid})
                current_utt_index += 1
            print("long:")
            print(self.data[Discrete_4context_long][impid])
    def prepare_context_Discrete_Utility4context_short(self, data_file, negative_number=3, bert_true=True):
        with open(data_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if self.debug:
                lines = lines[:1024]
            '''
            记录当前对话的id，哪一个对话
            '''
            impid = 0
            current_utt_index = 0
            for line in tqdm(lines):
                dialog = json.loads(line)
                if len(dialog['rec']) == 0:  # 特别关键，这里只保留有推荐的电影对话作为resp，也就是label
                    continue
                prompt_context = ''
                u1=''
                u2=''
                if (len(dialog['context'])-1)% 2 == 0:
                    prompt_context += 'User: '
                else:
                    prompt_context += 'System: '
                prompt_context+=dialog['context'][-1]
                if bert_true == True:
                    prompt_context += self.bert_tokenizer.sep_token
                else:
                    prompt_context += self.prompt_tokenizer.sep_token  # '</s>'，该字符是roberta的结尾符

                resp = dialog['resp']

                if (len(dialog['context'])-1) % 2 == 0:  # i为偶数则是user最近一次的说话，回复则为system；相似的i为奇数则是system最近一次的说话，回复则为user
                    u1+= 'System'
                    u2+='User'
                else:
                    u1+='User'
                    u2+= 'System'  # 特别注意：rep也可能是User的回复

                '''
                template
                '''
                template = u1+" replying <user_response> to the "+u2+" is a [MASK] choice According to the most recent dialogue <dialogue_history>"
                prompt_context_ids = self.bert_tokenizer.encode(prompt_context, add_special_tokens=False)[-self.prompt_max_length:]
                prompt_context = self.bert_tokenizer.decode(prompt_context_ids)
                base_sentence = template.replace("<dialogue_history>", prompt_context)

                for _ in dialog['rec']:  # 若resp中有多个推荐的电影，则每一个推荐的电影对应的历史内容相同。
                    impid += 1
                    '''
                    a positive simple
                    '''
                    resp_ids = self.bert_tokenizer.encode(resp, add_special_tokens=False)[-self.prompt_max_length:]
                    resp = self.bert_tokenizer.decode(resp_ids)
                    sentence = base_sentence.replace("<user_response>", resp)
                    self.data['Context_Discrete_Utility4context_short'][impid] = []
                    self.data['Context_Discrete_Utility4context_short'][impid].append(
                        {'sentence': sentence, 'target': 1, 'impd': impid})

                    '''n=3, a negative simple'''
                    obs_idxs = []
                    neg_resp = []
                    num_options = len(lines)
                    while len(obs_idxs) < negative_number:
                        obs_i = int(np.random.randint(0, num_options))
                        if obs_i != current_utt_index and obs_i not in obs_idxs:
                            resp_ids = self.bert_tokenizer.encode(json.loads(lines[obs_i])['resp'], add_special_tokens=False)[-self.prompt_max_length:]
                            resp = self.bert_tokenizer.decode(resp_ids)
                            neg_resp.append(json.loads(lines[obs_i])['resp'])
                            obs_idxs.append(obs_i)
                    for K in neg_resp:
                        sentence = base_sentence.replace("<user_response>", K)
                        self.data['Context_Discrete_Utility4context_short'][impid].append(
                            {'sentence': sentence, 'target': 0, 'impd': impid})

                '''
                data['prompt']:['System',':',word,...word,'</s>','User',':',word,...,word,'</s>','System',....,''</s>','User',....,'</s>'] 用roberta编码
                data['entity']:[历史实体] 
                data['rec']:dialog['resp']中的实体

                值得注意的是： data['context']和data['prompt']是dialog['context']+dialog['resp']的组合，而data['rec']是dialog['resp']中命名实体短语对应的实体id。
                '''
                current_utt_index += 1
    def prepare_context_Discrete_Utility4context_long(self, data_file, negative_number=3, bert_true=True):
        with open(data_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if self.debug:
                lines = lines[:1024]
            '''
            记录当前对话的id，哪一个对话
            '''
            impid = 0
            current_utt_index = 0
            for line in tqdm(lines):
                dialog = json.loads(line)
                if len(dialog['rec']) == 0:  # 特别关键，这里只保留有推荐的电影对话作为resp，也就是label
                    continue
                prompt_context = ''
                u1=''
                u2=''
                for i, utt in enumerate(dialog['context']):
                    if i % 2 == 0:  # 偶数是user说话(除开0)
                        prompt_context += 'User: '
                    else:  # 奇数是system说话(除开0)，也就是默认都是System说第一句话
                        prompt_context += 'System: '
                    prompt_context += utt
                    if bert_true == True:
                        prompt_context += self.bert_tokenizer.sep_token
                    else:
                        prompt_context += self.prompt_tokenizer.sep_token  # '</s>'，该字符是roberta的结尾符

                if i % 2 == 0:  # i为偶数则是user最近一次的说话，回复则为system；相似的i为奇数则是system最近一次的说话，回复则为user
                    u1+= 'System'
                    u2+='User'
                else:
                    u2+= 'System'
                    u1+='User'
                resp = dialog['resp']

                '''
                template
                '''

                template=u1 + " replying <user_response> to the " + u2 + " is a [MASK] choice According to the historical dialogues <dialogue_history>"
                prompt_context_ids = self.bert_tokenizer.encode(prompt_context, add_special_tokens=False)[
                                     -self.prompt_max_length:]
                prompt_context = self.bert_tokenizer.decode(prompt_context_ids)
                base_sentence = template.replace("<dialogue_history>", prompt_context)

                for _ in dialog['rec']:  # 若resp中有多个推荐的电影，则每一个推荐的电影对应的历史内容相同。
                    impid += 1
                    '''
                    a positive simple
                    '''
                    resp_ids = self.bert_tokenizer.encode(resp, add_special_tokens=False)[-self.prompt_max_length:]
                    resp = self.bert_tokenizer.decode(resp_ids)
                    sentence = base_sentence.replace("<user_response>", resp)
                    self.data['Context_Discrete_Utility4context_long'][impid] = []
                    self.data['Context_Discrete_Utility4context_long'][impid].append(
                        {'sentence': sentence, 'target': 1, 'impd': impid})

                    '''n=3, a negative simple'''
                    obs_idxs = []
                    neg_resp = []
                    num_options = len(lines)
                    while len(obs_idxs) < negative_number:
                        obs_i = int(np.random.randint(0, num_options))
                        if obs_i != current_utt_index and obs_i not in obs_idxs:
                            resp_ids = self.bert_tokenizer.encode(json.loads(lines[obs_i])['resp'], add_special_tokens=False)[-self.prompt_max_length:]
                            resp = self.bert_tokenizer.decode(resp_ids)
                            neg_resp.append(json.loads(lines[obs_i])['resp'])
                            obs_idxs.append(obs_i)
                    for K in neg_resp:
                        sentence = base_sentence.replace("<user_response>", K)
                        self.data['Context_Discrete_Utility4context_long'][impid].append(
                            {'sentence': sentence, 'target': 0, 'impd': impid})

                '''
                data['prompt']:['System',':',word,...word,'</s>','User',':',word,...,word,'</s>','System',....,''</s>','User',....,'</s>'] 用roberta编码
                data['entity']:[历史实体] 
                data['rec']:dialog['resp']中的实体

                值得注意的是： data['context']和data['prompt']是dialog['context']+dialog['resp']的组合，而data['rec']是dialog['resp']中命名实体短语对应的实体id。
                '''
                current_utt_index += 1
    def prepare_data_add(self, split):

        if split=='train':
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
                    'Discrete_Relevance4context_short': self.data['Context_Discrete_Relevance4context_short'][impid],  # 1
                    'Discrete_Relevance4context_long': self.data['Context_Discrete_Relevance4context_long'][impid],  # 1
                    # 'Discrete_Utility4context_short': self.data['Context_Discrete_Utility4context_short'][impid],  # 1
                    # 'Discrete_Utility4context_long': self.data['Context_Discrete_Utility4context_long'][impid],  # 1
                }
                self.data['add'].append(all_data)
        else:
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
                    'impid': impid
                }
                self.data['add'].append(all_data)


    def __getitem__(self, ind):
        return self.data['add_nopair'][ind]

    def __len__(self):
        return len(self.data['add_nopair'])


class CRSDataCollator:
    def __init__(
            self, tokenizer, device, pad_entity_id, debug=False,
            max_length=None, entity_max_length=None,
            prompt_tokenizer=None, prompt_max_length=None,
            use_amp=False, bert_tokenizer=None,key_name=None,conti_tokens=None
    ):
        self.debug = debug
        self.device = device
        self.tokenizer = tokenizer
        self.prompt_tokenizer = prompt_tokenizer
        self.bert_tokenizer = bert_tokenizer
        self.key_name = key_name
        self.conti_tokens=conti_tokens

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

    def sentence2token(self,name,sentences,target,impd):
        # prompt={}
        # encode_dict = self.bert_tokenizer.batch_encode_plus(
        #     sentences,
        #     add_special_tokens=True,
        #     padding='max_length',
        #     max_length=500,
        #     truncation=True,
        #     pad_to_max_length=True,
        #     return_attention_mask=True,
        #     return_tensors='pt'
        # )
        # prompt[name+'_input_ids'] = encode_dict['input_ids']
        # prompt[name+'_attention_mask'] = encode_dict['attention_mask']
        # prompt[name+'_target'] = torch.LongTensor(target)
        prompt=[]
        encode_dict = self.bert_tokenizer.batch_encode_plus(
            sentences,
            add_special_tokens=True,
            padding='max_length',
            max_length=500,
            truncation=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        prompt.append(encode_dict['input_ids'])
        prompt.append(encode_dict['attention_mask'])
        prompt.append(torch.LongTensor(target))
        prompt.append(impd)
        return prompt

    def __call__(self, data_batch):
        input_all_batch = {}

        '''
        discrete_relevance_short
        '''
        discrete_sentences_short = []
        discrete_target_short = []
        discrete_impd_short=[]
        '''
        discrete_relevance_long
        '''
        discrete_sentences_long = []
        discrete_target_long = []
        discrete_impd_long = []

        for tmp in data_batch:
            Discrete_4context_prefix = 'discrete_' + self.key_name
            Discrete_4context_short=Discrete_4context_prefix+'4context_short'

            Discrete_4context_long = Discrete_4context_prefix + '4context_long'
            data={}
            data[Discrete_4context_short]=tmp[0]
            data[Discrete_4context_long]=tmp[1]

            discrete_sentences_short.append(data[Discrete_4context_short]['sentence'])
            discrete_target_short.append(data[Discrete_4context_short]['target'])
            discrete_impd_short.append(data[Discrete_4context_short]['impd'])

            discrete_sentences_long.append(data[Discrete_4context_long]['sentence'])
            discrete_target_long.append(data[Discrete_4context_long]['target'])
            discrete_impd_long.append(data[Discrete_4context_long]['impd'])


            # for index,K in enumerate(data['Discrete_Relevance4context_short']):
            #     discrete_relevance_sentences_short.append(K)
            #     if index==0:
            #         discrete_relevance_target_short.append(1)
            #     else:
            #         discrete_relevance_target_short.append(0)
            #
            #
            # '''
            # discrete_relevance_short
            # '''
            # for index,K in enumerate(data['Discrete_Relevance4context_short']):
            #     discrete_relevance_sentences_short.append(K)
            #     if index==0:
            #         discrete_relevance_target_short.append(1)
            #     else:
            #         discrete_relevance_target_short.append(0)
            # '''
            # discrete_relevance_long
            # '''
            # for index,K in enumerate(data['Discrete_Relevance4context_long']):
            #     discrete_relevance_sentences_long.append(K)
            #     if index==0:
            #         discrete_relevance_target_long.append(1)
            #     else:
            #         discrete_relevance_target_long.append(0)

            # '''
            # discrete_utility_short
            # '''
            #
            # for index,K in enumerate(data['Discrete_Utility4context_short']):
            #     discrete_utility_sentences_short.append(K)
            #     if index==0:
            #         discrete_utility_target_short.append(1)
            #     else:
            #         discrete_utility_target_short.append(0)
            #
            # '''
            # discrete_utility_long
            # '''
            # for index,K in enumerate(data['Discrete_Utility4context_long']):
            #     discrete_utility_sentences_long.append(K)
            #     if index==0:
            #         discrete_utility_target_long.append(1)
            #     else:
            #         discrete_utility_target_long.append(0)

        # input_batch = {}
        # '''
        # context_batch：当原始序列长度小于max_length(200时)，则用pad补齐，用50257表示，且对mask有值为1，没有值为0；当原始序列长度超出max_length，则不用pad补齐，mask全为1；注意max_length的设置很重要，往往设置为此前原始序列规定的最大长度即可，也就是len(context_batch)。
        # '''
        # context_batch = self.tokenizer.pad(
        #     context_batch, padding=self.padding, max_length=self.max_length, pad_to_multiple_of=self.pad_to_multiple_of
        # )
        # context_batch['rec_labels'] = label_batch
        # for k, v in context_batch.items():
        #     if not isinstance(v, torch.Tensor):
        #         context_batch[k] = torch.as_tensor(v, device=self.device)
        # input_batch['context'] = context_batch
        # '''
        # prompt_batch：当原始序列长度小于max_length(200)，则用pad补齐，用1表示，且对mask有值为1，没有值为0；当原始序列长度超出max_length，则不用pad补齐，mask全为1；注意max_length的设置很重要，往往设置为此前原始序列规定的最大长度即可，也就是len(prompt_batch)。
        # '''
        # prompt_batch = self.prompt_tokenizer.pad(
        #     prompt_batch, padding=self.padding, max_length=self.prompt_max_length,
        #     pad_to_multiple_of=self.pad_to_multiple_of
        # )
        # for k, v in prompt_batch.items():
        #     if not isinstance(v, torch.Tensor):
        #         prompt_batch[k] = torch.as_tensor(v, device=self.device)
        # input_batch['prompt'] = prompt_batch
        # '''
        # entity_batch：填充为当前序列中的最大长度为16，不足用31161补齐
        # '''
        # entity_batch = padded_tensor(entity_batch, pad_idx=self.pad_entity_id, pad_tail=True, device=self.device)
        # input_batch['entity'] = entity_batch
        #
        # input_all_batch['prompt_item_predict'] = input_batch


        '''
        discrete_relevance_short
        '''
        input_all_batch[Discrete_4context_short]=self.sentence2token(Discrete_4context_prefix,discrete_sentences_short,discrete_target_short,discrete_impd_short)
        '''
        discrete_relevance_long
        '''
        input_all_batch[Discrete_4context_long] = self.sentence2token(Discrete_4context_prefix,discrete_sentences_long,discrete_target_long,discrete_impd_long)

        # '''
        # discrete_utility_short
        # '''
        # input_all_batch['discrete_utility_short'] = self.sentence2token('discrete_utility',discrete_utility_sentences_short,discrete_utility_target_short)
        # '''
        # discrete_utility_long
        # '''
        # input_all_batch['discrete_utility_long'] = self.sentence2token('discrete_utility',discrete_utility_sentences_long,discrete_utility_target_long)

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
