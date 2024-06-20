import json
import os
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from utils import padded_tensor


class CRSRecDataset(Dataset):
    def __init__(
        self, dataset, split, tokenizer, debug=False,
        context_max_length=None, entity_max_length=None,
        prompt_tokenizer=None, prompt_max_length=None,
        use_resp=False,bert_tokenizer=None,conti_tokens=None
    ):
        super(CRSRecDataset, self).__init__()
        self.debug = debug
        self.tokenizer = tokenizer
        self.prompt_tokenizer = prompt_tokenizer
        self.use_resp = use_resp
        self.bert_tokenizer = bert_tokenizer
        self.split=split
        self.conti_tokens=conti_tokens

        self.context_max_length = context_max_length
        if self.context_max_length is None:
            self.context_max_length = self.tokenizer.model_max_length

        self.prompt_max_length = prompt_max_length
        if self.prompt_max_length is None:
            self.prompt_max_length = self.prompt_tokenizer.model_max_length
        self.prompt_max_length -= 1

        self.entity_max_length = entity_max_length
        if self.entity_max_length is None:
            self.entity_max_length = self.tokenizer.model_max_length

        dataset_dir = os.path.join('data', dataset)
        data_file = os.path.join(dataset_dir, f'{split}_data_processed.jsonl')
        self.data = []
        self.prepare_data(data_file)

    def prepare_data(self, data_file):
        with open(data_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if self.debug:
                lines = lines[:1024]

            no_resp=0
            yes_resp=0
            all_resp=0
            for line in tqdm(lines):
                dialog = json.loads(line)
                if len(dialog['rec']) == 0:
                    continue
                if len(dialog['context']) == 1 and dialog['context'][0] == '':
                    continue

                context = ''
                prompt_context_robert = ''
                relevance_context_bert_long=''
                relevance_context_bert_short = ''

                for i, utt in enumerate(dialog['context']):
                    if utt == '':
                        continue
                    if i % 2 == 0:
                        context += 'User: '
                        prompt_context_robert += 'User: '
                        relevance_context_bert_long+='User: '
                    else:
                        context += 'System: '
                        prompt_context_robert += 'System: '
                        relevance_context_bert_long+='System: '
                    context += utt
                    context += self.tokenizer.eos_token
                    #robert
                    prompt_context_robert += utt
                    prompt_context_robert += self.prompt_tokenizer.sep_token#'</s>'，该字符是roberta的结尾符
                    #bert_long
                    relevance_context_bert_long += utt
                    relevance_context_bert_long += "[ns]"
                # bert_short
                if (len(dialog['context'])-1)% 2 == 0:
                    relevance_context_bert_short += 'User: '
                else:
                    relevance_context_bert_short += 'System: '
                relevance_context_bert_short+=dialog['context'][-1]
                relevance_context_bert_short  += "[ns]"

                role = ''
                if context == '':
                    continue
                if self.use_resp:
                    if i % 2 == 0:
                        #resp = 'System: '
                        role += 'System: '
                    else:
                        #resp = 'User: '  # 特别注意：rep也可能是User的回复
                        role += 'User: '
                    dialog['resp'] += '[ns]'
                    resp = dialog['resp']
                    context += role+resp + self.tokenizer.eos_token
                    prompt_context_robert += role+resp + self.prompt_tokenizer.sep_token

                    #relevance_context_bert_long+= resp + self.bert_tokenizer.sep_token

                context_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(context))
                context_ids = context_ids[-self.context_max_length:]

                prompt_ids = self.prompt_tokenizer.convert_tokens_to_ids(self.prompt_tokenizer.tokenize(prompt_context_robert))
                prompt_ids = prompt_ids[-self.prompt_max_length:]
                prompt_ids.insert(0, self.prompt_tokenizer.cls_token_id)

                # relevance_short
                template = "[TASK] Predicting the correlation between short-term dialogue history and current responses. "
                template = template + "[SEP] The short-term dialogue history <dialogue_history> is [MASK] to the current response [SEP] <user_response> ."
                tmp = template.split(" ")
                index_mask = tmp.index("[MASK]")
                tmp[index_mask] = ''.join(self.conti_tokens[1]) + " " + tmp[index_mask]
                tmp[index_mask] = tmp[index_mask] + " " + ''.join(self.conti_tokens[0])
                template_short_relevance = " ".join(tmp)
                template_short_relevance=template_short_relevance.replace('[MASK]', 'related')
                relevance_context_ids_short = self.bert_tokenizer.encode(relevance_context_bert_short,
                                                                         add_special_tokens=False)[
                                              -self.prompt_max_length:]
                relevance_context_short = self.bert_tokenizer.decode(relevance_context_ids_short)
                base_sentence_short = template_short_relevance.replace("<dialogue_history>", relevance_context_short)

                sentence_relevance_short = base_sentence_short.replace("<user_response>", role+self.bert_tokenizer.decode(
                    self.bert_tokenizer.encode(resp+"[ns]", add_special_tokens=False)[-self.prompt_max_length:]))

                # relevance_long
                template = "[TASK] Predicting the correlation between long-term dialogue history and current responses. "
                template = template + "[SEP] The long-term dialogue history <dialogue_history> is [MASK] to the current response [SEP] <user_response> ."
                tmp = template.split(" ")
                index_mask = tmp.index("[MASK]")
                tmp[index_mask] = ''.join(self.conti_tokens[1]) + " " + tmp[index_mask]
                tmp[index_mask] = tmp[index_mask] + " " + ''.join(self.conti_tokens[0])
                template_long_relevance = " ".join(tmp)
                template_long_relevance=template_long_relevance.replace('[MASK]', 'related')
                relevance_context_ids_long = self.bert_tokenizer.encode(relevance_context_bert_long,
                                                                        add_special_tokens=False)[
                                             -self.prompt_max_length:]
                relevance_context_long = self.bert_tokenizer.decode(relevance_context_ids_long)
                base_sentence_long = template_long_relevance.replace("<dialogue_history>", relevance_context_long)

                sentence_relevance_long = base_sentence_long.replace("<user_response>", role+self.bert_tokenizer.decode(
                    self.bert_tokenizer.encode(resp+"[ns]", add_special_tokens=False)[-self.prompt_max_length:]))

                # untiy_short
                template = "[TASK] Predicting whether the response is the correct choice according short-term dialogue history and current response. "
                template = template + '[SEP] Responding <user_response> is a [MASK] choice [SEP] according to the short-term dialogue <dialogue_history> .'
                tmp = template.split(" ")
                index_mask = tmp.index("Responding")
                tmp[index_mask] = tmp[index_mask] + " " + ''.join(self.conti_tokens[0])
                index_mask = tmp.index("[MASK]")
                tmp[index_mask] = ''.join(self.conti_tokens[1]) + " " + tmp[index_mask]
                template_short_unity = " ".join(tmp)
                template_short_unity=template_short_unity.replace('[MASK]', 'good')
                unity_context_ids_short = self.bert_tokenizer.encode(relevance_context_bert_short,
                                                                     add_special_tokens=False)[-self.prompt_max_length:]
                unity_context_short = self.bert_tokenizer.decode(unity_context_ids_short)
                base_sentence_short = template_short_unity.replace("<dialogue_history>", unity_context_short)
                sentence_unity_short = base_sentence_short.replace("<user_response>", role+self.bert_tokenizer.decode(
                    self.bert_tokenizer.encode(dialog['resp']+"[ns]", add_special_tokens=False)[-self.prompt_max_length:]))

                # untiy_long
                template = "[TASK] Predicting whether the response is the correct choice according long-term dialogue history and current response. "
                template = template + '[SEP] Responding <user_response> is a [MASK] choice [SEP] according to the long-term dialogue <dialogue_history> .'
                tmp = template.split(" ")
                index_mask = tmp.index("Responding")
                tmp[index_mask] = tmp[index_mask] + " " + ''.join(self.conti_tokens[0])
                index_mask = tmp.index("[MASK]")
                tmp[index_mask] = ''.join(self.conti_tokens[1]) + " " + tmp[index_mask]
                template_long_unity = " ".join(tmp)
                template_long_unity=template_long_unity.replace('[MASK]', 'good')
                unity_context_ids_long = self.bert_tokenizer.encode(relevance_context_bert_long,
                                                                    add_special_tokens=False)[-self.prompt_max_length:]
                unity_context_long = self.bert_tokenizer.decode(unity_context_ids_long)
                base_sentence_long = template_long_unity.replace("<dialogue_history>", unity_context_long)
                sentence_unity_long = base_sentence_long.replace("<user_response>", role+self.bert_tokenizer.decode(
                    self.bert_tokenizer.encode(dialog['resp']+"[ns]", add_special_tokens=False)[-self.prompt_max_length:]))

                # action_short
                template = "[TASK] Predicting the action based on short-term dialogue history and current responses. "
                template_short_action = template + '[SEP] Does the user accept or reject the response? accept. [SEP] The short-term history: ' + ''.join(
                    self.conti_tokens[1]) + ' <dialogue_history> ' + '. [SEP] The current response: ' + ''.join(
                    self.conti_tokens[0]) + ' <user_response> .'
                action_context_ids_short = self.bert_tokenizer.encode(relevance_context_bert_short,
                                                                      add_special_tokens=False)[
                                           -self.prompt_max_length:]
                action_context_short = self.bert_tokenizer.decode(action_context_ids_short)
                base_sentence_short = template_short_action.replace("<dialogue_history>", action_context_short)
                sentence_action_short = base_sentence_short.replace("<user_response>", role+self.bert_tokenizer.decode(
                    self.bert_tokenizer.encode(dialog['resp']+"[ns]", add_special_tokens=False)[-self.prompt_max_length:]))

                # action_long
                template = "[TASK] Predicting the action based on long-term dialogue history and current responses. "
                template_long_action = template + '[SEP] Does the user accept or reject the response? accept. [SEP] The long-term history: ' + ''.join(
                    self.conti_tokens[1]) + ' <dialogue_history> ' + '. [SEP] The current response: ' + ''.join(
                    self.conti_tokens[0]) + ' <user_response> .'
                action_context_ids_long = self.bert_tokenizer.encode(relevance_context_bert_long,
                                                                     add_special_tokens=False)[-self.prompt_max_length:]
                action_context_long = self.bert_tokenizer.decode(action_context_ids_long)
                base_sentence_long = template_long_action.replace("<dialogue_history>", action_context_long)
                sentence_action_long = base_sentence_long.replace("<user_response>", role+self.bert_tokenizer.decode(
                    self.bert_tokenizer.encode(dialog['resp']+"[ns]", add_special_tokens=False)[-self.prompt_max_length:]))

                '''
                data['context']:['System',':',word,...word,'<|endoftext|>','User',':',word,...,word,'<|endoftext|>','System',....,'<|endoftext|>','User',....,'<|endoftext|>'] 用gpt编码
                data['prompt']:['System',':',word,...word,'</s>','User',':',word,...,word,'</s>','System',....,''</s>','User',....,'</s>'] 用roberta编码
                data['entity']:[历史实体] 
                data['rec']:dialog['resp']中的实体

                值得注意的是： data['context']和data['prompt']是dialog['context']+dialog['resp']的组合，而data['rec']是dialog['resp']中命名实体短语对应的实体id。
                '''

                for rec in dialog['rec']:  # 若resp中有多个推荐的电影，则每一个推荐的电影对应的历史内容相同。
                    data = {
                        'context': context_ids,  # 包含了resp
                        'prompt': prompt_ids,  # 包含了resp
                        'entity': dialog['entity'][-self.entity_max_length:],
                        'rec': rec,
                        'discrete_relavance_short': sentence_relevance_short,
                        'discrete_relavance_long': sentence_relevance_long,
                        'discrete_unity_short': sentence_unity_short,
                        'discrete_unity_long': sentence_unity_long,
                        'discrete_action_short': sentence_action_short,
                        'discrete_action_long': sentence_action_long
                    }
                    self.data.append(data)
                    yes_resp += 1
                # if dialog['resp']=="":
                #     for item in dialog['rec']:
                #         data = {
                #             'context': context_ids,
                #             'entity': dialog['entity'][-self.entity_max_length:],
                #             'rec': item,
                #             'prompt': prompt_ids,
                #             'discrete_relavance_short': "[MASK]",
                #             'discrete_relavance_long': "[MASK]",
                #             'discrete_unity_short': "[MASK]",
                #             'discrete_unity_long': "[MASK]",
                #             'have_resp':False
                #         }
                #         self.data.append(data)
                #         no_resp+=1
                # else:
                #     for item in dialog['rec']:
                #         data = {
                #             'context': context_ids,
                #             'entity': dialog['entity'][-self.entity_max_length:],
                #             'rec': item,
                #             'prompt': prompt_ids,
                #             'discrete_relavance_short': sentence_relevance_short,
                #             'discrete_relavance_long': sentence_relevance_long,
                #             'discrete_unity_short': sentence_unity_short,
                #             'discrete_unity_long': sentence_unity_long,
                #             'have_resp': True
                #         }
                #         self.data.append(data)
                #         yes_resp+=1
    def __getitem__(self, ind):
        return self.data[ind]

    def __len__(self):
        return len(self.data)


class CRSRecDataCollator:
    def __init__(
        self, tokenizer, device, pad_entity_id, use_amp=False, debug=False,
        context_max_length=None, entity_max_length=None,
        prompt_tokenizer=None, prompt_max_length=None,bert_tokenizer=None,conti_tokens=None
    ):
        self.debug = debug
        self.device = device
        self.tokenizer = tokenizer
        self.prompt_tokenizer = prompt_tokenizer
        self.bert_tokenizer=bert_tokenizer
        self.conti_tokens=conti_tokens

        self.padding = 'max_length' if self.debug else True
        self.pad_to_multiple_of = 8 if use_amp else None

        self.context_max_length = context_max_length
        if self.context_max_length is None:
            self.context_max_length = self.tokenizer.model_max_length

        self.prompt_max_length = prompt_max_length
        if self.prompt_max_length is None:
            self.prompt_max_length = self.prompt_tokenizer.model_max_length

        self.pad_entity_id = pad_entity_id
        self.entity_max_length = entity_max_length
        if self.entity_max_length is None:
            self.entity_max_length = self.tokenizer.model_max_length

        # self.rec_prompt_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('Recommend:'))

    def __call__(self, data_batch):

        input_all_batch={}
        context_batch = defaultdict(list)
        prompt_batch = defaultdict(list)
        entity_batch = []
        label_batch = []

        '''
        discrete_relevance
        '''

        discrete_prompt = {}
        discrete_prompt['discrete_relevance_short'] = []
        discrete_prompt['discrete_relevance_long'] = []
        discrete_prompt['discrete_unity_short'] = []
        discrete_prompt['discrete_unity_long'] = []
        discrete_prompt['discrete_action_short'] = []
        discrete_prompt['discrete_action_long'] = []

        count_have_resp=[]
        for data in data_batch:
            # input_ids = data['context'][-(self.context_max_length - len(self.rec_prompt_ids)):] + self.rec_prompt_ids
            input_ids = data['context']
            context_batch['input_ids'].append(input_ids)
            entity_batch.append(data['entity'])
            label_batch.append(data['rec'])
            prompt_batch['input_ids'].append(data['prompt'])

            discrete_prompt['discrete_relevance_short'].append(data['discrete_relavance_short'])
            discrete_prompt['discrete_relevance_long'].append(data['discrete_relavance_long'])

            discrete_prompt['discrete_unity_short'].append(data['discrete_unity_short'])
            discrete_prompt['discrete_unity_long'].append(data['discrete_unity_long'])

            discrete_prompt['discrete_action_short'].append(data['discrete_action_short'])
            discrete_prompt['discrete_action_long'].append(data['discrete_action_long'])





        input_batch = {}

        context_batch = self.tokenizer.pad(
            context_batch, padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of,
            max_length=self.context_max_length
        )
        context_batch['rec_labels'] = label_batch
        for k, v in context_batch.items():
            if not isinstance(v, torch.Tensor):
                context_batch[k] = torch.as_tensor(v, device=self.device)
        input_batch['context'] = context_batch

        prompt_batch = self.prompt_tokenizer.pad(
            prompt_batch, padding=self.padding, max_length=self.prompt_max_length,
            pad_to_multiple_of=self.pad_to_multiple_of
        )
        for k, v in prompt_batch.items():
            if not isinstance(v, torch.Tensor):
                prompt_batch[k] = torch.as_tensor(v, device=self.device)
        input_batch['prompt'] = prompt_batch

        entity_batch = padded_tensor(entity_batch, pad_idx=self.pad_entity_id, pad_tail=True, device=self.device)
        input_batch['entity'] = entity_batch

        input_all_batch['prompt_item_predict'] = input_batch
        '''
        discrete_relevance_short
        '''
        i=0
        for name,sentence in discrete_prompt.items():
            tmp=[]
            encode_dict = self.bert_tokenizer.batch_encode_plus(
                sentence,
                add_special_tokens=True,
                padding='max_length',
                max_length=500,
                truncation=True,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            tmp.append(encode_dict['input_ids'])
            tmp.append(encode_dict['attention_mask'])

            input_all_batch[name]=tmp
        return input_all_batch


if __name__ == '__main__':
    from dataset_dbpedia import DBpedia
    from config import gpt2_special_tokens_dict, prompt_special_tokens_dict
    from pprint import pprint

    debug = True
    device = torch.device('cpu')
    dataset = 'inspired'

    kg = DBpedia(dataset, debug=debug).get_entity_kg_info()

    model_name_or_path = "../utils/tokenizer/dialogpt-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.add_special_tokens(gpt2_special_tokens_dict)
    prompt_tokenizer = AutoTokenizer.from_pretrained('../utils/tokenizer/roberta-base')
    prompt_tokenizer.add_special_tokens(prompt_special_tokens_dict)

    dataset = CRSRecDataset(
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

    data_collator = CRSRecDataCollator(
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
            exit()

        input_max_len = max(input_max_len, batch['context']['input_ids'].shape[1])
        entity_max_len = max(entity_max_len, batch['entity'].shape[1])

    print(input_max_len)
    print(entity_max_len)
    # (767, 26), (645, 29), (528, 16) -> (767, 29)
    # inspired: (993, 25), (749, 20), (749, 31) -> (993, 31)
