import json
import os
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from config import gpt2_special_tokens_dict
from utils import padded_tensor


class CRSConvDataset(Dataset):
    def __init__(
        self, dataset, split, tokenizer, debug=False,
        context_max_length=None, resp_max_length=None, entity_max_length=None,
        prompt_tokenizer=None, prompt_max_length=None,bert_tokenizer=None,conti_tokens=None
    ):
        super(CRSConvDataset, self).__init__()
        self.tokenizer = tokenizer
        self.prompt_tokenizer = prompt_tokenizer
        self.debug = debug
        self.bert_tokenizer = bert_tokenizer
        self.split=split
        self.conti_tokens=conti_tokens

        self.context_max_length = context_max_length
        if self.context_max_length is None:
            self.context_max_length = self.tokenizer.model_max_length

        self.resp_max_length = resp_max_length
        if self.resp_max_length is None:
            self.resp_max_length = self.tokenizer.model_max_length
        self.resp_max_length -= 1

        self.entity_max_length = entity_max_length
        if self.entity_max_length is None:
            self.entity_max_length = self.tokenizer.model_max_length

        self.prompt_max_length = prompt_max_length
        if self.prompt_max_length is None:
            self.prompt_max_length = self.prompt_tokenizer.model_max_length
        self.prompt_max_length -= 1

        dataset_dir = os.path.join('data', dataset)
        data_file = os.path.join(dataset_dir, f'{split}_data_processed.jsonl')
        self.data = []
        self.prepare_data(data_file)

    def prepare_data(self, data_file):
        with open(data_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            #if self.debug:
            #if self.split == 'train':
            #lines = lines[:512]

            for line in tqdm(lines):
                dialog = json.loads(line)

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
                    relevance_context_bert_long += "[ns]"#'</s>'，该字符是roberta的结尾符
                # bert_short
                if (len(dialog['context'])-1)% 2 == 0:
                    relevance_context_bert_short += 'User: '
                else:
                    relevance_context_bert_short += 'System: '
                relevance_context_bert_short+=dialog['context'][-1]
                relevance_context_bert_short += "[ns]"#'</s>'，该字符是roberta的结尾符

                u1=''
                u2=''
                role = ''
                if i % 2 == 0:#i为偶数则是user最近一次的说话，回复则为system；相似的i为奇数则是system最近一次的说话，回复则为user
                    u1+= 'System'
                    u2+='User'
                    role += 'System: '
                else:
                    u2+= 'System'
                    u1+='User'
                    role += 'User: '
                if context == '':
                    continue
                '''
                roberta的输入序列:是以'CLS'开头，中间用'</s>'隔断，并用<'/s'>结尾。
                dialogGPT的输入序列是：中间用'<|endoftext|>'隔断，并用'<|endoftext|>'结尾。
                值得注意：
                1、roberta和dialogGPT操作的文本，都手动增加的'User:'和'System:'
                2、回复的操作(target context)只用dialogGPT
                '''
                context_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(context))#DialogGPT
                context_ids = context_ids[-self.context_max_length:]#-200  #只要历史对话中最近的self.context_max_length长度的序列

                prompt_ids = self.prompt_tokenizer.convert_tokens_to_ids(self.prompt_tokenizer.tokenize(prompt_context_robert))
                prompt_ids = prompt_ids[-self.prompt_max_length:]
                prompt_ids.insert(0, self.prompt_tokenizer.cls_token_id)#这是因为：roberta的输入序列 是以[CLS]开头，[/s]结尾

                resp = dialog['resp']
                #resp = 'System: ' + resp
                with self.tokenizer.as_target_tokenizer():
                    resp_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(resp))#DialogGPT
                    resp_ids = resp_ids[:self.resp_max_length]#182 #只要回复开头self.resp_max_length长度的序列
                    resp_ids.append(self.tokenizer.eos_token_id)#50256 #实际就是加的'<|endoftext|>'

                #relevance_short
                template="[TASK] Predicting the correlation between short-term dialogue history and current responses. "
                template=template+"[SEP] The short-term dialogue history <dialogue_history> is [MASK] to the current response [SEP] <user_response> ."
                tmp = template.split(" ")
                index_mask=tmp.index("[MASK]")
                tmp[index_mask]=''.join(self.conti_tokens[1])+" "+tmp[index_mask]
                tmp[index_mask] = tmp[index_mask] +" "+''.join(self.conti_tokens[0])
                template_short_relevance = " ".join(tmp)
                template_short_relevance=template_short_relevance.replace('[MASK]','related')
                relevance_context_ids_short = self.bert_tokenizer.encode(relevance_context_bert_short, add_special_tokens=False)[-self.prompt_max_length:]
                relevance_context_short = self.bert_tokenizer.decode(relevance_context_ids_short)
                base_sentence_short = template_short_relevance.replace("<dialogue_history>", relevance_context_short)

                sentence_relevance_short = base_sentence_short.replace("<user_response>", role+self.bert_tokenizer.decode(self.bert_tokenizer.encode(resp+"[ns]", add_special_tokens=False)[-self.prompt_max_length:]))


                #relevance_long
                template="[TASK] Predicting the correlation between long-term dialogue history and current responses. "
                template=template+"[SEP] The long-term dialogue history <dialogue_history> is [MASK] to the current response [SEP] <user_response> ."
                tmp=template.split(" ")
                index_mask=tmp.index("[MASK]")
                tmp[index_mask]=''.join(self.conti_tokens[1])+" "+tmp[index_mask]
                tmp[index_mask] = tmp[index_mask] +" "+''.join(self.conti_tokens[0])
                template_long_relevance = " ".join(tmp)
                template_long_relevance=template_long_relevance.replace('[MASK]', 'related')
                relevance_context_ids_long = self.bert_tokenizer.encode(relevance_context_bert_long, add_special_tokens=False)[-self.prompt_max_length:]
                relevance_context_long = self.bert_tokenizer.decode(relevance_context_ids_long)
                base_sentence_long = template_long_relevance.replace("<dialogue_history>", relevance_context_long)

                sentence_relevance_long = base_sentence_long.replace("<user_response>", role+self.bert_tokenizer.decode(self.bert_tokenizer.encode(resp+"[ns]", add_special_tokens=False)[-self.prompt_max_length:]))

                #untiy_short
                template="[TASK] Predicting whether the response is the correct choice according short-term dialogue history and current response. "
                template=template+'[SEP] Responding <user_response> is a [MASK] choice [SEP] according to the short-term dialogue <dialogue_history> .'
                tmp = template.split(" ")
                index_mask=tmp.index("Responding")
                tmp[index_mask]=tmp[index_mask]+" "+''.join(self.conti_tokens[0])
                index_mask = tmp.index("[MASK]")
                tmp[index_mask] = ''.join(self.conti_tokens[1])+" "+tmp[index_mask]
                template_short_unity = " ".join(tmp)
                template_short_unity=template_short_unity.replace('[MASK]', 'good')
                unity_context_ids_short = self.bert_tokenizer.encode(relevance_context_bert_short, add_special_tokens=False)[-self.prompt_max_length:]
                unity_context_short = self.bert_tokenizer.decode(unity_context_ids_short)
                base_sentence_short = template_short_unity.replace("<dialogue_history>", unity_context_short)
                sentence_unity_short = base_sentence_short.replace("<user_response>", role+self.bert_tokenizer.decode(
                    self.bert_tokenizer.encode(resp+"[ns]", add_special_tokens=False)[-self.prompt_max_length:]))

                #untiy_long
                template="[TASK] Predicting whether the response is the correct choice according long-term dialogue history and current response. "
                template=template+'[SEP] Responding <user_response> is a [MASK] choice [SEP] according to the long-term dialogue <dialogue_history> .'
                tmp = template.split(" ")
                index_mask=tmp.index("Responding")
                tmp[index_mask]=tmp[index_mask]+" "+''.join(self.conti_tokens[0])
                index_mask = tmp.index("[MASK]")
                tmp[index_mask] = ''.join(self.conti_tokens[1])+" "+tmp[index_mask]
                template_long_unity = " ".join(tmp)
                template_long_unity=template_long_unity.replace('[MASK]', 'good')
                unity_context_ids_long = self.bert_tokenizer.encode(relevance_context_bert_long, add_special_tokens=False)[-self.prompt_max_length:]
                unity_context_long = self.bert_tokenizer.decode(unity_context_ids_long)
                base_sentence_long = template_long_unity.replace("<dialogue_history>", unity_context_long)
                sentence_unity_long = base_sentence_long.replace("<user_response>", role+self.bert_tokenizer.decode(
                    self.bert_tokenizer.encode(resp+"[ns]", add_special_tokens=False)[-self.prompt_max_length:]))

                #action_short
                template="[TASK] Predicting the action based on short-term dialogue history and current responses. "
                template_short_action=template+'[SEP] Does the user accept or reject the response? accept. [SEP] The short-term history: '+''.join(self.conti_tokens[1])+' <dialogue_history> '+'. [SEP] The current response: '+''.join(self.conti_tokens[0])+' <user_response> .'
                action_context_ids_short = self.bert_tokenizer.encode(relevance_context_bert_short, add_special_tokens=False)[-self.prompt_max_length:]
                action_context_short = self.bert_tokenizer.decode(action_context_ids_short)
                base_sentence_short = template_short_action.replace("<dialogue_history>", action_context_short)
                sentence_action_short = base_sentence_short.replace("<user_response>", role+self.bert_tokenizer.decode(
                    self.bert_tokenizer.encode(resp+"[ns]", add_special_tokens=False)[-self.prompt_max_length:]))

                #action_long
                template="[TASK] Predicting the action based on long-term dialogue history and current responses. "
                template_long_action=template+'[SEP] Does the user accept or reject the response? accept. [SEP] The long-term history: '+''.join(self.conti_tokens[1])+' <dialogue_history> '+'. [SEP] The current response: '+''.join(self.conti_tokens[0])+' <user_response> .'
                action_context_ids_long = self.bert_tokenizer.encode(relevance_context_bert_long, add_special_tokens=False)[-self.prompt_max_length:]
                action_context_long = self.bert_tokenizer.decode(action_context_ids_long)
                base_sentence_long = template_long_action.replace("<dialogue_history>", action_context_long)
                sentence_action_long = base_sentence_long.replace("<user_response>", role+self.bert_tokenizer.decode(
                    self.bert_tokenizer.encode(resp+"[ns]", add_special_tokens=False)[-self.prompt_max_length:]))

                data = {
                    'context': context_ids,
                    'resp': resp_ids,
                    'entity': dialog['entity'][-self.entity_max_length:],
                    'prompt': prompt_ids,
                    'discrete_relavance_short': sentence_relevance_short,
                    'discrete_relavance_long': sentence_relevance_long,
                    'discrete_unity_short': sentence_unity_short,
                    'discrete_unity_long': sentence_unity_long,
                    'discrete_action_short':sentence_action_short,
                    'discrete_action_long': sentence_action_long
                }
                self.data.append(data)

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


class CRSConvDataCollator:
    def __init__(
        self, tokenizer, device, pad_entity_id, gen=False, use_amp=False, debug=False, ignore_pad_token_for_loss=True,
        context_max_length=None, resp_max_length=None, entity_max_length=None,
        prompt_tokenizer=None, prompt_max_length=None,bert_tokenizer=None,conti_tokens=None
    ):
        self.tokenizer = tokenizer
        self.prompt_tokenizer = prompt_tokenizer
        self.device = device
        self.use_amp = use_amp
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
        self.gen = gen
        self.debug = debug
        self.bert_tokenizer=bert_tokenizer
        self.conti_tokens=conti_tokens
        '''
        特别关注！！！
        使用debug=True可能会报错
        Padding==True：填充到批处理中最长的序列
        Padding=='max_length'：指定的最大长度(根据'max_length'参数)，如果没有提供该参数，则模型可接受的最大输入长度。
        padding==do_not_pad' '(默认):无填充(即，可以输出带有不同的长度)。
        '''
        self.padding = 'max_length' if self.debug else True#
        self.pad_to_multiple_of = 8 if use_amp else None

        self.context_max_length = context_max_length
        if self.context_max_length is None:
            self.context_max_length = self.tokenizer.model_max_length#1024

        self.resp_max_length = resp_max_length
        if self.resp_max_length is None:
            self.resp_max_length = self.tokenizer.model_max_length#1024

        self.entity_max_length = entity_max_length#32
        if self.entity_max_length is None:
            self.entity_max_length = self.tokenizer.model_max_length

        self.prompt_max_length = prompt_max_length
        if self.prompt_max_length is None:
            self.prompt_max_length = self.prompt_tokenizer.model_max_length

        self.pad_entity_id = pad_entity_id

        self.generate_prompt_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('System:'))

    def __call__(self, data_batch):
        input_all_batch = {}
        context_batch = defaultdict(list)
        prompt_batch = defaultdict(list)
        entity_batch = []
        resp_batch = []
        context_len_batch = []


        '''
        discrete_relevance
        '''

        discrete_prompt={}
        discrete_prompt['discrete_relevance_short'] = []
        discrete_prompt['discrete_relevance_long'] = []
        discrete_prompt['discrete_unity_short'] = []
        discrete_prompt['discrete_unity_long'] = []
        discrete_prompt['discrete_action_short'] = []
        discrete_prompt['discrete_action_long'] = []
        if self.gen:
            self.tokenizer.padding_side = 'left'#填充方向是向从左往右
            for data in data_batch:
                context_ids = data['context']
                context_ids = context_ids[-(self.context_max_length - len(self.generate_prompt_ids)):]#这里注意训练的时候正常，测试/验证的时候在文本末端再加入System(没有进入模型利于展示) : 两个对应的id
                context_len_batch.append(len(context_ids))
                context_ids += self.generate_prompt_ids#
                context_batch['input_ids'].append(context_ids)

                prompt_batch['input_ids'].append(data['prompt'])
                resp_batch.append(data['resp'])
                entity_batch.append(data['entity'])

                #relevance
                template=data['discrete_relavance_short']
                tmp = template.split(" ")
                tmp = tmp[::-1]
                start = tmp.index("[SEP]")
                tmp = tmp[start:][::-1]
                discrete_prompt['discrete_relevance_short'].append(" ".join(tmp))

                template=data['discrete_relavance_long']
                tmp = template.split(" ")
                tmp = tmp[::-1]
                start = tmp.index("[SEP]")
                tmp = tmp[start:][::-1]
                discrete_prompt['discrete_relevance_long'].append(" ".join(tmp))

                #unity
                template=data['discrete_unity_short']
                tmp=template.split(" ")
                start=tmp.index("[SEP]")
                end=tmp.index("good")
                tmp=tmp[:start+3]+tmp[end-2:]
                discrete_prompt['discrete_unity_short'].append(" ".join(tmp))

                template=data['discrete_unity_long']
                tmp=template.split(" ")
                start=tmp.index("[SEP]")
                end=tmp.index("good")
                tmp=tmp[:start+3]+tmp[end-2:]
                discrete_prompt['discrete_unity_long'].append(" ".join(tmp))

                #action
                template = data['discrete_action_short']
                tmp=template.split(" ")
                tmp=tmp[::-1]
                start=tmp.index("[SEP]")
                tmp=tmp[start-4:][::-1]
                discrete_prompt['discrete_action_short'].append(" ".join(tmp))

                template = data['discrete_action_long']
                tmp=template.split(" ")
                tmp=tmp[::-1]
                start=tmp.index("[SEP]")
                tmp=tmp[start-4:][::-1]
                discrete_prompt['discrete_action_long'].append(" ".join(tmp))
            # print("relevance_short："+ str(discrete_prompt['discrete_relevance_short'][-1]))
            # print("relevance_long：" + str(discrete_prompt['discrete_relevance_long'][-1]))
            # print("unity_short：" + str(discrete_prompt['discrete_unity_short'][-1]))
            # print("unity_long：" + str(discrete_prompt['discrete_unity_long'][-1]))
            # print("action_short：" + str(discrete_prompt['discrete_action_short'][-1]))
            # print("action_long：" + str(discrete_prompt['discrete_action_long'][-1]))
        else:
            self.tokenizer.padding_side = 'right'#填充方向是向从右往左

            for data in data_batch:
                input_ids = data['context'] + data['resp']
                input_ids = input_ids[-self.context_max_length:]#383，这其实是多余的因为处理的时候data['context']就一定最多只有200个，data['resp']一定最多只有183个
                context_batch['input_ids'].append(input_ids)

                prompt_batch['input_ids'].append(data['prompt'])#注意这里的data['prompt']内容不包含回复的字符都是历史对话内容
                entity_batch.append(data['entity'])#历史entity
                discrete_prompt['discrete_relevance_short'].append(data['discrete_relavance_short'])
                discrete_prompt['discrete_relevance_long'].append(data['discrete_relavance_long'])
                discrete_prompt['discrete_unity_short'].append(data['discrete_unity_short'])
                discrete_prompt['discrete_unity_long'].append(data['discrete_unity_long'])
                discrete_prompt['discrete_action_short'].append(data['discrete_action_short'])
                discrete_prompt['discrete_action_long'].append(data['discrete_action_long'])

        input_batch = {}
        '''
        当debug=True时候，padding='max_length'会填充到指定参数下max_length；则用pad补齐，用50257表示，且对mask有值为1，没有值为0；注意参数'max_length'的设置很重要，往往设置为此前原始序列规定的最大长度即可。
        当debug=False时候，padding=True 会填充到批处理中最长的序列
        '''
        context_batch = self.tokenizer.pad(
            context_batch, padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of,
            max_length=self.context_max_length
        )
        if not self.gen:
            '''
            resp_batch：resp_batch[0]与context_batch.data['input_ids'][0]，除pad表示以外其余完全一样，前者用-100后者用50257.
            '''
            resp_batch = context_batch['input_ids']
            resp_batch = [[token_id if token_id != self.tokenizer.pad_token_id else -100 for token_id in resp] for resp
                          in resp_batch]
            input_batch['resp'] = torch.as_tensor(resp_batch, device=self.device)
        else:
            input_batch['resp'] = resp_batch
            input_batch['context_len'] = context_len_batch
        for k, v in context_batch.items():
            if not isinstance(v, torch.Tensor):
                context_batch[k] = torch.as_tensor(v, device=self.device)
        input_batch['context'] = context_batch
        '''
        当debug=True时候，padding='max_length'会填充到指定参数下max_length；则用pad补齐，用1表示，且对mask有值为1，没有值为0；当注意max_length的设置很重要，往往设置为此前原始序列规定的最大长度即可。
        当debug=False时候，padding=True 会填充到批处理中最长的序列
        '''
        prompt_batch = self.prompt_tokenizer.pad(
            prompt_batch, padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of,
            max_length=self.prompt_max_length
        )
        for k, v in prompt_batch.items():
            if not isinstance(v, torch.Tensor):
                prompt_batch[k] = torch.as_tensor(v, device=self.device)
        input_batch['prompt'] = prompt_batch
        '''
        当debug=True时候，会填充到指定参数下max_len；则用pad补齐，用31161表示；注意max_len的设置很重要，往往设置为此前原始序列规定的最大长度即可。
        当debug=False时候，会填充到批处理中最长的序列
        '''
        entity_batch = padded_tensor(
            entity_batch, pad_idx=self.pad_entity_id, pad_tail=True, device=self.device,
            use_amp=self.use_amp, debug=self.debug, max_len=self.entity_max_length
        )
        input_batch['entity'] = entity_batch

        input_all_batch['prompt_item_predict'] = input_batch

        '''
        discrete_relevance_short
        '''
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
    from pprint import pprint

    debug = False
    gen = True
    device = torch.device('cpu')
    dataset = 'redial'

    kg = DBpedia(dataset=dataset, debug=debug).get_entity_kg_info()

    model_name_or_path = "../utils/tokenizer/dialogpt-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.add_special_tokens(gpt2_special_tokens_dict)

    prompt_tokenizer = AutoTokenizer.from_pretrained('../utils/tokenizer/roberta-base')

    dataset = CRSConvDataset(dataset, 'test', tokenizer=tokenizer, prompt_tokenizer=prompt_tokenizer, debug=debug)
    for i in range(len(dataset)):
        if i == 3:
            break
        data = dataset[i]
        print(data)
        print(tokenizer.decode(data['context']))
        print(tokenizer.decode(data['resp']))
        print(prompt_tokenizer.decode(data['prompt']))
        print()

    data_collator = CRSConvDataCollator(
        tokenizer=tokenizer, device=device, ignore_pad_token_for_loss=True, pad_entity_id=kg['pad_entity_id'],
        prompt_tokenizer=prompt_tokenizer,
        gen=gen
    )
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=data_collator,
    )

    context_max_len, resp_max_len = 0, 0
    entity_max_len = 0
    for batch in tqdm(dataloader):
        if debug:
            pprint(batch)
            if gen:
                print(tokenizer.decode(batch['context']['input_ids'][0]))
                print(tokenizer.decode(batch['resp'][0]))
            exit()

        context_max_len = max(context_max_len, batch['context']['input_ids'].shape[1])
        if gen:
            for resp in batch['resp']:
                resp_max_len = max(resp_max_len, len(resp))
        entity_max_len = max(entity_max_len, batch['entity'].shape[1])

    print(context_max_len, resp_max_len)
    print(entity_max_len)
    # redial:   (1024, 183, 31), (671, 121, 29), (581, 115, 19) -> (1024, 183, 31)
    # inspired: (1024, 140, 28), (831, 117, 23), (919, 104, 32) -> (1024, 140, 32)
