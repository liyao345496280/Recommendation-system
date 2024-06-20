import os
import torch
from transformers import BertForMaskedLM
import torch.nn as nn


class discrete_Prompt(nn.Module):
    def __init__(self,bert_tokenizer, model_name, answer_ids,mask_id,num_conti1,num_conti2):
        super(discrete_Prompt, self).__init__()
        self.BERT = BertForMaskedLM.from_pretrained("../model/" + model_name)  # model_name
        self.BERT.resize_token_embeddings(len(bert_tokenizer))

        self.answer_ids = answer_ids
        self.mask_token_id = mask_id[0]#<MASK>
        self.loss_func = nn.CrossEntropyLoss()
        self.start_conti_tokens1 = bert_tokenizer.encode('[P1]', add_special_tokens=False)
        self.end_conti_tokens1 = bert_tokenizer.encode('[P'+str(num_conti1)+']', add_special_tokens=False)
        self.start_conti_tokens2 = bert_tokenizer.encode('[Q1]', add_special_tokens=False)
        self.end_conti_tokens2 = bert_tokenizer.encode('[Q'+str(num_conti2)+']', add_special_tokens=False)
        # self.conti_tokens1=[]
        # for i in range(num_conti1):
        #     self.conti_tokens1.append(bert_tokenizer.encode('[P' + str(i + 1) + ']', add_special_tokens=False))
        # self.conti_tokens2 = []
        # for i in range(num_conti2):
        #     self.conti_tokens2.append(bert_tokenizer.encode('[Q' + str(i + 1) + ']', add_special_tokens=False))

    def forward(self, batch_enc, batch_attn, batch_labs=None,have_resp=None):
        #对话模块/预训练融合模块/rec模块
        outputs = self.BERT(input_ids=batch_enc,
                            attention_mask=batch_attn,output_hidden_states=True)
        mask_position = batch_enc.eq(self.mask_token_id)
        P_star_end=[]
        Q_star_end = []
        star_end=[]
        for sentence in batch_enc:
            P_star_end.append([x for x in range(sentence.tolist().index(self.start_conti_tokens1[0]),sentence.tolist().index(self.end_conti_tokens1[0])+1)])
            Q_star_end.append([x for x in range(sentence.tolist().index(self.start_conti_tokens2[0]),
                                                sentence.tolist().index(self.end_conti_tokens2[0]) + 1)])
        for i,x in enumerate(P_star_end):
            star_end.append(P_star_end[i]+Q_star_end[i])



        #微调知识生成器
        if batch_labs!=None:
            out_logits = outputs.logits
            mask_logits = out_logits[mask_position, :].view(out_logits.size(0), -1, out_logits.size(-1))[:, -1, :]
            answer_logits = mask_logits[:, self.answer_ids]
            loss = self.loss_func(answer_logits, batch_labs)
            return loss, answer_logits.softmax(dim=1)
        else:#其他情况都用这个得到P、Q知识特征
            out_hidden_states = outputs.hidden_states[-1]#(64,500,768)

            mask_hidden_states = torch.stack([out_hidden_states[i].squeeze()[star_end[i]] for i in range(len(out_hidden_states))],dim=0)

            #mask_hidden_states = out_hidden_states[mask_position, :].view(out_hidden_states.size(0), -1,
                                                                          #out_hidden_states.size(-1))[:, -1, :]
            return mask_hidden_states#(64,768)-->(64,20,768)


    def save(self, save_path):
        state_dict = {k: v for k, v in self.state_dict().items() if 'edge' not in k}
        torch.save(state_dict, save_path)

    def load(self, load_dir):
        load_path = os.path.join(load_dir, 'model.pt')
        missing_keys, unexpected_keys = self.load_state_dict(
            torch.load(load_path, map_location=torch.device('cpu')), strict=False
        )
        print(missing_keys, unexpected_keys)


