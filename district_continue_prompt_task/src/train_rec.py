import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import transformers
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from loguru import logger
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, AutoModel,BertTokenizer


from config import gpt2_special_tokens_dict, prompt_special_tokens_dict
from dataset_dbpedia import DBpedia
from dataset_rec import CRSRecDataset, CRSRecDataCollator
from evaluate_rec import RecEvaluator
from model_gpt2 import PromptGPT2forCRS
from model_masking_prompt import Fusion_KGPrompt
from model_bert_discrete import discrete_Prompt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--output_dir", type=str, default='save', help="Where to store the final model.")
    parser.add_argument("--debug", action='store_true', help="Debug mode.")
    # data
    parser.add_argument("--dataset", type=str, required=True, help="A file containing all data.")
    parser.add_argument("--shot", type=float, default=1)
    parser.add_argument("--use_resp", action="store_true")
    parser.add_argument("--context_max_length", type=int, help="max input length in dataset.")
    parser.add_argument("--prompt_max_length", type=int)
    parser.add_argument("--entity_max_length", type=int, help="max entity length in dataset.")
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--text_tokenizer", type=str)
    # model
    parser.add_argument("--model", type=str, required=True,
                        help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--text_encoder", type=str)
    parser.add_argument("--num_bases", type=int, default=8, help="num_bases in RGCN.")
    parser.add_argument("--n_prefix_rec", type=int)
    parser.add_argument("--prompt_encoder", type=str)
    parser.add_argument("--discrete_prompt_encoder_relevance", type=str)
    parser.add_argument("--discrete_prompt_encoder_unity", type=str)
    parser.add_argument("--discrete_prompt_encoder_action", type=str)
    # optim
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Total number of training epochs to perform.")
    parser.add_argument("--max_train_steps", type=int, default=None,
                        help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4,
                        help="Batch size (per device) for the evaluation dataloader.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay to use.")
    parser.add_argument('--max_grad_norm', type=float)
    parser.add_argument('--num_warmup_steps', type=int)
    parser.add_argument('--fp16', action='store_true')
    # wandb
    parser.add_argument("--use_wandb", action="store_true", help="whether to use wandb")
    parser.add_argument("--entity", type=str, help="wandb username",default='wangyulin')
    parser.add_argument("--project", type=str, help="wandb exp project")
    parser.add_argument("--name", type=str, help="wandb exp name")
    parser.add_argument("--log_all", action="store_true", help="log in all processes, otherwise only in rank0")

    parser.add_argument('--num_conti1', default=3, type=int, help='number of continuous tokens')
    parser.add_argument('--num_conti2', default=3, type=int, help='number of continuous tokens')

    args = parser.parse_args()
    return args

def load_tokenizer(model_name, args):
    tokenizer = BertTokenizer.from_pretrained("../model/"+model_name)
    conti_tokens1 = []
    for i in range(args.num_conti1):
        conti_tokens1.append('[P' + str(i + 1) + ']')
    conti_tokens2 = []
    for i in range(args.num_conti2):
        conti_tokens2.append('[Q' + str(i + 1) + ']')

    new_tokens = ['[NS]','[TASK]']
    tokenizer.add_tokens(new_tokens)

    conti_tokens = conti_tokens1 + conti_tokens2
    tokenizer.add_tokens(conti_tokens)

    new_vocab_size = len(tokenizer)
    args.vocab_size = new_vocab_size

    return tokenizer, conti_tokens1, conti_tokens2

if __name__ == '__main__':
    args = parse_args()
    config = vars(args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us.
    accelerator = Accelerator(device_placement=False, fp16=args.fp16)
    device = accelerator.device

    # Make one log on every process with the configuration for debugging.
    local_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    logger.remove()
    logger.add(sys.stderr, level='DEBUG' if accelerator.is_local_main_process else 'ERROR')
    logger.add(f'log/{local_time}.log', level='DEBUG' if accelerator.is_local_main_process else 'ERROR')
    logger.info(accelerator.state)
    logger.info(config)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
    # wandb
    if args.use_wandb:
        name = args.name if args.name else local_time
        #name += '_' + str(accelerator.process_index)
        name += '_' + 'usepre_userelvanceandunityandaction_usepreprompt(final_loss_infer)'

        if args.log_all:
            group = args.name if args.name else 'DDP_' + local_time
            run = wandb.init(entity=args.entity, project=args.project, group=group, config=config, name=name)
        else:
            if accelerator.is_local_main_process:
                run = wandb.init(entity=args.entity, project=args.project, config=config, name=name)
            else:
                run = None
    else:
        run = None

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    kg = DBpedia(dataset=args.dataset, debug=args.debug).get_entity_kg_info()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.add_special_tokens(gpt2_special_tokens_dict)
    model = PromptGPT2forCRS.from_pretrained(args.model)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model = model.to(device)

    text_tokenizer = AutoTokenizer.from_pretrained(args.text_tokenizer)
    text_tokenizer.add_special_tokens(prompt_special_tokens_dict)
    text_encoder = AutoModel.from_pretrained(args.text_encoder)
    text_encoder.resize_token_embeddings(len(text_tokenizer))
    text_encoder = text_encoder.to(device)

    prompt_encoder = Fusion_KGPrompt(
        model.config.n_embd, text_encoder.config.hidden_size, model.config.n_head, model.config.n_layer, 2,
        n_entity=kg['num_entities'], num_relations=kg['num_relations'], num_bases=args.num_bases,
        edge_index=kg['edge_index'], edge_type=kg['edge_type'],
        n_prefix_rec=args.n_prefix_rec
    )
    if args.prompt_encoder is not None:
        prompt_encoder.load(args.prompt_encoder)
    prompt_encoder = prompt_encoder.to(device)

    #bert
    model_name = 'bert-base-uncased'
    bert_tokenizer, conti_tokens1, conti_tokens2 = load_tokenizer(model_name, args)
    conti_tokens = [conti_tokens1, conti_tokens2]
    bert_tokenizer.add_special_tokens(prompt_special_tokens_dict)

    '''
    bert:discrete_relevance
    '''
    answer = ['unrelated', 'related']
    answer_ids = bert_tokenizer.encode(answer, add_special_tokens=False)
    mask_id=bert_tokenizer.encode('[MASK]', add_special_tokens=False)
    net4bert_discrete_relevance = discrete_Prompt(bert_tokenizer,model_name, answer_ids,mask_id,args.num_conti1,args.num_conti2)
    if args.discrete_prompt_encoder_relevance is not None:
        net4bert_discrete_relevance.load(args.discrete_prompt_encoder_relevance)
    net4bert_discrete_relevance = net4bert_discrete_relevance.to(device)

    '''
    bert:discrete_unity
    '''
    answer = ['bad', 'good']
    answer_ids = bert_tokenizer.encode(answer, add_special_tokens=False)
    mask_id=bert_tokenizer.encode('[MASK]', add_special_tokens=False)
    net4bert_discrete_unity = discrete_Prompt(bert_tokenizer,model_name, answer_ids,mask_id,args.num_conti1,args.num_conti2)
    if args.discrete_prompt_encoder_unity is not None:
        net4bert_discrete_unity.load(args.discrete_prompt_encoder_unity)
    net4bert_discrete_unity = net4bert_discrete_unity.to(device)

    '''
    bert:discrete_action
    '''
    answer = ['reject', 'accept']
    answer_ids = bert_tokenizer.encode(answer, add_special_tokens=False)
    mask_id=bert_tokenizer.encode('[MASK]', add_special_tokens=False)
    net4bert_discrete_action = discrete_Prompt(bert_tokenizer,model_name, answer_ids,mask_id,args.num_conti1,args.num_conti2)
    if args.discrete_prompt_encoder_action is not None:
        net4bert_discrete_action.load(args.discrete_prompt_encoder_action)
    net4bert_discrete_action = net4bert_discrete_action.to(device)

    fix_modules = [model, text_encoder,net4bert_discrete_action,net4bert_discrete_unity,net4bert_discrete_relevance]
    for module in fix_modules:
        module.requires_grad_(False)

    # optim & amp
    modules = [prompt_encoder]
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for model in modules for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for model in modules for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    '''
    预处理的每一个样本的结果：
    data[0]={'context': [11964, 25, 18435, 50256, 12982, 25, 17207, 0, 50256, 11964, 25, 1867, 12121, 286, 6918, 466, 345, 588, 30, 50256, 12982, 25, 1312, 588, 8258, 6918, 13, 703, 546, 345, 50256],
             'entity': [], 
             'rec': 22029,
             'prompt': [0, 36383, 35, 20920, 2, 44518, 35, 17232, 328, 2, 36383, 35, 653, 11581, 9, 4133, 109, 47, 101, 116, 2, 44518, 35, 939, 101, 6269, 4133, 4, 141, 59, 47, 2]}
    data[0]['context']:是dialogGPT编码得到的序列，其中人工增加了'User:'、'System:'，并且默认每一句话以'<|endoftext|>'隔开，并以词作为结尾符。
    data[0]['entity']:是历史中提到的实体序列
    data[0]['rec']:dialog['resp']中的实体；注意这里的'resp'并不是原始数据的'resp'，而是对话生成模块生成好的，其中带了'<movie>'字符。
    data[0]['prompt']:是roberta编码得到的序列，其中人工增加了'User:'、'System:'，并且默认每一句话以'</s>'隔开，并以词作为结尾符；并以'CLS'作为整个序列的开始符。
    '''
    # data
    train_dataset = CRSRecDataset(
        dataset=args.dataset, split='train', debug=args.debug,
        tokenizer=tokenizer, context_max_length=args.context_max_length, use_resp=args.use_resp,
        prompt_tokenizer=text_tokenizer, prompt_max_length=args.prompt_max_length,
        entity_max_length=args.entity_max_length,bert_tokenizer=bert_tokenizer,conti_tokens=conti_tokens#roberta，200
    )
    shot_len = int(len(train_dataset) * args.shot)
    train_dataset = random_split(train_dataset, [shot_len, len(train_dataset) - shot_len])[0]
    assert len(train_dataset) == shot_len
    valid_dataset = CRSRecDataset(
        dataset=args.dataset, split='valid', debug=args.debug,
        tokenizer=tokenizer, context_max_length=args.context_max_length, use_resp=args.use_resp,
        prompt_tokenizer=text_tokenizer, prompt_max_length=args.prompt_max_length,
        entity_max_length=args.entity_max_length,bert_tokenizer=bert_tokenizer,conti_tokens=conti_tokens#roberta，200
    )
    test_dataset = CRSRecDataset(
        dataset=args.dataset, split='test', debug=args.debug,
        tokenizer=tokenizer, context_max_length=args.context_max_length, use_resp=args.use_resp,
        prompt_tokenizer=text_tokenizer, prompt_max_length=args.prompt_max_length,
        entity_max_length=args.entity_max_length,bert_tokenizer=bert_tokenizer,conti_tokens=conti_tokens#roberta，200
    )
    data_collator = CRSRecDataCollator(
        tokenizer=tokenizer, device=device, debug=args.debug,
        context_max_length=args.context_max_length, entity_max_length=args.entity_max_length,
        pad_entity_id=kg['pad_entity_id'],
        prompt_tokenizer=text_tokenizer, prompt_max_length=args.prompt_max_length,bert_tokenizer=bert_tokenizer,conti_tokens=conti_tokens#roberta，200
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        collate_fn=data_collator,
        shuffle=True
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.per_device_eval_batch_size,
        collate_fn=data_collator,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.per_device_eval_batch_size,
        collate_fn=data_collator,
    )
    evaluator = RecEvaluator()
    prompt_encoder, optimizer, train_dataloader, valid_dataloader, test_dataloader = accelerator.prepare(
        prompt_encoder, optimizer, train_dataloader, valid_dataloader, test_dataloader
    )
    # step, epoch, batch size
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    completed_steps = 0
    # lr_scheduler
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, args.num_warmup_steps, args.max_train_steps)
    lr_scheduler = accelerator.prepare(lr_scheduler)
    # training info
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)

    # save model with best metric
    metric, mode = 'loss', -1
    assert mode in (-1, 1)
    if mode == 1:
        best_metric = 0
    else:
        best_metric = float('inf')
    best_metric_dir = os.path.join(args.output_dir, 'best')
    os.makedirs(best_metric_dir, exist_ok=True)

    # train loop
    for epoch in range(args.num_train_epochs):
        train_loss = []
        prompt_encoder.train()
        for step, batch in enumerate(train_dataloader):
            with torch.no_grad():
                token_embeds = text_encoder(**batch['prompt_item_predict']['prompt']).last_hidden_state
                #relevance
                mask_hidden_states_discrete_relevance_short = net4bert_discrete_relevance(
                    batch['discrete_relevance_short'][0].to(device), batch['discrete_relevance_short'][1].to(device))
                mask_hidden_states_discrete_relevance_long = net4bert_discrete_relevance(
                    batch['discrete_relevance_long'][0].to(device), batch['discrete_relevance_long'][1].to(device))
                #unity
                mask_hidden_states_discrete_unity_short = net4bert_discrete_unity(
                    batch['discrete_unity_short'][0].to(device), batch['discrete_unity_short'][1].to(device))
                mask_hidden_states_discrete_unity_long = net4bert_discrete_unity(
                    batch['discrete_unity_long'][0].to(device), batch['discrete_unity_long'][1].to(device))
                #action
                mask_hidden_states_discrete_action_short = net4bert_discrete_action(
                    batch['discrete_action_short'][0].to(device), batch['discrete_action_short'][1].to(device))
                mask_hidden_states_discrete_action_long = net4bert_discrete_action(
                    batch['discrete_action_long'][0].to(device), batch['discrete_action_long'][1].to(device))
                mask_hidden_states = [mask_hidden_states_discrete_relevance_long,
                                      mask_hidden_states_discrete_unity_long,
                                      mask_hidden_states_discrete_action_long]
                # mask_hidden_states = [mask_hidden_states_discrete_relevance_short.unsqueeze(1),
                #                       mask_hidden_states_discrete_relevance_long.unsqueeze(1),
                #                       mask_hidden_states_discrete_unity_short.unsqueeze(1),
                #                       mask_hidden_states_discrete_unity_long.unsqueeze(1)]

            mask_prompt_hidden = torch.cat(mask_hidden_states, dim=1)
            prompt_embeds = prompt_encoder(
                entity_ids=batch['prompt_item_predict']['entity'],
                token_embeds=token_embeds,
                output_entity=True,
                use_rec_prefix=True,
                mask_prompt_hidden = mask_prompt_hidden,
            )
            batch['prompt_item_predict']['context']['prompt_embeds'] = prompt_embeds
            batch['prompt_item_predict']['context']['entity_embeds'] = prompt_encoder.get_entity_embeds()

            loss = model(**batch['prompt_item_predict']['context'], rec=True).rec_loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            train_loss.append(float(loss))

            # optim step
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                if args.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(prompt_encoder.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                progress_bar.update(1)
                completed_steps += 1
                if run:
                    run.log({'loss': np.mean(train_loss) * args.gradient_accumulation_steps})

            if completed_steps >= args.max_train_steps:
                break

        # metric
        train_loss = np.mean(train_loss) * args.gradient_accumulation_steps
        logger.info(f'epoch {epoch} train loss {train_loss}')

        del train_loss, batch

        # valid
        valid_loss = []
        prompt_encoder.eval()
        for batch in tqdm(valid_dataloader):
            with torch.no_grad():
                token_embeds = text_encoder(**batch['prompt_item_predict']['prompt']).last_hidden_state
                #relevance
                mask_hidden_states_discrete_relevance_short = net4bert_discrete_relevance(
                    batch['discrete_relevance_short'][0].to(device), batch['discrete_relevance_short'][1].to(device))
                mask_hidden_states_discrete_relevance_long = net4bert_discrete_relevance(
                    batch['discrete_relevance_long'][0].to(device), batch['discrete_relevance_long'][1].to(device))
                #unity
                mask_hidden_states_discrete_unity_short = net4bert_discrete_unity(
                    batch['discrete_unity_short'][0].to(device), batch['discrete_unity_short'][1].to(device))
                mask_hidden_states_discrete_unity_long = net4bert_discrete_unity(
                    batch['discrete_unity_long'][0].to(device), batch['discrete_unity_long'][1].to(device))
                #action
                mask_hidden_states_discrete_action_short = net4bert_discrete_action(
                    batch['discrete_action_short'][0].to(device), batch['discrete_action_short'][1].to(device))
                mask_hidden_states_discrete_action_long = net4bert_discrete_action(
                    batch['discrete_action_long'][0].to(device), batch['discrete_action_long'][1].to(device))
                mask_hidden_states = [mask_hidden_states_discrete_relevance_long,
                                      mask_hidden_states_discrete_unity_long,
                                      mask_hidden_states_discrete_action_long]
                # mask_hidden_states=[mask_hidden_states_discrete_relevance_short.unsqueeze(1),mask_hidden_states_discrete_relevance_long.unsqueeze(1),mask_hidden_states_discrete_unity_short.unsqueeze(1),mask_hidden_states_discrete_unity_long.unsqueeze(1)]
                mask_prompt_hidden = torch.cat(mask_hidden_states, dim=1)
                prompt_embeds = prompt_encoder(
                    entity_ids=batch['prompt_item_predict']['entity'],
                    token_embeds=token_embeds,
                    output_entity=True,
                    use_rec_prefix=True,
                    mask_prompt_hidden=mask_prompt_hidden,
                )
                batch['prompt_item_predict']['context']['prompt_embeds'] = prompt_embeds
                batch['prompt_item_predict']['context']['entity_embeds'] = prompt_encoder.get_entity_embeds()

                outputs = model(**batch['prompt_item_predict']['context'], rec=True)
                valid_loss.append(float(outputs.rec_loss))
                logits = outputs.rec_logits[:, kg['item_ids']]
                ranks = torch.topk(logits, k=50, dim=-1).indices.tolist()
                ranks = [[kg['item_ids'][rank] for rank in batch_rank] for batch_rank in ranks]
                labels = batch['prompt_item_predict']['context']['rec_labels']
                evaluator.evaluate(ranks, labels)

        # metric
        report = accelerator.gather(evaluator.report())
        for k, v in report.items():
            report[k] = v.sum().item()

        valid_report = {}
        for k, v in report.items():
            if k != 'count':
                valid_report[f'valid/{k}'] = v / report['count']
        valid_report['valid/loss'] = np.mean(valid_loss)
        valid_report['epoch'] = epoch
        logger.info(f'{valid_report}')
        if run:
            run.log(valid_report)
        evaluator.reset_metric()

        if valid_report[f'valid/{metric}'] * mode > best_metric * mode:
            prompt_encoder.save(best_metric_dir)
            best_metric = valid_report[f'valid/{metric}']
            logger.info(f'new best model with {metric}')

        # test
        test_loss = []
        prompt_encoder.eval()
        for batch in tqdm(test_dataloader):
            with torch.no_grad():
                token_embeds = text_encoder(**batch['prompt_item_predict']['prompt']).last_hidden_state
                #relevance
                mask_hidden_states_discrete_relevance_short = net4bert_discrete_relevance(
                    batch['discrete_relevance_short'][0].to(device), batch['discrete_relevance_short'][1].to(device))
                mask_hidden_states_discrete_relevance_long = net4bert_discrete_relevance(
                    batch['discrete_relevance_long'][0].to(device), batch['discrete_relevance_long'][1].to(device))
                #unity
                mask_hidden_states_discrete_unity_short = net4bert_discrete_unity(
                    batch['discrete_unity_short'][0].to(device), batch['discrete_unity_short'][1].to(device))
                mask_hidden_states_discrete_unity_long = net4bert_discrete_unity(
                    batch['discrete_unity_long'][0].to(device), batch['discrete_unity_long'][1].to(device))
                #action
                mask_hidden_states_discrete_action_short = net4bert_discrete_action(
                    batch['discrete_action_short'][0].to(device), batch['discrete_action_short'][1].to(device))
                mask_hidden_states_discrete_action_long = net4bert_discrete_action(
                    batch['discrete_action_long'][0].to(device), batch['discrete_action_long'][1].to(device))
                mask_hidden_states = [mask_hidden_states_discrete_relevance_long,
                                      mask_hidden_states_discrete_unity_long,
                                      mask_hidden_states_discrete_action_long]
                # mask_hidden_states=[mask_hidden_states_discrete_relevance_short.unsqueeze(1),mask_hidden_states_discrete_relevance_long.unsqueeze(1),mask_hidden_states_discrete_unity_short.unsqueeze(1),mask_hidden_states_discrete_unity_long.unsqueeze(1)]
                mask_prompt_hidden = torch.cat(mask_hidden_states, dim=1)
                prompt_embeds = prompt_encoder(
                    entity_ids=batch['prompt_item_predict']['entity'],
                    token_embeds=token_embeds,
                    output_entity=True,
                    use_rec_prefix=True,
                    mask_prompt_hidden=mask_prompt_hidden,
                )
                batch['prompt_item_predict']['context']['prompt_embeds'] = prompt_embeds
                batch['prompt_item_predict']['context']['entity_embeds'] = prompt_encoder.get_entity_embeds()

                outputs = model(**batch['prompt_item_predict']['context'], rec=True)
                test_loss.append(float(outputs.rec_loss))
                logits = outputs.rec_logits[:, kg['item_ids']]
                ranks = torch.topk(logits, k=50, dim=-1).indices.tolist()
                ranks = [[kg['item_ids'][rank] for rank in batch_rank] for batch_rank in ranks]
                labels = batch['prompt_item_predict']['context']['rec_labels']
                evaluator.evaluate(ranks, labels)

        # metric
        report = accelerator.gather(evaluator.report())
        for k, v in report.items():
            report[k] = v.sum().item()

        test_report = {}
        for k, v in report.items():
            if k != 'count':
                test_report[f'test/{k}'] = v / report['count']
        test_report['test/loss'] = np.mean(test_loss)
        test_report['epoch'] = epoch
        logger.info(f'{test_report}')
        if run:
            run.log(test_report)
        evaluator.reset_metric()

    final_dir = os.path.join(args.output_dir, 'final')
    prompt_encoder.save(final_dir)
    logger.info(f'save final model')
