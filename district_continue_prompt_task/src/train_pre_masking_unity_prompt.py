import argparse
import shutil
from datetime import datetime
import math
import os
import sys
import time
import utils_prompt
import numpy as np
import torch
import transformers
import wandb
from utils_prompt import evaluate
from accelerate import Accelerator
from accelerate.utils import set_seed
from loguru import logger
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, AutoModel,BertTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP

from dataset_dbpedia import DBpedia
from dataset_masking_prompt_unity import CRSDataset, CRSDataCollator
from evaluate_rec import RecEvaluator
from model_gpt2 import PromptGPT2forCRS
from config import gpt2_special_tokens_dict, prompt_special_tokens_dict
from model_prompt import KGPrompt
from model_bert_discrete import discrete_Prompt
from compute_loss_coherent import compulate

from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--output_dir", type=str, default='save', help="Where to store the final model.")
    parser.add_argument("--debug", action='store_true', help="Debug mode.")
    # data
    parser.add_argument("--dataset", type=str, required=True, help="A file containing all data.")
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument("--max_length", type=int, help="max input length in dataset.")
    parser.add_argument("--prompt_max_length", type=int)
    parser.add_argument("--entity_max_length", type=int, help="max entity length in dataset.")
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--text_tokenizer", type=str)
    # model
    parser.add_argument("--model", type=str, required=True,
                        help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--text_encoder", type=str)
    parser.add_argument("--key_name", type=str)
    parser.add_argument("--num_bases", type=int, default=8, help="num_bases in RGCN.")
    # optim
    parser.add_argument("--num_train_epochs", type=int, default=50, help="Total number of training epochs to perform.")
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
    parser.add_argument('--num_warmup_steps', type=int, default=10000)
    parser.add_argument("--fp16", action='store_true')
    # wandb
    parser.add_argument("--use_wandb", action="store_true", help="whether to use wandb")
    parser.add_argument("--entity", type=str, help="wandb username")
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
    logger.info(config)
    logger.info(accelerator.state)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    # wandb
    if args.use_wandb:
        name = args.name if args.name else local_time
        name += '_' + str(accelerator.process_index)

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
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    kg = DBpedia(dataset=args.dataset, debug=args.debug).get_entity_kg_info()
    #Dialogpt-2
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.add_special_tokens(gpt2_special_tokens_dict)
    model = PromptGPT2forCRS.from_pretrained(args.model)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model = model.to(device)
    #roberta
    text_tokenizer = AutoTokenizer.from_pretrained(args.text_tokenizer)
    text_tokenizer.add_special_tokens(prompt_special_tokens_dict)
    text_encoder = AutoModel.from_pretrained(args.text_encoder)
    text_encoder.resize_token_embeddings(len(text_tokenizer))
    text_encoder = text_encoder.to(device)

    #bert
    model_name = 'bert-base-uncased'
    bert_tokenizer, conti_tokens1, conti_tokens2 = load_tokenizer(model_name, args)
    conti_tokens = [conti_tokens1, conti_tokens2]
    bert_tokenizer.add_special_tokens(prompt_special_tokens_dict)
    '''
    bert:discrete
    '''
    answer = ['bad', 'good']

    special_template=""
    answer_ids = bert_tokenizer.encode(answer, add_special_tokens=False)
    mask_id=bert_tokenizer.encode('[MASK]', add_special_tokens=False)
    net4bert_discrete = discrete_Prompt(bert_tokenizer,model_name, answer_ids,mask_id,args.num_conti1,args.num_conti2).to(device)

    train_modules = [net4bert_discrete]
    for module in train_modules:
        module.requires_grad_(True)


    fix_modules = [model, text_encoder]
    for module in fix_modules:
        module.requires_grad_(False)

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in net4bert_discrete.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': float(0.001)},
        {'params': [p for n, p in net4bert_discrete.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    train_dataset = CRSDataset(
        dataset=args.dataset, split='train', tokenizer=tokenizer, debug=args.debug, max_length=args.max_length,
        prompt_tokenizer=text_tokenizer, prompt_max_length=args.prompt_max_length,
        entity_max_length=args.entity_max_length,bert_tokenizer=bert_tokenizer,key_name=args.key_name,special_template=special_template,conti_tokens=conti_tokens#key_name=='unity'
    )
    valid_dataset = CRSDataset(
        dataset=args.dataset, split='valid', tokenizer=tokenizer, debug=args.debug, max_length=args.max_length,
        prompt_tokenizer=text_tokenizer, prompt_max_length=args.prompt_max_length,
        entity_max_length=args.entity_max_length,bert_tokenizer=bert_tokenizer,key_name=args.key_name,special_template=special_template,conti_tokens=conti_tokens
    )
    test_dataset = CRSDataset(
        dataset=args.dataset, split='test', tokenizer=tokenizer, debug=args.debug, max_length=args.max_length,
        prompt_tokenizer=text_tokenizer, prompt_max_length=args.prompt_max_length,
        entity_max_length=args.entity_max_length,bert_tokenizer=bert_tokenizer,key_name=args.key_name,special_template=special_template,conti_tokens=conti_tokens
    )
    data_collator = CRSDataCollator(
        tokenizer=tokenizer, device=device, pad_entity_id=kg['pad_entity_id'],
        max_length=args.max_length, entity_max_length=args.entity_max_length,
        use_amp=accelerator.use_fp16, debug=args.debug,
        prompt_tokenizer=text_tokenizer, prompt_max_length=args.prompt_max_length,bert_tokenizer=bert_tokenizer,key_name=args.key_name,conti_tokens=conti_tokens
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
    net4bert_discrete, optimizer, train_dataloader, valid_dataloader, test_dataloader = accelerator.prepare(
        net4bert_discrete, optimizer, train_dataloader, valid_dataloader, test_dataloader
    )
    # step, epoch, batch size
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)#每个epoch的step
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch#所有的epoch总的step
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    completed_steps = 0
    # lr_scheduler
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, args.num_warmup_steps, args.max_train_steps)#args.num_warmup_steps，实际就是在0到args.num_warmup_steps不断曾大的step中，学习率也不断增大，最大为1，然后再args.num_warmup_steps到args.max_train_steps中不断减小
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


    save_dir = './model_save/'+args.key_name+'/'+ datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # save model with best metric
    metric, mode = 'loss', -1
    assert mode in (-1, 1)
    if mode == 1:
        best_metric = 0
    else:
        best_metric = float('inf')

    metrics = ['auc', 'mrr', 'ndcg5', 'ndcg10']
    best_val_result = {}
    best_val_epoch = {}
    for m in metrics:
        best_val_result[m] = 0.0
        best_val_epoch[m] = 0

    # train loop
    for epoch in range(args.num_train_epochs):
        train_loss = []
        net4bert_discrete.train()
        #prompt_encoder.train()
        # train
        '''
        batch={dict:3}
        batch['context']
            batch['context'][data]['input_ids']:torch.Size([64, 200])
            batch['context'][data]['attention_mask']:torch.Size([64, 200])
            batch['context'][data]['rec_labels']:torch.Size([64])
        batch['prompt'] 
            batch['prompt'][data]['input_ids']:torch.Size([64, 200])
            batch['prompt'][data]['attention_mask']:torch.Size([64, 200])
        batch['entity']:torch.Size([64, 16])，注意：这里的'entity'可以理解为历史对话中的历史entity(最多为16个)，这里考虑可以与序列推荐结合？
        '''
        mean_loss_short = torch.zeros(2).to(device)
        mean_loss_long = torch.zeros(2).to(device)

        acc_cnt_short = torch.zeros(2).to(device)
        acc_cnt_pos_short = torch.zeros(2).to(device)

        acc_cnt_long = torch.zeros(2).to(device)
        acc_cnt_pos_long = torch.zeros(2).to(device)

        st_tra = time.time()

        if True:
            print('--------------------------------------------------------------------')
            print('start training: ', datetime.now())
            print('Epoch: ', epoch)
            print('lr:', optimizer.state_dict()['param_groups'][0]['lr'])

        for step, batch in enumerate(train_dataloader):
            discrete_short='discrete_'+args.key_name+'4context_short'
            discrete_long = 'discrete_'+args.key_name +'4context_long'
            short_batch_enc, short_batch_attn, short_batch_labs,short_batch_impd = batch[discrete_short]#tensor([1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0])
            long_batch_enc, long_batch_attn, long_batch_labs,long_batch_impd = batch[discrete_long]


            short_batch_enc=short_batch_enc.to(device)
            short_batch_attn=short_batch_attn.to(device)
            short_batch_labs=short_batch_labs.to(device)

            long_batch_enc=long_batch_enc.to(device)
            long_batch_attn=long_batch_attn.to(device)
            long_batch_labs=long_batch_labs.to(device)


            #discrete_coheret(训练不需要target)
            loss_discrete_relevance_long, scores_long=net4bert_discrete(long_batch_enc,long_batch_attn,long_batch_labs)
            loss_discrete_relevance_short, scores_short=net4bert_discrete(short_batch_enc,short_batch_attn,short_batch_labs)
            # loss_discrete_utility_short, scores=net4bert_discrete(batch['discrete_utility_short']['discrete_utility_input_ids'].to(device),batch['discrete_utility_short']['discrete_utility_attention_mask'].to(device),batch['discrete_utility_short']['discrete_utility_target'].to(device))
            # loss_discrete_utility_long, scores=net4bert_discrete(batch['discrete_utility_long']['discrete_utility_input_ids'].to(device),batch['discrete_utility_long']['discrete_utility_attention_mask'].to(device),batch['discrete_utility_long']['discrete_utility_target'].to(device))

            loss=0.5*loss_discrete_relevance_short+loss_discrete_relevance_long
            accelerator.backward(loss)
            train_loss.append(float(loss))

            mean_loss_short[0] += loss_discrete_relevance_short.item()
            mean_loss_short[1] += 1

            mean_loss_long[0] += loss_discrete_relevance_long.item()
            mean_loss_long[1] += 1

            predict_long = torch.argmax(scores_long.detach(), dim=1)
            num_correct_long = (predict_long == long_batch_labs).sum()
            acc_cnt_long[0] += num_correct_long
            acc_cnt_long[1] += predict_long.size(0)
            positive_idx = torch.where(long_batch_labs == 1)[0]
            num_correct_pos = (predict_long[positive_idx] == long_batch_labs[positive_idx]).sum()
            acc_cnt_pos_long[0] += num_correct_pos
            acc_cnt_pos_long[1] += positive_idx.size(0)

            #精确度
            predict_short = torch.argmax(scores_short.detach(), dim=1)
            num_correct_short = (predict_short == short_batch_labs).sum()
            acc_cnt_short[0] += num_correct_short
            acc_cnt_short[1] += predict_short.size(0)
            positive_idx = torch.where(short_batch_labs == 1)[0]
            #召回率
            num_correct_pos = (predict_short[positive_idx] == short_batch_labs[positive_idx]).sum()
            acc_cnt_pos_short[0] += num_correct_pos
            acc_cnt_pos_short[1] += positive_idx.size(0)

            # optim step
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                if args.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(net4bert_discrete.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                progress_bar.update(1)
                completed_steps += 1
                if run:
                    run.log({'loss': np.mean(train_loss) * args.gradient_accumulation_steps})

            if completed_steps >= args.max_train_steps:
                break
        #short
        short_loss_epoch = mean_loss_short[0] / mean_loss_short[1]
        short_acc = acc_cnt_short[0] / acc_cnt_short[1]#预测正确的个数/正负样本总数
        short_acc_pos = acc_cnt_pos_short[0] / acc_cnt_pos_short[1] #recall
        short_pos_ratio = acc_cnt_pos_short[1] / acc_cnt_short[1] #正样本的个数/正负样本总数

        #long
        long_loss_epoch = mean_loss_long[0] / mean_loss_long[1]
        long_acc = acc_cnt_long[0] / acc_cnt_long[1]
        long_acc_pos = acc_cnt_pos_long[0] / acc_cnt_pos_long[1]
        long_pos_ratio = acc_cnt_pos_long[1] / acc_cnt_long[1]

        end_tra = time.time()
        train_spend = (end_tra - st_tra) / 3600
        print("Short: Train Loss: %0.4f" % short_loss_epoch.item())
        print("Short: Train ACC: %0.4f\tACC-Positive: %0.4f\tPositiveRatio: %0.4f\t[%0.2f]" %
              (short_acc.item(), short_acc_pos.item(), short_pos_ratio.item(), train_spend))

        print("Long: Train Loss: %0.4f" % long_loss_epoch.item())
        print("Long: Train ACC: %0.4f\tACC-Positive: %0.4f\tPositiveRatio: %0.4f\t[%0.2f]" %
              (long_acc.item(), long_acc_pos.item(), long_pos_ratio.item(), train_spend))
        file = save_dir + '/Epoch-' + str(epoch) + '.pt'
        print('save file', file)
        net4bert_discrete.save(file)

        # #################################  val  ###################################
        # valid
        st_val = time.time()
        val_scores_short = []
        val_scores_long = []

        acc_cnt_short = torch.zeros(2).to(device)
        acc_cnt_pos_short = torch.zeros(2).to(device)

        acc_cnt_long = torch.zeros(2).to(device)
        acc_cnt_pos_long = torch.zeros(2).to(device)


        acc_cnt = torch.zeros(2).to(device)
        acc_cnt_pos = torch.zeros(2).to(device)

        labels_short = []
        labels_long=[]

        val_scores_long=[]
        val_scores_short=[]

        short_imp_ids = []
        long_imp_ids = []

        valid_loss = []
        net4bert_discrete.eval()
        for batch in tqdm(valid_dataloader):
            with torch.no_grad():
                discrete_short = 'discrete_' + args.key_name + '4context_short'
                discrete_long = 'discrete_' + args.key_name + '4context_long'
                short_batch_enc, short_batch_attn, short_batch_labs,short_batch_impd = batch[discrete_short]
                long_batch_enc, long_batch_attn, long_batch_labs,long_batch_impd = batch[discrete_long]

                labels_short = labels_short + short_batch_labs.cpu().numpy().tolist()
                labels_long = labels_long + long_batch_labs.cpu().numpy().tolist()

                short_imp_ids = short_imp_ids + short_batch_impd
                long_imp_ids = long_imp_ids + long_batch_impd

                short_batch_enc = short_batch_enc.to(device)
                short_batch_attn = short_batch_attn.to(device)
                short_batch_labs = short_batch_labs.to(device)

                long_batch_enc = long_batch_enc.to(device)
                long_batch_attn = long_batch_attn.to(device)
                long_batch_labs = long_batch_labs.to(device)

                # discrete_coheret(训练不需要target)
                loss_discrete_relevance_long, scores_long = net4bert_discrete(long_batch_enc, long_batch_attn,
                                                                                     long_batch_labs)
                loss_discrete_relevance_short, scores_short = net4bert_discrete(short_batch_enc,
                                                                                       short_batch_attn,
                                                                                       short_batch_labs)
                loss = loss_discrete_relevance_short + 0.5 * loss_discrete_relevance_long

                valid_loss.append(float(loss))

                ranking_scores_short = scores_short[:, 1].detach()
                val_scores_short.append(ranking_scores_short)

                ranking_scores_long = scores_long[:, 1].detach()
                val_scores_long.append(ranking_scores_long)

                predict_long = torch.argmax(scores_long.detach(), dim=1)
                num_correct_long = (predict_long == long_batch_labs).sum()
                acc_cnt_long[0] += num_correct_long
                acc_cnt_long[1] += predict_long.size(0)
                positive_idx = torch.where(long_batch_labs == 1)[0]
                num_correct_pos = (predict_long[positive_idx] == long_batch_labs[positive_idx]).sum()
                acc_cnt_pos_long[0] += num_correct_pos
                acc_cnt_pos_long[1] += positive_idx.size(0)

                predict_short = torch.argmax(scores_short.detach(), dim=1)
                num_correct_short = (predict_short == short_batch_labs).sum()
                acc_cnt_short[0] += num_correct_short
                acc_cnt_short[1] += predict_short.size(0)
                positive_idx = torch.where(short_batch_labs == 1)[0]
                num_correct_pos = (predict_short[positive_idx] == short_batch_labs[positive_idx]).sum()
                acc_cnt_pos_short[0] += num_correct_pos
                acc_cnt_pos_short[1] += positive_idx.size(0)

        # short
        short_acc = acc_cnt_short[0] / acc_cnt_short[1]
        short_acc_pos = acc_cnt_pos_short[0] / acc_cnt_pos_short[1]
        short_pos_ratio = acc_cnt_pos_short[1] / acc_cnt_short[1]

        # long
        long_acc = acc_cnt_long[0] / acc_cnt_long[1]
        long_acc_pos = acc_cnt_pos_long[0] / acc_cnt_pos_long[1]
        long_pos_ratio = acc_cnt_pos_long[1] / acc_cnt_long[1]
        print("-----------------------------------------------")
        print("Short: Test ACC: %0.4f\tACC-Positive: %0.4f\tPositiveRatio: %0.4f\t[%0.2f]" %
              (short_acc.item(), short_acc_pos.item(), short_pos_ratio.item(), train_spend))

        print("Long: Test ACC: %0.4f\tACC-Positive: %0.4f\tPositiveRatio: %0.4f\t[%0.2f]" %
              (long_acc.item(), long_acc_pos.item(), long_pos_ratio.item(), train_spend))
        print("-----------------------------------------------")


        val_scores_short = torch.cat(val_scores_short, dim=0)
        val_scores_list_short = [val_scores_short]
        val_labels_short = torch.IntTensor(labels_short).to(device)
        val_labels_list_short=[val_labels_short]
        val_impids_short = torch.IntTensor(short_imp_ids).to(device)
        val_impids_list_short=[val_impids_short]

        val_scores_long = torch.cat(val_scores_long, dim=0)
        val_scores_list_long = [val_scores_long]
        val_labels_long = torch.IntTensor(labels_long).to(device)
        val_labels_list_long=[val_labels_long]
        val_impids_long = torch.IntTensor(long_imp_ids).to(device)
        val_impids_list_long=[val_impids_long]


        predicts, truths=utils_prompt.impressions(val_scores_list_long,val_impids_list_long,val_labels_list_long)

        auc_val, mrr_val, ndcg5_val, ndcg10_val = evaluate(predicts, truths)
        if auc_val > best_val_result['auc']:
            best_val_result['auc'] = auc_val
            best_val_epoch['auc'] = epoch
        if mrr_val > best_val_result['mrr']:
            best_val_result['mrr'] = mrr_val
            best_val_epoch['mrr'] = epoch
        if ndcg5_val > best_val_result['ndcg5']:
            best_val_result['ndcg5'] = ndcg5_val
            best_val_epoch['ndcg5'] = epoch
        if ndcg10_val > best_val_result['ndcg10']:
            best_val_result['ndcg10'] = ndcg10_val
            best_val_epoch['ndcg10'] = epoch
        end_val = time.time()
        val_spend = (end_val - st_val) / 3600
        rank=0
        if rank == 0:
            print("Validate: AUC: %0.4f\tMRR: %0.4f\tnDCG@5: %0.4f\tnDCG@10: %0.4f\t[Val-Time: %0.2f]" %
                    (auc_val, mrr_val, ndcg5_val, ndcg10_val, val_spend))
            print('Best Result: AUC: %0.4f \tMRR: %0.4f \tNDCG@5: %0.4f \t NDCG@10: %0.4f' %
                    (best_val_result['auc'], best_val_result['mrr'], best_val_result['ndcg5'],
                    best_val_result['ndcg10']))
            print('Best Epoch: AUC: %d \tMRR: %d \tNDCG@5: %d \t NDCG@10: %d' %
                    (best_val_epoch['auc'], best_val_epoch['mrr'], best_val_epoch['ndcg5'], best_val_epoch['ndcg10']))
        if rank == 0:
            best_epochs = [best_val_epoch['auc'], best_val_epoch['mrr'], best_val_epoch['ndcg5'],
                           best_val_epoch['ndcg10']]
            best_epoch = max(set(best_epochs), key=best_epochs.count)
            old_file = save_dir + '/Epoch-' + str(best_epoch) + '.pt'
            if not os.path.exists('./temp/'+args.key_name):
                os.makedirs('./temp/'+args.key_name)
            copy_file = './temp/'+args.key_name + '/BestModel.pt'
            shutil.copy(old_file, copy_file)
            print('Copy ' + old_file + ' >>> ' + copy_file)

        # metric
        valid_report = {}
        valid_report['valid/loss'] = np.mean(valid_loss)
        valid_report['epoch'] = epoch
        logger.info(f'{valid_report}')
        if run:
            run.log(valid_report)
        evaluator.reset_metric()


    final_dir = os.path.join(args.output_dir, 'final')
    os.makedirs(final_dir, exist_ok=True)
    save_path = os.path.join(final_dir, 'model.pt')
    net4bert_discrete.save(save_path)
    logger.info(f'save final model')
