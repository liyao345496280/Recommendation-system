import argparse
import os
import sys
import time

import torch
import transformers
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from loguru import logger
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel,BertTokenizer

from config import gpt2_special_tokens_dict, prompt_special_tokens_dict
from dataset_conv import CRSConvDataCollator, CRSConvDataset
from dataset_dbpedia import DBpedia
from evaluate_conv import ConvEvaluator
from model_gpt2 import PromptGPT2forCRS
from model_masking_prompt import Fusion_KGPrompt
from model_bert_discrete import discrete_Prompt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--output_dir", type=str, help="Where to store the final model.")
    parser.add_argument("--debug", action='store_true', help="Debug mode.")
    # data
    parser.add_argument("--dataset", type=str, required=True, help="A file containing all data.")
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--context_max_length', type=int, help="max length of both encoder and decoder input.")
    parser.add_argument('--resp_max_length', type=int, help="max length of decoder input.")
    parser.add_argument("--entity_max_length", type=int, help="max entity length in dataset.")
    parser.add_argument("--prompt_max_length", type=int)
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--ignore_pad_token_for_loss", action='store_true')
    parser.add_argument("--text_tokenizer", type=str)
    # model
    parser.add_argument("--model", type=str)
    parser.add_argument("--max_gen_len", type=int, default=50)
    parser.add_argument("--text_encoder", type=str)
    parser.add_argument("--prompt_encoder", type=str)
    parser.add_argument("--n_prefix_conv", type=int)
    parser.add_argument("--num_bases", type=int, default=8, help="num_bases in RGCN")
    parser.add_argument("--discrete_prompt_encoder", type=str)
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
    parser.add_argument('--num_warmup_steps', type=int, default=10000)
    parser.add_argument('--fp16', action='store_true', help='use automatic mixed precision to speed up.')
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
    logger.info(accelerator.state)
    logger.info(config)

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
        n_prefix_conv=args.n_prefix_conv
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

    # data
    dataset = CRSConvDataset(
        args.dataset, args.split, tokenizer, debug=args.debug,
        context_max_length=args.context_max_length, resp_max_length=args.resp_max_length,
        entity_max_length=args.entity_max_length,
        prompt_tokenizer=text_tokenizer, prompt_max_length=args.prompt_max_length,bert_tokenizer=bert_tokenizer,conti_tokens=conti_tokens
    )
    data_collator_generator = CRSConvDataCollator(
        tokenizer=tokenizer, device=device, gen=True, use_amp=accelerator.use_fp16, debug=args.debug,
        ignore_pad_token_for_loss=args.ignore_pad_token_for_loss,
        context_max_length=args.context_max_length, resp_max_length=args.resp_max_length,
        entity_max_length=args.entity_max_length, pad_entity_id=kg['pad_entity_id'],
        prompt_tokenizer=text_tokenizer,bert_tokenizer=bert_tokenizer,conti_tokens=conti_tokens
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_eval_batch_size,
        num_workers=args.num_workers,
        collate_fn=data_collator_generator,
    )
    gen_dir = os.path.join('save', args.dataset)
    os.makedirs(gen_dir, exist_ok=True)
    model_name = args.prompt_encoder.split('/')[-2]
    gen_file_path = os.path.join(gen_dir, f'{model_name}_{args.split}.jsonl')
    evaluator = ConvEvaluator(tokenizer=tokenizer, log_file_path=gen_file_path)

    for batch in tqdm(dataloader, disable=not accelerator.is_local_main_process):
        with torch.no_grad():
            token_embeds = text_encoder(**batch['prompt_item_predict']['prompt']).last_hidden_state
            # relevance
            mask_hidden_states_discrete_relevance_short = net4bert_discrete_relevance(
                batch['discrete_relevance_short'][0].to(device), batch['discrete_relevance_short'][1].to(device))
            mask_hidden_states_discrete_relevance_long = net4bert_discrete_relevance(
                batch['discrete_relevance_long'][0].to(device), batch['discrete_relevance_long'][1].to(device))
            # unity
            # unity
            mask_hidden_states_discrete_unity_short = net4bert_discrete_unity(
                batch['discrete_unity_short'][0].to(device), batch['discrete_unity_short'][1].to(device))
            mask_hidden_states_discrete_unity_long = net4bert_discrete_unity(
                batch['discrete_unity_long'][0].to(device), batch['discrete_unity_long'][1].to(device))
            # action
            mask_hidden_states_discrete_action_short = net4bert_discrete_unity(
                batch['discrete_action_short'][0].to(device), batch['discrete_action_short'][1].to(device))
            mask_hidden_states_discrete_action_long = net4bert_discrete_unity(
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
                output_entity=False,
                use_conv_prefix=True,
                mask_prompt_hidden=mask_prompt_hidden
            )
            batch['prompt_item_predict']['context']['prompt_embeds'] = prompt_embeds

            gen_seqs = accelerator.unwrap_model(model).generate(
                **batch['prompt_item_predict']['context'],
                max_new_tokens=args.max_gen_len,
                no_repeat_ngram_size=3,
            )
            gen_resp_ids = []
            for gen_seq, length in zip(gen_seqs, batch['prompt_item_predict']['context_len']):
                gen_seq = [token_id for token_id in gen_seq if token_id != tokenizer.pad_token_id]
                gen_resp_ids.append(gen_seq[length:])
            evaluator.evaluate(gen_resp_ids, batch['prompt_item_predict']['resp'], log=accelerator.is_local_main_process)

    # metric
    accelerator.wait_for_everyone()
    report = evaluator.report()
    test_report = {}
    for k, v in report.items():
        test_report[f'{args.split}/{k}'] = v
    logger.info(test_report)
    if run:
        run.log(test_report)
