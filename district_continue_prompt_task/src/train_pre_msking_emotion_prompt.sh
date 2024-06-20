--dataset redial  --tokenizer ../model/DialogGPT --model ../model/DialogGPT --text_tokenizer ../model/roberta-base --text_encoder ../model/roberta-base --num_train_epochs 3 --gradient_accumulation_steps 1 --per_device_train_batch_size 16 --per_device_eval_batch_size 32 --num_warmup_steps 3333 --max_length 200 --prompt_max_length 200 --entity_max_length 32 --learning_rate 5e-5 --output_dir ../output/unity_pre/dialogpt_redial_resp_5e-5_repetition --project EmoCRS --name pre_unity --log_all --key_name unity --debug