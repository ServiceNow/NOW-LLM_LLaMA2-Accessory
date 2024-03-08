#!/bin/bash

cd /path/to/LLaMA2-Accessory-nowllm/accessory

pretrained_path=<pretrained_weights>
pretrained_type=consolidated
llama_config=<config_for_pretrained_weights>
tokenizer_path=<tokenizer.model_path>
data_config=<data_cofig>

data_parallel=fsdp
model_parallel=8

exp_name=<exp_name>
echo "exp name: $exp_name"
mkdir -p output/"$exp_name"

command="python main_finetune.py \
--output_dir output/${exp_name} --epochs 3 --warmup_epochs 0.1 \
--batch_size 4 --accum_iter 8 --num_workers 1 \
--max_words 8192 \
--lr 0.00002 --min_lr 0.0 --clip_grad 1 --weight_decay 0.1 \
--data_parallel ${data_parallel} --model_parallel_size ${model_parallel} --checkpointing \
--llama_type mixtral --llama_config ${llama_config} --tokenizer_path ${tokenizer_path} \
--no_visual \
--pretrained_path ${pretrained_path} --pretrained_type=${pretrained_type} \
--data_config ${data_config}"

bcprun -d -p 8 --nnodes 4 -c "${command}" --log output/${exp_name}/logs