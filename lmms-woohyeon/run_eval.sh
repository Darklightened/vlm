#!/bin/bash


## generation_type: [recursion, recursion_retain_base, default, base-only, split-only]
# string "recusion" must be in generation_type for recursive generation

## fix_grid: [default, 2x2] - 2x2: force the image to be square
## attention_thresholding_type: [layer_mean]
## remove_unpadding: set True to remove unpadding and set mm_patch_merge_type='spatial' else 'spatial_unpad'
## regenerate_condition: currently "all": always regenerate when type is "recursion"

    
python3 -m accelerate.commands.launch \
    --num_processes=2 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
    --tasks pope,mme \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.6_pope \
    --output_path ./logs/ \
    --generation_type recursion \
    --fix_grid 2x2 \
    --attention_thresholding_type layer_mean_with_top_k \
    --attention_threshold 0.1 \
    --remove_unpadding True \
    --regenerate_condition all \
    --verbosity DEBUG \
    --positional_embedding_type interpolation \
    --wandb_args "project=llava1.6_recursive_eval_woohye0n,entity=VLM_Hallucination_Woohyeon,name=llava-v1.6-7b-168-lin-336-topk0.1"

python3 -m accelerate.commands.launch \
    --num_processes=2 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
    --tasks pope,mme \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.6_pope \
    --output_path ./logs/ \
    --generation_type recursion \
    --fix_grid 2x2 \
    --attention_thresholding_type layer_mean_with_top_k \
    --attention_threshold 0.3 \
    --remove_unpadding True \
    --regenerate_condition all \
    --verbosity DEBUG \
    --positional_embedding_type interpolation \
    --wandb_args "project=llava1.6_recursive_eval_woohye0n,entity=VLM_Hallucination_Woohyeon,name=llava-v1.6-7b-168-lin-336-topk0.3"

python3 -m accelerate.commands.launch \
    --num_processes=2 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
    --tasks pope,mme \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.6_pope \
    --output_path ./logs/ \
    --generation_type recursion \
    --fix_grid 2x2 \
    --attention_thresholding_type layer_mean_with_top_k \
    --attention_threshold 0.3 \
    --remove_unpadding True \
    --regenerate_condition all \
    --verbosity DEBUG \
    --positional_embedding_type interpolation \
    --wandb_args "project=llava1.6_recursive_eval_woohye0n,entity=VLM_Hallucination_Woohyeon,name=llava-v1.6-7b-168-lin-336-topk0.3"

python3 -m accelerate.commands.launch \
    --num_processes=2 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
    --tasks pope,mme \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.6_pope \
    --output_path ./logs/ \
    --generation_type recursion \
    --fix_grid 2x2 \
    --attention_thresholding_type layer_mean_with_top_k \
    --attention_threshold 0.5 \
    --remove_unpadding True \
    --regenerate_condition all \
    --verbosity DEBUG \
    --positional_embedding_type interpolation \
    --wandb_args "project=llava1.6_recursive_eval_woohye0n,entity=VLM_Hallucination_Woohyeon,name=llava-v1.6-7b-168-lin-336-topk0.5"

python3 -m accelerate.commands.launch \
    --num_processes=2 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
    --tasks pope,mme \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.6_pope \
    --output_path ./logs/ \
    --generation_type recursion \
    --fix_grid 2x2 \
    --attention_thresholding_type layer_mean_with_top_k \
    --attention_threshold 0.7 \
    --remove_unpadding True \
    --regenerate_condition all \
    --verbosity DEBUG \
    --positional_embedding_type interpolation \
    --wandb_args "project=llava1.6_recursive_eval_woohye0n,entity=VLM_Hallucination_Woohyeon,name=llava-v1.6-7b-168-lin-336-topk0.7"

python3 -m accelerate.commands.launch \
    --num_processes=2 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
    --tasks pope,mme \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.6_pope \
    --output_path ./logs/ \
    --generation_type recursion \
    --fix_grid 2x2 \
    --attention_thresholding_type layer_mean_with_top_k \
    --attention_threshold 0.9 \
    --remove_unpadding True \
    --regenerate_condition all \
    --verbosity DEBUG \
    --positional_embedding_type interpolation \
    --wandb_args "project=llava1.6_recursive_eval_woohye0n,entity=VLM_Hallucination_Woohyeon,name=llava-v1.6-7b-168-lin-336-topk0.9"

python3 -m accelerate.commands.launch \
    --num_processes=2 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
    --tasks pope,mme \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.6_pope \
    --output_path ./logs/ \
    --generation_type recursion \
    --fix_grid 2x2 \
    --attention_thresholding_type layer_mean \
    --attention_threshold 0.1 \
    --remove_unpadding True \
    --regenerate_condition all \
    --verbosity DEBUG \
    --positional_embedding_type interpolation \
    --wandb_args "project=llava1.6_recursive_eval_woohye0n,entity=VLM_Hallucination_Woohyeon,name=llava-v1.6-7b-168-lin-336-recursion0.1"

python3 -m accelerate.commands.launch \
    --num_processes=2 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
    --tasks pope,mme \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.6_pope \
    --output_path ./logs/ \
    --generation_type recursion \
    --fix_grid 2x2 \
    --attention_thresholding_type layer_mean \
    --attention_threshold 0.3 \
    --remove_unpadding True \
    --regenerate_condition all \
    --verbosity DEBUG \
    --positional_embedding_type interpolation \
    --wandb_args "project=llava1.6_recursive_eval_woohye0n,entity=VLM_Hallucination_Woohyeon,name=llava-v1.6-7b-168-lin-336-recursion0.3"


python3 -m accelerate.commands.launch \
    --num_processes=2 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
    --tasks pope,mme \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.6_pope \
    --output_path ./logs/ \
    --generation_type recursion \
    --fix_grid 2x2 \
    --attention_thresholding_type layer_mean \
    --attention_threshold 0.5 \
    --remove_unpadding True \
    --regenerate_condition all \
    --verbosity DEBUG \
    --positional_embedding_type interpolation \
    --wandb_args "project=llava1.6_recursive_eval_woohye0n,entity=VLM_Hallucination_Woohyeon,name=llava-v1.6-7b-168-lin-336-recursion0.5"
