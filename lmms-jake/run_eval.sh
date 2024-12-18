#!/bin/bash


## generation_type: [recursion, recursion_retain_base, default, base-only, split-only]
# string "recursion" must be in generation_type for recursive generation

## fix_grid: [default, 2x2] - 2x2: force the image to be square
## attention_thresholding_type: [layer_mean]
## remove_unpadding: set True to remove unpadding and set mm_patch_merge_type='spatial' else 'spatial_unpad'
## regenerate_condition: currently "all": always regenerate when type is "recursion"

python3 -m accelerate.commands.launch \
    --num_processes=1 \
    -m lmms_eval \
    --device cuda:1 \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
    --tasks mme,pope,mmstar \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.6_pope \
    --output_path ./logs/ \
    --generation_type recursion \
    --fix_grid 2x2 \
    --attention_thresholding_type confidence_topk \
    --attention_threshold 0.15 \
    --positional_embedding_type reduced_sin \
    --remove_unpadding True \
    --attn_norm True \
    --stages "-1" "0" "1" \
    --verbosity DEBUG \
    --wandb_args "project=llava1.6_recursive_eval_jake3,entity=VLM_Hallucination_Woohyeon,name=confidence_top15_reduced_sin "

python3 -m accelerate.commands.launch \
    --num_processes=1 \
    -m lmms_eval \
    --device cuda:1\
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
    --tasks mme,pope,mmstar \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.6_pope \
    --output_path ./logs/ \
    --generation_type recursion \
    --fix_grid 2x2 \
    --attention_thresholding_type layer_mean_topk \
    --attention_threshold 0.3 \
    --positional_embedding_type reduced_sin \
    --remove_unpadding True \
    --attn_norm True \
    --stages "-1" "0" "1" \
    --verbosity DEBUG \
    --wandb_args "project=llava1.6_recursive_eval_jake3,entity=VLM_Hallucination_Woohyeon,name=layer_mean_top30_reduced_sin "
