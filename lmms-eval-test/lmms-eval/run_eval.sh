#!/bin/bash


## generation_type: [recursion, recursion_retain_base, default, base-only, split-only]
# string "recursion" must be in generation_type for recursive generation

## fix_grid: [default, 2x2] - 2x2: force the image to be square
## attention_thresholding_type: [layer_mean, layer_mean_with_top_k]
## remove_unpadding: set True to remove unpadding and set mm_patch_merge_type='spatial' else 'spatial_unpad'
## regenerate_condition: currently "all": always regenerate when type is "recursion"


## run in default setting
# python3 -m accelerate.commands.launch \
#     --num_processes=1 \
#     -m lmms_eval \
#     --model llava \
#     --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
#     --tasks pope \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix llava_v1.6_pope \
#     --output_path ./logs/ \
#     --generation_type default \
#     --fix_grid default \
#     --verbosity INFO 
    #--wandb_args "project=llava1.6_recursive_eval,entity=VLM_Hallucination_Woohyeon,name=mme_default" 

# python3 -m accelerate.commands.launch \
#     --num_processes=1 \
#     -m lmms_eval \
#     --model llava \
#     --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
#     --tasks mme \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix llava_v1.6_mme \
#     --output_path ./logs/ \
#     --generation_type recursion \
#     --fix_grid 2x2 \
#     --attention_thresholding_type layer_mean_with_top_k \
#     --attention_threshold 0.1 \
#     --remove_unpadding True \
#     --regenerate_condition all \
#     --detection False \
#     --detection_threshold 1.0 \
#     --verbosity DEBUG \
#     --wandb_args "project=llava1.6_recursive_eval,entity=VLM_Hallucination_Woohyeon,name=mme_recursive_no_detection_topk_0.1"

# python3 -m accelerate.commands.launch \
#     --num_processes=1 \
#     -m lmms_eval \
#     --model llava \
#     --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
#     --tasks scienceqa \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix llava_v1.6_scienceqa \
#     --output_path ./logs/ \
#     --generation_type recursion_retain_base \
#     --fix_grid 2x2 \
#     --attention_thresholding_type layer_mean_with_top_k \
#     --attention_threshold 0.1 \
#     --remove_unpadding True \
#     --regenerate_condition all \
#     --detection False \
#     --detection_threshold 1.0 \
#     --verbosity DEBUG \
#     --wandb_args "project=llava1.6_recursive_eval,entity=VLM_Hallucination_Woohyeon,name=scienceqa_recursion_retain_base_no_detection_topk_0.1"

# python3 -m accelerate.commands.launch \
#     --num_processes=1 \
#     -m lmms_eval \
#     --model llava \
#     --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
#     --tasks mme \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix llava_v1.6_mme \
#     --output_path ./logs/ \
#     --generation_type recursion_retain_base \
#     --fix_grid 2x2 \
#     --attention_thresholding_type layer_mean_with_top_k \
#     --attention_threshold 0.2 \
#     --remove_unpadding True \
#     --regenerate_condition all \
#     --detection False \
#     --detection_threshold 1.0 \
#     --verbosity DEBUG \
#     --wandb_args "project=llava1.6_recursive_eval,entity=VLM_Hallucination_Woohyeon,name=mme_recursive_retain_base_no_detection_topk_0.2"

python3 -m accelerate.commands.launch \
    --num_processes=1 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
    --tasks pope,mme,scienceqa,mmstar,realworldqa \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.6_default_remove_unpad_squared \
    --output_path ./logs/ \
    --generation_type default \
    --fix_grid 2x2 \
    --attention_thresholding_type layer_mean_with_top_k \
    --attention_threshold 0.1 \
    --detection False \
    --detection_threshold 1.0 \
    --remove_unpadding True \
    --verbosity DEBUG \
    --wandb_args "project=llava1.6_recursive_eval,entity=VLM_Hallucination_Woohyeon,name=ND_default_remove_unpad_squared"

python3 -m accelerate.commands.launch \
    --num_processes=1 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
    --tasks mmstar,realworldqa \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.6_default_with_unpad_squared \
    --output_path ./logs/ \
    --generation_type default \
    --fix_grid 2x2 \
    --attention_thresholding_type layer_mean_with_top_k \
    --attention_threshold 0.1 \
    --detection False \
    --detection_threshold 1.0 \
    --verbosity DEBUG \
    --wandb_args "project=llava1.6_recursive_eval,entity=VLM_Hallucination_Woohyeon,name=ND_default_with_unpad_squared"





    
    