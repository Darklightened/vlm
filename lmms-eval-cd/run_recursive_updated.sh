#!/bin/bash


## generation_type: [recursion, recursion_retain_base, default, base-only, split-only]
# string "recursion" must be in generation_type for recursive generation

## fix_grid: [default, 2x2] - 2x2: force the image to be square
## attention_thresholding_type: [layer_mean, layer_mean_with_top_k]
## remove_unpadding: set True to remove unpadding and set mm_patch_merge_type='spatial' else 'spatial_unpad'
## regenerate_condition: currently "all": always regenerate when type is "recursion"
## --model_args pretrained="[liuhaotian/llava-v1.6-vicuna-7b, liuhaotian/llava-v1.6-mistral-7b]"\
## --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b"\
    # --device cuda:3 \

# CUDA_VISIBLE_DEVICES=2 python3 -m accelerate.commands.launch \
#     --num_processes=1 \
#     --main_process_port 29834 \
#     -m lmms_eval \
#     --model llava \
#     --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
#     --tasks vqav2_val_lite \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix llava_v1.6_pope \
#     --output_path ./logs/ \
#     --generation_type recursion \
#     --fix_grid 2x2 \
#     --attention_thresholding_type attn_topk \
#     --attention_threshold "[1.5,1.5,1.5]" \
#     --positional_embedding_type bilinear_interpolation \
#     --remove_unpadding True \
#     --attn_norm None \
#     --stages "-2" "-1" "0" "1" \
#     --verbosity DEBUG \
#     --square 1 \
#     --contrastive_alphas "0.0" "0.0" "0.0" \
#     --wandb_args "project=llava1.6_recursive_eval_check,entity=VLM_Hallucination_Woohyeon,name=vqav2_val_lite_1.5_1.5_1.5_bilinear_s1" \
#     # --save_output True \
#     # --output_json_path "./logit_mmstar_100_70_50_bilinear_s1.json" \
#     # --output_csv_path "./generation_output_mmstar_100_70_50_bilinear_s1.csv"

# CUDA_VISIBLE_DEVICES=2 python3 -m accelerate.commands.launch \
#     --num_processes=1 \
#     --main_process_port 29834 \
#     -m lmms_eval \
#     --model llava \
#     --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
#     --tasks mmbench_en_dev_lite \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix llava_v1.6_pope \
#     --output_path ./logs/ \
#     --generation_type recursion \
#     --fix_grid 2x2 \
#     --attention_thresholding_type attn_topk \
#     --attention_threshold "[2.0,2.0,2.0]" \
#     --positional_embedding_type bilinear_interpolation \
#     --remove_unpadding True \
#     --attn_norm None \
#     --stages "-2" "-1" "0" "1" \
#     --verbosity DEBUG \
#     --square 1 \
#     --contrastive_alphas "0.0" "0.0" "0.0" \
#     --wandb_args "project=llava1.6_recursive_eval_check,entity=VLM_Hallucination_Woohyeon,name=mmbench_en_dev_lite_1.0_1.0_1.0_bilinear_s1" \
#     --save_output True \
#     --output_json_path "./logit_mmbench_en_dev_lite_1.0_1.0_1.0_bilinear_s1.json" \
#     --output_csv_path "./generation_output_mmbench_en_dev_lite_1.0_1.0_1.0_bilinear_s1.csv"

# CUDA_VISIBLE_DEVICES=2 python3 -m accelerate.commands.launch \
#     --num_processes=1 \
#     --main_process_port 29834 \
#     -m lmms_eval \
#     --model llava \
#     --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
#     --tasks mmstar \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix llava_v1.6_pope \
#     --output_path ./logs/ \
#     --generation_type recursion \
#     --fix_grid 2x2 \
#     --attention_thresholding_type attn_topk \
#     --attention_threshold "[10.0,10.0,10.0]" \
#     --positional_embedding_type bilinear_interpolation \
#     --remove_unpadding True \
#     --attn_norm None \
#     --stages "-2" "-1" "0" "1" \
#     --verbosity DEBUG \
#     --square 1 \
#     --contrastive_alphas "0.0" "0.0" "0.0" \
#     --wandb_args "project=llava1.6_recursive_eval_check,entity=VLM_Hallucination_Woohyeon,name=mmstar_10.0_10.0_10.0_bilinear_s1" \
#     --save_output True \
#     --output_json_path "./logit_mmstar_10.0_10.0_10.0_bilinear_s1.json" \
#     --output_csv_path "./generation_output_mmstar_10.0_10.0_10.0_bilinear_s1.csv"

CUDA_VISIBLE_DEVICES=1,2 python3 -m accelerate.commands.launch \
    --num_processes=2 \
    --main_process_port 29834 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
    --tasks pope_pop \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.6_pope \
    --output_path ./logs/ \
    --generation_type recursion \
    --fix_grid 2x2 \
    --attention_thresholding_type attn_topk \
    --attention_threshold "[1.0,1.0,1.0]" \
    --positional_embedding_type bilinear_interpolation \
    --remove_unpadding True \
    --attn_norm None \
    --stages "-2" "-1" "0" "1" \
    --verbosity DEBUG \
    --square 1 \
    --contrastive_alphas "2.4" "2.2" "1.6" \
    --wandb_args "project=llava1.6_recursive_eval_7b_vicuna_cd,entity=VLM_Hallucination_Woohyeon,name=pope_new_1.0_1.0_1.0_bilinear_s1_cd_1.6-2.2-2.4" 

CUDA_VISIBLE_DEVICES=1,2 python3 -m accelerate.commands.launch \
    --num_processes=2 \
    --main_process_port 29834 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
    --tasks mmstar \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.6_pope \
    --output_path ./logs/ \
    --generation_type recursion \
    --fix_grid 2x2 \
    --attention_thresholding_type attn_topk \
    --attention_threshold "[10.0,10.0,10.0]" \
    --positional_embedding_type bilinear_interpolation \
    --remove_unpadding True \
    --attn_norm None \
    --stages "-2" "-1" "0" "1" \
    --verbosity DEBUG \
    --square 1 \
    --contrastive_alphas "1.6" "2.0" "0.9" \
    --wandb_args "project=llava1.6_recursive_eval_7b_vicuna_cd,entity=VLM_Hallucination_Woohyeon,name=mmstar_new_10.0_10.0_10.0_bilinear_s1_cd_0.9-2.0-1.6" 

CUDA_VISIBLE_DEVICES=1,2 python3 -m accelerate.commands.launch \
    --num_processes=2 \
    --main_process_port 29834 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
    --tasks mmbench_en_dev_lite \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.6_pope \
    --output_path ./logs/ \
    --generation_type recursion \
    --fix_grid 2x2 \
    --attention_thresholding_type attn_topk \
    --attention_threshold "[1.0,1.0,1.0]" \
    --positional_embedding_type bilinear_interpolation \
    --remove_unpadding True \
    --attn_norm None \
    --stages "-2" "-1" "0" "1" \
    --verbosity DEBUG \
    --square 1 \
    --contrastive_alphas "0.3" "0.0" "0.0" \
    --wandb_args "project=llava1.6_recursive_eval_7b_vicuna_cd,entity=VLM_Hallucination_Woohyeon,name=mmbench_new_1.0_1.0_1.0_bilinear_s1_cd_1.6-2.2-2.4" 
    
CUDA_VISIBLE_DEVICES=1,2 python3 -m accelerate.commands.launch \
    --num_processes=2 \
    --main_process_port 29834 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
    --tasks mmbench_en_dev_lite \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.6_pope \
    --output_path ./logs/ \
    --generation_type recursion \
    --fix_grid 2x2 \
    --attention_thresholding_type attn_topk \
    --attention_threshold "[1.0,1.0,1.0]" \
    --positional_embedding_type bilinear_interpolation \
    --remove_unpadding True \
    --attn_norm None \
    --stages "-2" "-1" "0" "1" \
    --verbosity DEBUG \
    --square 1 \
    --contrastive_alphas "0.6" "0.2" "0.4" \
    --wandb_args "project=llava1.6_recursive_eval_7b_vicuna_cd,entity=VLM_Hallucination_Woohyeon,name=mmbench_new_1.0_1.0_1.0_bilinear_s1_cd_0.4-0.2-0.6"