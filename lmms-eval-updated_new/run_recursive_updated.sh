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
#     --attention_threshold "[0.5,0.5,0.5]" \
#     --positional_embedding_type bilinear_interpolation \
#     --remove_unpadding True \
#     --attn_norm None \
#     --stages "-2" "-1" "0" "1" \
#     --verbosity DEBUG \
#     --square 1 \
#     --contrastive_alphas "0.0" "0.0" "0.0" \
#     --wandb_args "project=llava1.6_recursive_eval_check_0112,entity=VLM_Hallucination_Woohyeon,name=vqav2_val_lite_0.5_0.5_0.5_bilinear_s1" \
#     # --save_output True \
#     # --output_json_path "./logit_mmstar_100_70_50_bilinear_s1.json" \
#     # --output_csv_path "./generation_output_mmstar_100_70_50_bilinear_s1.csv"

CUDA_VISIBLE_DEVICES=2 python3 -m accelerate.commands.launch \
    --num_processes=1 \
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
    --attention_threshold "[2.0,2.0,2.0]" \
    --positional_embedding_type bilinear_interpolation \
    --remove_unpadding True \
    --attn_norm None \
    --stages "-2" "-1" "0" "1" \
    --verbosity DEBUG \
    --square 1 \
    --contrastive_alphas "0.0" "0.0" "0.0" \
    --wandb_args "project=llava1.6_recursive_eval_check,entity=VLM_Hallucination_Woohyeon,name=mmbench_en_dev_lite_2.0_2.0_2.0_bilinear_s1" \
    --save_output True \
    --output_json_path "./logit_mmbench_en_dev_lite_2.0_2.0_2.0_bilinear_s1.json" \
    --output_csv_path "./generation_output_mmbench_en_dev_lite_2.0_2.0_2.0_bilinear_s1.csv"

CUDA_VISIBLE_DEVICES=2 python3 -m accelerate.commands.launch \
    --num_processes=1 \
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
    --attention_threshold "[0.7,0.7,0.7]" \
    --positional_embedding_type bilinear_interpolation \
    --remove_unpadding True \
    --attn_norm None \
    --stages "-2" "-1" "0" "1" \
    --verbosity DEBUG \
    --square 1 \
    --contrastive_alphas "0.0" "0.0" "0.0" \
    --wandb_args "project=llava1.6_recursive_eval_check,entity=VLM_Hallucination_Woohyeon,name=mmstar_0.7_0.7_0.7_bilinear_s1" \
    --save_output True \
    --output_json_path "./logit_mmstar_0.7_0.7_0.7_bilinear_s1.json" \
    --output_csv_path "./generation_output_mmstar_0.7_0.7_0.7_bilinear_s1.csv"

CUDA_VISIBLE_DEVICES=2 python3 -m accelerate.commands.launch \
    --num_processes=1 \
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
    --attention_threshold "[2.5,2.5,2.5]" \
    --positional_embedding_type bilinear_interpolation \
    --remove_unpadding True \
    --attn_norm None \
    --stages "-2" "-1" "0" "1" \
    --verbosity DEBUG \
    --square 1 \
    --contrastive_alphas "0.0" "0.0" "0.0" \
    --wandb_args "project=llava1.6_recursive_eval_check,entity=VLM_Hallucination_Woohyeon,name=pope_pop_2.5_2.5_2.5_bilinear_s1" \
    --save_output True \
    --output_json_path "./logit_pope_pop_2.5_2.5_2.5_bilinear_s1.json" \
    --output_csv_path "./generation_output_pope_pop_2.5_2.5_2.5_bilinear_s1.csv"

CUDA_VISIBLE_DEVICES=2 python3 -m accelerate.commands.launch \
    --num_processes=1 \
    --main_process_port 29834 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
    --tasks pope_gqa_pop \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.6_pope \
    --output_path ./logs/ \
    --generation_type recursion \
    --fix_grid 2x2 \
    --attention_thresholding_type attn_topk \
    --attention_threshold "[2.5,2.5,2.5]" \
    --positional_embedding_type bilinear_interpolation \
    --remove_unpadding True \
    --attn_norm None \
    --stages "-2" "-1" "0" "1" \
    --verbosity DEBUG \
    --square 1 \
    --contrastive_alphas "0.0" "0.0" "0.0" \
    --wandb_args "project=llava1.6_recursive_eval_check,entity=VLM_Hallucination_Woohyeon,name=pope_gqa_pop_2.5_2.5_2.5_bilinear_s1" \
    --save_output True \
    --output_json_path "./logit_pope_gqa_pop_2.5_2.5_2.5_bilinear_s1.json" \
    --output_csv_path "./generation_output_pope_gqa_pop_2.5_2.5_2.5_bilinear_s1.csv"

CUDA_VISIBLE_DEVICES=2 python3 -m accelerate.commands.launch \
    --num_processes=1 \
    --main_process_port 29834 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
    --tasks pope_aokvqa_pop \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.6_pope \
    --output_path ./logs/ \
    --generation_type recursion \
    --fix_grid 2x2 \
    --attention_thresholding_type attn_topk \
    --attention_threshold "[2.5,2.5,2.5]" \
    --positional_embedding_type bilinear_interpolation \
    --remove_unpadding True \
    --attn_norm None \
    --stages "-2" "-1" "0" "1" \
    --verbosity DEBUG \
    --square 1 \
    --contrastive_alphas "0.0" "0.0" "0.0" \
    --wandb_args "project=llava1.6_recursive_eval_check,entity=VLM_Hallucination_Woohyeon,name=pope_pop_2.5_2.5_2.5_bilinear_s1" \
    --save_output True \
    --output_json_path "./logit_pope_aokvqa_pop_2.5_2.5_2.5_bilinear_s1.json" \
    --output_csv_path "./generation_output_pope_aokvqa_pop_2.5_2.5_2.5_bilinear_s1.csv"
