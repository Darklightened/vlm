#!/bin/bash


## generation_type: [recursion, recursion_retain_base, default, base-only, split-only]
# string "recursion" must be in generation_type for recursive generation

## fix_grid: [default, 2x2] - 2x2: force the image to be square
## attention_thresholding_type: [layer_mean, layer_mean_with_top_k]
## remove_unpadding: set True to remove unpadding and set mm_patch_merge_type='spatial' else 'spatial_unpad'
## regenerate_condition: currently "all": always regenerate when type is "recursion"

# python3 -m accelerate.commands.launch \
#     --num_processes=3 \
#     -m lmms_eval \
#     --model llava \
#     --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
#     --tasks mmbench_en_dev_lite,mme,mmmu_val,mmstar,vqav2_val_lite,realworldqa,pope_pop,textvqa_val_lite,chartqa_lite,seedbench_lite \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix llava_v1.6 \
#     --output_path ./logs/ \
#     --generation_type total \
#     --fix_grid 2x2 \
#     --remove_unpadding True \
#     --attention_thresholding_type layer_mean_topk \
#     --attention_threshold "0.3" \
#     --attn_norm None \
#     --stages "0" "1" \
#     --verbosity DEBUG \
#     --wandb_args "project=llava1.6_recursive_eval,entity=VLM_Hallucination_Woohyeon,name=336-672-total" \
#     # --save_output True \
#     # --output_csv_path "./generation_output_pope_84-168-336-672-topk-80.csv" \
#     # --visualize_heatmap True \

# python3 -m accelerate.commands.launch \
#     --num_processes=1 \
#     -m lmms_eval \
#     --device cuda:0 \
#     --model llava \
#     --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
#     --tasks vqav2_val_lite \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix llava_v1.6_pope \
#     --output_path ./logs/ \
#     --generation_type total \
#     --fix_grid 2x2 \
#     --attention_thresholding_type layer_mean_topk \
#     --attention_threshold 0.3 \
#     --remove_unpadding True \
#     --attn_norm None \
#     --stages "-2" "-1" "0" "1" \
#     --verbosity DEBUG \
#     --wandb_args "project=llava1.6_recursive_eval_1126,entity=VLM_Hallucination_Woohyeon,name=82-168-336-672-pad-total" \
#     # --save_output True \
#     # --output_csv_path "./generation_output_pope_84-168-336-672-topk-80.csv" \
#     # --visualize_heatmap True \

# python3 -m accelerate.commands.launch \
#     --num_processes=3 \
#     -m lmms_eval \
#     --model llava \
#     --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
#     --tasks mmbench_en_dev_lite,mme,mmmu_val,mmstar,vqav2_val_lite,realworldqa,pope_pop,textvqa_val_lite,chartqa_lite \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix llava_v1.6 \
#     --output_path ./logs/ \
#     --generation_type total \
#     --fix_grid 2x2 \
#     --remove_unpadding True \
#     --attention_thresholding_type layer_mean_topk \
#     --attention_threshold "0.3" \
#     --attn_norm None \
#     --stages "-1" "0" "1" \
#     --verbosity DEBUG \
#     --wandb_args "project=llava1.6_recursive_eval,entity=VLM_Hallucination_Woohyeon,name=168-336-672-total" \
#     # --save_output True \
#     # --output_csv_path "./generation_output_pope_84-168-336-672-topk-80.csv" \
#     # --visualize_heatmap True \

CUDA_VISIBLE_DEVICES=1 python3 -m accelerate.commands.launch \
    --main_process_port 12345 \
    --num_processes=1 \
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
    --attention_thresholding_type layer_mean_topk \
    --attention_threshold "[1.0,1.0,0.7]" \
    --positional_embedding_type bilinear_interpolation \
    --remove_unpadding True \
    --attn_norm norm \
    --stages "-2" "-1" "0" "1" \
    --verbosity DEBUG \
    --square 1 \
    --tta_learning_rate 0.01 \
    --tta_n_iter 0 \
    --per_sample_iter 1 \
    --contrastive_alpha 0.7 \
    --wandb_args "project=llava1.6_recursive_eval_1219,entity=VLM_Hallucination_Woohyeon,name=tta-topk-100-100-70-bilinear-norm-lr001-iter0-n1-alpha07" \
    --save_output True \
    --output_csv_path "./generation_output_pope_aokvqa_pop_tta-topk-100-100-70-bilinear-norm-lr001-iter0-n1-alpha07.csv" \
