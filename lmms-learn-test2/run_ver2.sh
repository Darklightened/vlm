#!/bin/bash


## generation_type: [recursion, recursion_retain_base, default, base-only, split-only]
# string "recursion" must be in generation_type for recursive generation

## fix_grid: [default, 2x2] - 2x2: force the image to be square
## attention_thresholding_type: [layer_mean, layer_mean_with_top_k]
## remove_unpadding: set True to remove unpadding and set mm_patch_merge_type='spatial' else 'spatial_unpad'
## regenerate_condition: currently "all": always regenerate when type is "recursion"

python3 -m accelerate.commands.launch \
    --num_processes=4 \
    -m lmms_eval \
    --tasks mmbench_en_dev_lite,mme,mmmu_val,mmstar,vqav2_val_lite,realworldqa,pope_pop,textvqa_val_lite,chartqa_lite,seedbench_lite \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.6_pope \
    --output_path ./logs/ \
    --generation_type recursion \
    --fix_grid 2x2 \
    --attention_thresholding_type layer_mean_topk \
    --attention_threshold "0.9" \
    --positional_embedding_type bilinear_interpolation \
    --remove_unpadding True \
    --attn_norm None \
    --stages "-2" "-1" "0" "1" \
    --verbosity DEBUG \
    --save_output True \
    --output_csv_path "./generation_output_5stages-bil-0.9.csv" \
    --wandb_args "project=llava1.6_recursive_eval_1126,entity=VLM_Hallucination_Woohyeon,name=5stages-bil-0.9" \
    # --visualize_heatmap True \

python3 -m accelerate.commands.launch \
    --num_processes=4 \
    -m lmms_eval \
    --tasks mmbench_en_dev_lite,mme,mmmu_val,mmstar,vqav2_val_lite,realworldqa,pope_pop,textvqa_val_lite,chartqa_lite,seedbench_lite \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.6_pope \
    --output_path ./logs/ \
    --generation_type recursion \
    --fix_grid 2x2 \
    --attention_thresholding_type layer_mean_topk \
    --attention_threshold "0.9" \
    --positional_embedding_type reduced \
    --remove_unpadding True \
    --attn_norm None \
    --stages "-2" "-1" "0" "1" \
    --verbosity DEBUG \
    --save_output True \
    --output_csv_path "./generation_output_5stages-reduced-0.9.csv" \
    --wandb_args "project=llava1.6_recursive_eval_1126,entity=VLM_Hallucination_Woohyeon,name=5stages-reduced-0.9" \
    # --visualize_heatmap True \

python3 -m accelerate.commands.launch \
    --num_processes=4 \
    -m lmms_eval \
    --tasks mmbench_en_dev_lite,mme,mmmu_val,mmstar,vqav2_val_lite,realworldqa,pope_pop,textvqa_val_lite,chartqa_lite,seedbench_lite \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.6_pope \
    --output_path ./logs/ \
    --generation_type recursion \
    --fix_grid 2x2 \
    --attention_thresholding_type layer_mean_topk \
    --attention_threshold "0.5" \
    --positional_embedding_type bilinear_interpolation \
    --remove_unpadding True \
    --attn_norm None \
    --stages "-2" "-1" "0" "1" \
    --verbosity DEBUG \
    --save_output True \
    --output_csv_path "./generation_output_5stages-bil-0.5.csv" \
    --wandb_args "project=llava1.6_recursive_eval_1126,entity=VLM_Hallucination_Woohyeon,name=5stages-bil-0.5" \
    # --visualize_heatmap True \
