#!/bin/bash


## generation_type: [recursion, recursion_retain_base, default, base-only, split-only]
# string "recursion" must be in generation_type for recursive generation

## fix_grid: [default, 2x2] - 2x2: force the image to be square
## attention_thresholding_type: [layer_mean, layer_mean_with_top_k]
## remove_unpadding: set True to remove unpadding and set mm_patch_merge_type='spatial' else 'spatial_unpad'
## regenerate_condition: currently "all": always regenerate when type is "recursion"

python3 -m accelerate.commands.launch \
    --num_processes=1 \
    -m lmms_eval \
    --device cuda:3 \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
    --tasks pope,mme,mmstar \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.6_pope \
    --output_path ./logs/ \
    --generation_type recursion \
    --fix_grid 2x2 \
    --attention_thresholding_type layer_mean_topk \
    --attention_threshold 0.2 \
    --remove_unpadding True \
    --attn_norm True \
    --stages "-1" "0" "1" \
    --verbosity DEBUG \
    --wandb_args "project=llava1.6_recursive_eval_woohye0n,entity=VLM_Hallucination_Woohyeon,name=168-336-672-topk-20"
    # --visualize_heatmap True \
    # --save_output True \
    # --output_csv_path "./generation_output_pope_168-336-672-topk-20.csv" \

