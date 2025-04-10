#!/bin/bash


## generation_type: [recursion, recursion_retain_base, default, base-only, split-only]
# string "recursion" must be in generation_type for recursive generation

## fix_grid: [default, 2x2] - 2x2: force the image to be square
## attention_thresholding_type: [layer_mean, layer_mean_with_top_k]
## remove_unpadding: set True to remove unpadding and set mm_patch_merge_type='spatial' else 'spatial_unpad'
## regenerate_condition: currently "all": always regenerate when type is "recursion"


# python3 -m accelerate.commands.launch \
#     --num_processes=1 \
#     -m lmms_eval \
#     --device cuda:0 \
#     --model llava \
#     --model_args pretrained="microsoft/llava-med-v1.5-mistral-7b" \
#     --tasks vqa-rad \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix llava_v1.6_vqa-rad \
#     --output_path ./logs/ \
#     --generation_type total \
#     --fix_grid 2x2 \
#     --attention_thresholding_type layer_mean_topk \
#     --attention_threshold "0.9" "0.6" \
#     --remove_unpadding True \
#     --attn_norm None \
#     --stages "0" "1" \
#     --verbosity DEBUG \
#     --wandb_args "project=llava1.6_recursive_eval,entity=VLM_Hallucination_Woohyeon,name=168-336-672-pad-topk-90-60" \
#     --save_output True \
#     --output_csv_path "./generation_output_vqa-rad_168-336-672-topk-90-60.csv" \

python3 -m accelerate.commands.launch \
    --num_processes=1 \
    -m lmms_eval \
    --device cuda:1 \
    --model llava \
    --model_args pretrained="microsoft/llava-med-v1.5-mistral-7b" \
    --tasks vqa-rad \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.6_vqa-rad \
    --output_path ./logs/ \
    --generation_type recursion \
    --attention_thresholding_type layer_mean_topk \
    --attention_threshold "0.9" \
    --attn_norm None \
    --stages "0" "1" \
    --verbosity DEBUG \
    
    