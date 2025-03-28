#!/bin/bash
#nohup ./run_tta.sh > ./nohup_log.out 2>&1 &

## generation_type: [recursion, recursion_retain_base, default, base-only, split-only]
# string "recursion" must be in generation_type for recursive generation

## fix_grid: [default, 2x2] - 2x2: force the image to be square
## attention_thresholding_type: [layer_mean, layer_mean_with_top_k]
## remove_unpadding: set True to remove unpadding and set mm_patch_merge_type='spatial' else 'spatial_unpad'
## regenerate_condition: currently "all": always regenerate when type is "recursion"
## --model_args pretrained="[liuhaotian/llava-v1.6-vicuna-7b, liuhaotian/llava-v1.6-mistral-7b]"\
## --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b"\

python3 -m accelerate.commands.launch \
    --num_processes=1 \
    -m lmms_eval \
    --device cuda:1 \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
    --tasks pope_pop \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.6_pope \
    --output_path ./logs/ \
    --generation_type recursion \
    --fix_grid 2x2 \
    --attention_thresholding_type layer_mean \
    --attention_threshold "[0.3,0.3,0.3]" \
    --positional_embedding_type bilinear_interpolation \
    --remove_unpadding True \
    --attn_norm norm_relu \
    --stages "-2" "-1" "0" "1" \
    --verbosity DEBUG \
    --square 1 \
    --tta_learning_rate 2e-2 \
    --tta_n_iter 500 \
    --wandb_args "project=llava1.6_recursive_eval_search_TTA_1211,entity=VLM_Hallucination_Woohyeon,name=tta-30-30-30-relu-loss-inverse-lr0.02-iter500" \
    # --visualize_heatmap False \
    # --save_output True \
    # --output_csv_path "./generation_output_pope_84-168-336-672-pad-topk-80.csv" \

# python3 -m accelerate.commands.launch \
#     --num_processes=1 \
#     -m lmms_eval \
#     --device cuda:3 \
#     --model llava \
#     --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
#     --tasks pope_pop \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix llava_v1.6_pope \
#     --output_path ./logs/ \
#     --generation_type recursion \
#     --fix_grid 2x2 \
#     --attention_thresholding_type layer_mean \
#     --attention_threshold "[0.1,0.1,0.3]" \
#     --positional_embedding_type reduced \
#     --remove_unpadding True \
#     --attn_norm norm_relu \
#     --stages "-2" "-1" "0" "1" \
#     --verbosity DEBUG \
#     --square 1 \
#     --tta_learning_rate 1e-02 \
#     --tta_n_iter 40 \
#     --wandb_args "project=llava1.6_recursive_eval_1209,entity=VLM_Hallucination_Woohyeon,name=tta-10-10-30-reduced-relu-lr0.01-iter40" \
#     # --visualize_heatmap False \
#     # --save_output True \
#     # --output_csv_path "./generation_output_pope_84-168-336-672-pad-topk-80.csv" \

# python3 -m accelerate.commands.launch \
#     --num_processes=1 \
#     -m lmms_eval \
#     --device cuda:3 \
#     --model llava \
#     --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
#     --tasks pope_pop \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix llava_v1.6_pope \
#     --output_path ./logs/ \
#     --generation_type recursion \
#     --fix_grid 2x2 \
#     --attention_thresholding_type layer_mean \
#     --attention_threshold "[0.1,0.1,0.3]" \
#     --positional_embedding_type bilinear_interpolation \
#     --remove_unpadding True \
#     --attn_norm norm_relu \
#     --stages "-2" "-1" "0" "1" \
#     --verbosity DEBUG \
#     --square 1 \
#     --tta_learning_rate 1e-02 \
#     --tta_n_iter 100 \
#     --wandb_args "project=llava1.6_recursive_eval_1209,entity=VLM_Hallucination_Woohyeon,name=tta-10-10-30-bilinear-relu-lr0.01-iter100" \
#     # --visualize_heatmap False \
#     # --save_output True \
#     # --output_csv_path "./generation_output_pope_84-168-336-672-pad-topk-80.csv" \

# python3 -m accelerate.commands.launch \
#     --num_processes=1 \
#     -m lmms_eval \
#     --device cuda:3 \
#     --model llava \
#     --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
#     --tasks pope_pop \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix llava_v1.6_pope \
#     --output_path ./logs/ \
#     --generation_type recursion \
#     --fix_grid 2x2 \
#     --attention_thresholding_type layer_mean \
#     --attention_threshold "[0.01,0.01,0.3]" \
#     --positional_embedding_type bilinear_interpolation \
#     --remove_unpadding True \
#     --attn_norm norm_relu \
#     --stages "-2" "-1" "0" "1" \
#     --verbosity DEBUG \
#     --square 1 \
#     --tta_learning_rate 1e-02 \
#     --tta_n_iter 40 \
#     --wandb_args "project=llava1.6_recursive_eval_1209,entity=VLM_Hallucination_Woohyeon,name=tta-1-1-30-bilinear-relu-lr0.01-iter40" \
#     # --visualize_heatmap False \
#     # --save_output True \
#     # --output_csv_path "./generation_output_pope_84-168-336-672-pad-topk-80.csv" \

# python3 -m accelerate.commands.launch \
#     --num_processes=1 \
#     -m lmms_eval \
#     --device cuda:3 \
#     --model llava \
#     --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
#     --tasks pope_pop \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix llava_v1.6_pope \
#     --output_path ./logs/ \
#     --generation_type recursion \
#     --fix_grid 2x2 \
#     --attention_thresholding_type layer_mean \
#     --attention_threshold "[0.1,0.3,0.5]" \
#     --positional_embedding_type bilinear_interpolation \
#     --remove_unpadding True \
#     --attn_norm norm_relu \
#     --stages "-2" "-1" "0" "1" \
#     --verbosity DEBUG \
#     --square 1 \
#     --tta_learning_rate 1e-02 \
#     --tta_n_iter 40 \
#     --wandb_args "project=llava1.6_recursive_eval_1209,entity=VLM_Hallucination_Woohyeon,name=tta-10-30-50-bilinear-relu-lr0.01-iter40" \
#     # --visualize_heatmap False \
#     # --save_output True \
#     # --output_csv_path "./generation_output_pope_84-168-336-672-pad-topk-80.csv" \

# python3 -m accelerate.commands.launch \
#     --num_processes=1 \
#     -m lmms_eval \
#     --device cuda:3 \
#     --model llava \
#     --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
#     --tasks pope_pop \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix llava_v1.6_pope \
#     --output_path ./logs/ \
#     --generation_type recursion \
#     --fix_grid 2x2 \
#     --attention_thresholding_type layer_mean \
#     --attention_threshold "[0.1,0.3]" \
#     --positional_embedding_type bilinear_interpolation \
#     --remove_unpadding True \
#     --attn_norm norm_relu \
#     --stages "-1" "0" "1" \
#     --verbosity DEBUG \
#     --square 1 \
#     --tta_learning_rate 1e-02 \
#     --tta_n_iter 40 \
#     --wandb_args "project=llava1.6_recursive_eval_1209,entity=VLM_Hallucination_Woohyeon,name=tta-10-30-bilinear-relu-lr0.01-iter40" \
#     # --visualize_heatmap False \
#     # --save_output True \
#     # --output_csv_path "./generation_output_pope_84-168-336-672-pad-topk-80.csv" \

# python3 -m accelerate.commands.launch \
#     --num_processes=1 \
#     -m lmms_eval \
#     --device cuda:3 \
#     --model llava \
#     --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
#     --tasks pope_pop \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix llava_v1.6_pope \
#     --output_path ./logs/ \
#     --generation_type recursion \
#     --fix_grid 2x2 \
#     --attention_thresholding_type layer_mean \
#     --attention_threshold "[0.1,0.1]" \
#     --positional_embedding_type bilinear_interpolation \
#     --remove_unpadding True \
#     --attn_norm norm_relu \
#     --stages "-2" "-1" "0" "1" \
#     --verbosity DEBUG \
#     --square 1 \
#     --tta_learning_rate 1e-02 \
#     --tta_n_iter 40 \
#     --wandb_args "project=llava1.6_recursive_eval_1209,entity=VLM_Hallucination_Woohyeon,name=tta-10-10-bilinear-relu-lr0.01-iter40" \
#     # --visualize_heatmap False \
#     # --save_output True \
#     # --output_csv_path "./generation_output_pope_84-168-336-672-pad-topk-80.csv" \

# python3 -m accelerate.commands.launch \
#     --num_processes=1 \
#     -m lmms_eval \
#     --device cuda:3 \
#     --model llava \
#     --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
#     --tasks pope_pop \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix llava_v1.6_pope \
#     --output_path ./logs/ \
#     --generation_type recursion \
#     --fix_grid 2x2 \
#     --attention_thresholding_type layer_mean \
#     --attention_threshold "[0.1,0.1,0.1]" \
#     --positional_embedding_type bilinear_interpolation \
#     --remove_unpadding True \
#     --attn_norm norm_relu \
#     --stages "-2" "-1" "0" "1" \
#     --verbosity DEBUG \
#     --square 1 \
#     --tta_learning_rate 1e-02 \
#     --tta_n_iter 40 \
#     --wandb_args "project=llava1.6_recursive_eval_1209,entity=VLM_Hallucination_Woohyeon,name=tta-10-10-10-bilinear-relu-lr0.01-iter40" \
#     # --visualize_heatmap False \
#     # --save_output True \
#     # --output_csv_path "./generation_output_pope_84-168-336-672-pad-topk-80.csv" \

# python3 -m accelerate.commands.launch \
#     --num_processes=1 \
#     -m lmms_eval \
#     --device cuda:3 \
#     --model llava \
#     --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
#     --tasks pope_pop \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix llava_v1.6_pope \
#     --output_path ./logs/ \
#     --generation_type recursion \
#     --fix_grid 2x2 \
#     --attention_thresholding_type layer_mean \
#     --attention_threshold "[0.1,0.1,0.3]" \
#     --positional_embedding_type bilinear_interpolation \
#     --remove_unpadding True \
#     --attn_norm norm_relu \
#     --stages "-2" "-1" "0" "1" \
#     --verbosity DEBUG \
#     --square 1 \
#     --tta_learning_rate 5e-03 \
#     --tta_n_iter 100 \
#     --wandb_args "project=llava1.6_recursive_eval_1209,entity=VLM_Hallucination_Woohyeon,name=tta-10-10-30-bilinear-relu-lr0.005-iter100" \
#     # --visualize_heatmap False \
#     # --save_output True \
#     # --output_csv_path "./generation_output_pope_84-168-336-672-pad-topk-80.csv" \

# python3 -m accelerate.commands.launch \
#     --num_processes=1 \
#     -m lmms_eval \
#     --device cuda:3 \
#     --model llava \
#     --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
#     --tasks pope_pop \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix llava_v1.6_pope \
#     --output_path ./logs/ \
#     --generation_type recursion \
#     --fix_grid 2x2 \
#     --attention_thresholding_type layer_mean \
#     --attention_threshold "[0.1,0.1,0.3]" \
#     --positional_embedding_type bilinear_interpolation \
#     --remove_unpadding True \
#     --attn_norm norm_relu \
#     --stages "-2" "-1" "0" "1" \
#     --verbosity DEBUG \
#     --square 1 \
#     --tta_learning_rate 1e-03 \
#     --tta_n_iter 300 \
#     --wandb_args "project=llava1.6_recursive_eval_1209,entity=VLM_Hallucination_Woohyeon,name=tta-10-10-30-bilinear-relu-lr0.01-iter300" \
#     # --visualize_heatmap False \
#     # --save_output True \
#     # --output_csv_path "./generation_output_pope_84-168-336-672-pad-topk-80.csv" \