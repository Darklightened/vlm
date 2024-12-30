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
#     --tta_learning_rate 5e-03 \
#     --tta_n_iter 200 \
#     --wandb_args "project=llava1.6_recursive_eval_1209,entity=VLM_Hallucination_Woohyeon,name=tta-10-10-10-bilinear-relu-lr0.005-iter200" \
#     --save_output True \
#     --output_csv_path "./generation_output_pope_pop_tta-10-10-10-bilinear-relu-lr5e-03-iter200.csv" \

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
#     --attention_threshold "[0.2,0.2,0.2]" \
#     --positional_embedding_type bilinear_interpolation \
#     --remove_unpadding True \
#     --attn_norm norm_relu \
#     --stages "-2" "-1" "0" "1" \
#     --verbosity DEBUG \
#     --square 1 \
#     --tta_learning_rate 5e-03 \
#     --tta_n_iter 200 \
#     --wandb_args "project=llava1.6_recursive_eval_1209,entity=VLM_Hallucination_Woohyeon,name=tta-20-20-20-bilinear-relu-lr0.005-iter200" \
#     --save_output True \
#     --output_csv_path "./generation_output_pope_pop_tta-20-20-20-bilinear-relu-lr5e-03-iter200.csv" \

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
#     --attention_threshold "[0.3,0.3,0.3]" \
#     --positional_embedding_type bilinear_interpolation \
#     --remove_unpadding True \
#     --attn_norm norm_relu \
#     --stages "-2" "-1" "0" "1" \
#     --verbosity DEBUG \
#     --square 1 \
#     --tta_learning_rate 5e-03 \
#     --tta_n_iter 200 \
#     --wandb_args "project=llava1.6_recursive_eval_1209,entity=VLM_Hallucination_Woohyeon,name=tta-30-30-30-bilinear-relu-lr0.005-iter200" \
#     --save_output True \
#     --output_csv_path "./generation_output_pope_pop_tta-30-30-30-bilinear-relu-lr5e-03-iter200.csv" \

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
#     --tta_n_iter 200 \
#     --wandb_args "project=llava1.6_recursive_eval_1209,entity=VLM_Hallucination_Woohyeon,name=tta-10-10-10-bilinear-relu-lr0.01-iter200" \
#     --save_output True \
#     --output_csv_path "./generation_output_pope_pop_tta-10-10-10-bilinear-relu-lr1e-02-iter200.csv" \

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
#     --tta_learning_rate 5e-02 \
#     --tta_n_iter 200 \
#     --wandb_args "project=llava1.6_recursive_eval_1209,entity=VLM_Hallucination_Woohyeon,name=tta-10-10-10-bilinear-relu-lr0.05-iter200" \
#     --save_output True \
#     --output_csv_path "./generation_output_pope_pop_tta-10-10-10-bilinear-relu-lr5e-02-iter200.csv" \

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
#     --attention_threshold "[0.05,0.05,0.1]" \
#     --positional_embedding_type bilinear_interpolation \
#     --remove_unpadding True \
#     --attn_norm norm_relu \
#     --stages "-2" "-1" "0" "1" \
#     --verbosity DEBUG \
#     --square 1 \
#     --tta_learning_rate 5e-03 \
#     --tta_n_iter 200 \
#     --wandb_args "project=llava1.6_recursive_eval_1209,entity=VLM_Hallucination_Woohyeon,name=tta-5-5-10-bilinear-relu-lr0.005-iter200" \
#     --save_output True \
#     --output_csv_path "./generation_output_pope_pop_tta-10-10-10-bilinear-relu-lr5e-03-iter200.csv" \

# python3 -m accelerate.commands.launch \
#     --num_processes=1 \
#     -m lmms_eval \
#     --device cuda:2 \
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
#     --attention_threshold "[0.2,0.2,0.2]" \
#     --positional_embedding_type bilinear_interpolation \
#     --remove_unpadding True \
#     --attn_norm norm \
#     --stages "-2" "-1" "0" "1" \
#     --verbosity DEBUG \
#     --square 1 \
#     --tta_learning_rate 0.005 \
#     --tta_n_iter 100 \
#     --per_sample_iter 10 \
#     --wandb_args "project=llava1.6_recursive_eval_1211,entity=VLM_Hallucination_Woohyeon,name=tta-top-20-20-20-bilinear-norm-lr0.005-iter100-n10" \
#     # --save_output True \
#     # --output_csv_path "./generation_output_pope_pop_tta-10-10-10-bilinear-relu-lr5e-02-iter3000-n3.csv" \

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
#     --attention_thresholding_type layer_mean_topk \
#     --attention_threshold "[1.0,1.0,0.4]" \
#     --positional_embedding_type bilinear_interpolation \
#     --remove_unpadding True \
#     --attn_norm norm \
#     --stages "-2" "-1" "0" "1" \
#     --verbosity DEBUG \
#     --square 1 \
#     --tta_learning_rate 0.01 \
#     --tta_n_iter 100 \
#     --per_sample_iter 1 \
#     --wandb_args "project=llava1.6_recursive_eval_1211,entity=VLM_Hallucination_Woohyeon,name=tta-topk-100-100-40-bilinear-norm-lr0.01-iter100-n1" \
#     --save_output True \
#     --output_csv_path "./generation_output_pope_pop_tta-topk-100-100-40-bilinear-norm-lr0.01-iter100-n1.csv" \

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
#     --attention_thresholding_type layer_mean_topk \
#     --attention_threshold "[1.0,1.0,0.5]" \
#     --positional_embedding_type bilinear_interpolation \
#     --remove_unpadding True \
#     --attn_norm norm \
#     --stages "-2" "-1" "0" "1" \
#     --verbosity DEBUG \
#     --square 1 \
#     --tta_learning_rate 0.01 \
#     --tta_n_iter 100 \
#     --per_sample_iter 1 \
#     --wandb_args "project=llava1.6_recursive_eval_1211,entity=VLM_Hallucination_Woohyeon,name=tta-topk-100-100-50-bilinear-norm-lr0.01-iter100-n1" \
#     --save_output True \
#     --output_csv_path "./generation_output_pope_pop_tta-topk-100-100-50-bilinear-norm-lr0.01-iter100-n1.csv" \

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
#     --attention_thresholding_type layer_mean_topk \
#     --attention_threshold "[1.0,1.0,0.3]" \
#     --positional_embedding_type bilinear_interpolation \
#     --remove_unpadding True \
#     --attn_norm norm \
#     --stages "-2" "-1" "0" "1" \
#     --verbosity DEBUG \
#     --square 1 \
#     --tta_learning_rate 0.01 \
#     --tta_n_iter 200 \
#     --per_sample_iter 1 \
#     --wandb_args "project=llava1.6_recursive_eval_1211,entity=VLM_Hallucination_Woohyeon,name=tta-topk-100-100-30-bilinear-norm-lr0.01-iter200-n1" \
#     --save_output True \
#     --output_csv_path "./generation_output_pope_pop_tta-topk-100-100-30-bilinear-norm-lr0.01-iter200-n1.csv" \

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
#     --attention_thresholding_type layer_mean_topk \
#     --attention_threshold "[1.0,1.0,0.3]" \
#     --positional_embedding_type bilinear_interpolation \
#     --remove_unpadding True \
#     --attn_norm norm \
#     --stages "-2" "-1" "0" "1" \
#     --verbosity DEBUG \
#     --square 1 \
#     --tta_learning_rate 0.01 \
#     --tta_n_iter 300 \
#     --per_sample_iter 1 \
#     --wandb_args "project=llava1.6_recursive_eval_1211,entity=VLM_Hallucination_Woohyeon,name=tta-topk-100-100-30-bilinear-norm-lr0.01-iter300-n1" \
#     --save_output True \
#     --output_csv_path "./generation_output_pope_pop_tta-topk-100-100-30-bilinear-norm-lr0.01-iter300-n1.csv" \

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
#     --attention_thresholding_type layer_mean_topk \
#     --attention_threshold "[0.95,0.95,0.95]" \
#     --positional_embedding_type bilinear_interpolation \
#     --remove_unpadding True \
#     --attn_norm norm \
#     --stages "-2" "-1" "0" "1" \
#     --verbosity DEBUG \
#     --square 1 \
#     --tta_learning_rate 0.01 \
#     --tta_n_iter 100 \
#     --per_sample_iter 1 \
#     --wandb_args "project=llava1.6_recursive_eval_1211,entity=VLM_Hallucination_Woohyeon,name=tta-topk-95-95-95-bilinear-norm-lr0.01-iter100-n1" \
#     --save_output True \
#     --output_csv_path "./generation_output_pope_pop_tta-topk-95-95-95-bilinear-norm-lr0.01-iter100-n1.csv" \

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
#     --attention_thresholding_type layer_mean_topk \
#     --attention_threshold "[1.0,1.0,0.3]" \
#     --positional_embedding_type bilinear_interpolation \
#     --remove_unpadding True \
#     --attn_norm norm \
#     --stages "-2" "-1" "0" "1" \
#     --verbosity DEBUG \
#     --square 1 \
#     --tta_learning_rate 0.01 \
#     --tta_n_iter 100 \
#     --per_sample_iter 5 \
#     --wandb_args "project=llava1.6_recursive_eval_1211,entity=VLM_Hallucination_Woohyeon,name=tta-topk-100-100-30-bilinear-norm-lr0.01-iter100-n5" \
#     --save_output True \
#     --output_csv_path "./generation_output_pope_pop_tta-topk-100-100-30-bilinear-norm-lr0.01-iter100-n5.csv" \

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
#     --attention_thresholding_type layer_mean_topk \
#     --attention_threshold "[1.0,1.0,0.3]" \
#     --positional_embedding_type reduced \
#     --remove_unpadding True \
#     --attn_norm norm \
#     --stages "-2" "-1" "0" "1" \
#     --verbosity DEBUG \
#     --square 1 \
#     --tta_learning_rate 0.01 \
#     --tta_n_iter 100 \
#     --per_sample_iter 1 \
#     --wandb_args "project=llava1.6_recursive_eval_1211,entity=VLM_Hallucination_Woohyeon,name=tta-topk-100-100-30-reduced-norm-lr0.01-iter100-n1" \
#     --save_output True \
#     --output_csv_path "./generation_output_pope_pop_tta-topk-100-100-30-reduced-norm-lr0.01-iter100-n1.csv" \

# python3 -m accelerate.commands.launch \
#     --num_processes=1 \
#     -m lmms_eval \
#     --device cuda:3 \
#     --model llava \
#     --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
#     --tasks mmbench_en_dev_lite,mmmu_val,realworldqa,mme,mmstar,vqav2_val_lite \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix llava_v1.6_pope \
#     --output_path ./logs/ \
#     --generation_type recursion \
#     --fix_grid 2x2 \
#     --attention_thresholding_type layer_mean_topk \
#     --attention_threshold "[1.0,1.0,0.4]" \
#     --positional_embedding_type bilinear_interpolation \
#     --remove_unpadding True \
#     --attn_norm norm \
#     --stages "-2" "-1" "0" "1" \
#     --verbosity DEBUG \
#     --square 1 \
#     --tta_learning_rate 0.01 \
#     --tta_n_iter 100 \
#     --per_sample_iter 1 \
#     --wandb_args "project=llava1.6_recursive_eval_1211,entity=VLM_Hallucination_Woohyeon,name=tta-topk-100-100-40-bilinear-norm-lr0.01-iter100-n1" \    

# python3 -m accelerate.commands.launch \
#     --num_processes=1 \
#     -m lmms_eval \
#     --device cuda:3 \
#     --model llava \
#     --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
#     --tasks mmbench_en_dev_lite,mmmu_val,realworldqa,mme,mmstar,vqav2_val_lite \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix llava_v1.6_pope \
#     --output_path ./logs/ \
#     --generation_type recursion \
#     --fix_grid 2x2 \
#     --attention_thresholding_type layer_mean_topk \
#     --attention_threshold "[1.0,1.0,0.5]" \
#     --positional_embedding_type bilinear_interpolation \
#     --remove_unpadding True \
#     --attn_norm norm \
#     --stages "-2" "-1" "0" "1" \
#     --verbosity DEBUG \
#     --square 1 \
#     --tta_learning_rate 0.01 \
#     --tta_n_iter 100 \
#     --per_sample_iter 1 \
#     --wandb_args "project=llava1.6_recursive_eval_1211,entity=VLM_Hallucination_Woohyeon,name=tta-topk-100-100-50-bilinear-norm-lr0.01-iter100-n1" \
    

# python3 -m accelerate.commands.launch \
#     --num_processes=1 \
#     -m lmms_eval \
#     --device cuda:3 \
#     --model llava \
#     --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
#     --tasks mmbench_en_dev_lite,mmmu_val,realworldqa,mme,mmstar,vqav2_val_lite \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix llava_v1.6_pope \
#     --output_path ./logs/ \
#     --generation_type recursion \
#     --fix_grid 2x2 \
#     --attention_thresholding_type layer_mean_topk \
#     --attention_threshold "[1.0,1.0,0.3]" \
#     --positional_embedding_type bilinear_interpolation \
#     --remove_unpadding True \
#     --attn_norm norm \
#     --stages "-2" "-1" "0" "1" \
#     --verbosity DEBUG \
#     --square 1 \
#     --tta_learning_rate 0.01 \
#     --tta_n_iter 200 \
#     --per_sample_iter 1 \
#     --wandb_args "project=llava1.6_recursive_eval_1211,entity=VLM_Hallucination_Woohyeon,name=tta-topk-100-100-30-bilinear-norm-lr0.01-iter200-n1" \
    

# python3 -m accelerate.commands.launch \
#     --num_processes=1 \
#     -m lmms_eval \
#     --device cuda:3 \
#     --model llava \
#     --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
#     --tasks mmbench_en_dev_lite,mmmu_val,realworldqa,mme,mmstar,vqav2_val_lite \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix llava_v1.6_pope \
#     --output_path ./logs/ \
#     --generation_type recursion \
#     --fix_grid 2x2 \
#     --attention_thresholding_type layer_mean_topk \
#     --attention_threshold "[1.0,1.0,0.3]" \
#     --positional_embedding_type bilinear_interpolation \
#     --remove_unpadding True \
#     --attn_norm norm \
#     --stages "-2" "-1" "0" "1" \
#     --verbosity DEBUG \
#     --square 1 \
#     --tta_learning_rate 0.01 \
#     --tta_n_iter 300 \
#     --per_sample_iter 1 \
#     --wandb_args "project=llava1.6_recursive_eval_1211,entity=VLM_Hallucination_Woohyeon,name=tta-topk-100-100-30-bilinear-norm-lr0.01-iter300-n1" \
    

# python3 -m accelerate.commands.launch \
#     --num_processes=1 \
#     -m lmms_eval \
#     --device cuda:3 \
#     --model llava \
#     --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
#     --tasks mmbench_en_dev_lite,mmmu_val,realworldqa,mme,mmstar,vqav2_val_lite \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix llava_v1.6_pope \
#     --output_path ./logs/ \
#     --generation_type recursion \
#     --fix_grid 2x2 \
#     --attention_thresholding_type layer_mean_topk \
#     --attention_threshold "[0.95,0.95,0.95]" \
#     --positional_embedding_type bilinear_interpolation \
#     --remove_unpadding True \
#     --attn_norm norm \
#     --stages "-2" "-1" "0" "1" \
#     --verbosity DEBUG \
#     --square 1 \
#     --tta_learning_rate 0.01 \
#     --tta_n_iter 100 \
#     --per_sample_iter 1 \
#     --wandb_args "project=llava1.6_recursive_eval_1211,entity=VLM_Hallucination_Woohyeon,name=tta-topk-95-95-95-bilinear-norm-lr0.01-iter100-n1" \
    

# python3 -m accelerate.commands.launch \
#     --num_processes=1 \
#     -m lmms_eval \
#     --device cuda:3 \
#     --model llava \
#     --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
#     --tasks mmbench_en_dev_lite,mmmu_val,realworldqa,mme,mmstar,vqav2_val_lite \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix llava_v1.6_pope \
#     --output_path ./logs/ \
#     --generation_type recursion \
#     --fix_grid 2x2 \
#     --attention_thresholding_type layer_mean_topk \
#     --attention_threshold "[1.0,1.0,0.3]" \
#     --positional_embedding_type bilinear_interpolation \
#     --remove_unpadding True \
#     --attn_norm norm \
#     --stages "-2" "-1" "0" "1" \
#     --verbosity DEBUG \
#     --square 1 \
#     --tta_learning_rate 0.01 \
#     --tta_n_iter 100 \
#     --per_sample_iter 5 \
#     --wandb_args "project=llava1.6_recursive_eval_1211,entity=VLM_Hallucination_Woohyeon,name=tta-topk-100-100-30-bilinear-norm-lr0.01-iter100-n5" \
    
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
#     --attention_thresholding_type layer_mean_topk \
#     --attention_threshold "[0.5,0.5,0.5]" \
#     --positional_embedding_type bilinear_interpolation \
#     --remove_unpadding True \
#     --attn_norm norm \
#     --stages "-2" "-1" "0" "1" \
#     --verbosity DEBUG \
#     --square 1 \
#     --tta_learning_rate 0.001 \
#     --tta_n_iter 100 \
#     --per_sample_iter 1 \
#     --wandb_args "project=llava1.6_recursive_eval_1211,entity=VLM_Hallucination_Woohyeon,name=tta-topk-50-50-50-bilinear-norm-lr0.001-iter100-n1" \
#     --save_output True \
#     --output_csv_path "./generation_output_pope_pop_tta-topk-50-50-50-bilinear-norm-lr0.001-iter100-n1.csv" \

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
#     --attention_thresholding_type layer_mean_topk \
#     --attention_threshold "[0.5,0.5,0.5]" \
#     --positional_embedding_type bilinear_interpolation \
#     --remove_unpadding True \
#     --attn_norm norm \
#     --stages "-2" "-1" "0" "1" \
#     --verbosity DEBUG \
#     --square 1 \
#     --tta_learning_rate 0.001 \
#     --tta_n_iter 100 \
#     --per_sample_iter 1 \
#     --wandb_args "project=llava1.6_recursive_eval_1211,entity=VLM_Hallucination_Woohyeon,name=tta-topk-50-50-50-bilinear-norm-lr0.001-iter100-n1" \
#     --save_output True \
#     --output_csv_path "./generation_output_pope_pop_tta-topk-50-50-50-bilinear-norm-lr0.001-iter100-n1.csv" \

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
#     --attention_thresholding_type layer_mean_topk \
#     --attention_threshold "[0.5,0.5,0.5]" \
#     --positional_embedding_type bilinear_interpolation \
#     --remove_unpadding True \
#     --attn_norm norm \
#     --stages "-2" "-1" "0" "1" \
#     --verbosity DEBUG \
#     --square 1 \
#     --tta_learning_rate 0.001 \
#     --tta_n_iter 1000 \
#     --per_sample_iter 1 \
#     --wandb_args "project=llava1.6_recursive_eval_1211,entity=VLM_Hallucination_Woohyeon,name=tta-topk-50-50-50-bilinear-norm-lr0.001-iter1000-n1" \
#     --save_output True \
#     --output_csv_path "./generation_output_pope_pop_tta-topk-50-50-50-bilinear-norm-lr0.001-iter1000-n1.csv" \

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
#     --attention_thresholding_type layer_mean_topk \
#     --attention_threshold "[0.5,0.5,0.5]" \
#     --positional_embedding_type bilinear_interpolation \
#     --remove_unpadding True \
#     --attn_norm norm \
#     --stages "-2" "-1" "0" "1" \
#     --verbosity DEBUG \
#     --square 1 \
#     --tta_learning_rate 0.01 \
#     --tta_n_iter 100 \
#     --per_sample_iter 1 \
#     --wandb_args "project=llava1.6_recursive_eval_1211,entity=VLM_Hallucination_Woohyeon,name=tta-topk-50-50-50-bilinear-norm-lr0.01-iter100-n1" \
#     --save_output True \
#     --output_csv_path "./generation_output_pope_pop_tta-topk-50-50-50-bilinear-norm-lr0.01-iter100-n1.csv" \

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
#     --attention_thresholding_type layer_mean_topk \
#     --attention_threshold "[0.5,0.5,0.5]" \
#     --positional_embedding_type bilinear_interpolation \
#     --remove_unpadding True \
#     --attn_norm norm \
#     --stages "-2" "-1" "0" "1" \
#     --verbosity DEBUG \
#     --square 1 \
#     --tta_learning_rate 0.01 \
#     --tta_n_iter 1000 \
#     --per_sample_iter 1 \
#     --wandb_args "project=llava1.6_recursive_eval_1211,entity=VLM_Hallucination_Woohyeon,name=tta-topk-50-50-50-bilinear-norm-lr0.01-iter1000-n1" \
#     --save_output True \
#     --output_csv_path "./generation_output_pope_pop_tta-topk-50-50-50-bilinear-norm-lr0.01-iter1000-n1.csv" \

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
#     --attention_thresholding_type layer_mean_topk \
#     --attention_threshold "[0.5,0.5,0.5]" \
#     --positional_embedding_type bilinear_interpolation \
#     --remove_unpadding True \
#     --attn_norm norm \
#     --stages "-2" "-1" "0" "1" \
#     --verbosity DEBUG \
#     --square 1 \
#     --tta_learning_rate 0.01 \
#     --tta_n_iter 1000 \
#     --per_sample_iter 3 \
#     --wandb_args "project=llava1.6_recursive_eval_1211,entity=VLM_Hallucination_Woohyeon,name=tta-topk-50-50-50-bilinear-norm-lr0.01-iter1000-n3" \
#     --save_output True \
#     --output_csv_path "./generation_output_pope_pop_tta-topk-50-50-50-bilinear-norm-lr0.01-iter1000-n3.csv" \

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
#     --attention_thresholding_type layer_mean_topk \
#     --attention_threshold "[0.5,0.5,0.5]" \
#     --positional_embedding_type bilinear_interpolation \
#     --remove_unpadding True \
#     --attn_norm norm \
#     --stages "-2" "-1" "0" "1" \
#     --verbosity DEBUG \
#     --square 1 \
#     --tta_learning_rate 0.01 \
#     --tta_n_iter 1000 \
#     --per_sample_iter 5 \
#     --wandb_args "project=llava1.6_recursive_eval_1211,entity=VLM_Hallucination_Woohyeon,name=tta-topk-50-50-50-bilinear-norm-lr0.01-iter1000-n5" \
#     --save_output True \
#     --output_csv_path "./generation_output_pope_pop_tta-topk-50-50-50-bilinear-norm-lr0.01-iter1000-n5.csv" \

# CUDA_VISIBLE_DEVICES=3 python3 -m accelerate.commands.launch \
#     --main_process_port 12345 \
#     --num_processes=1 \
#     -m lmms_eval \
#     --model llava \
#     --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
#     --tasks pope \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix llava_v1.6_pope \
#     --output_path ./logs/ \
#     --generation_type recursion \
#     --fix_grid 2x2 \
#     --attention_thresholding_type layer_mean_topk \
#     --attention_threshold "[0.9,0.9,0.4]" \
#     --positional_embedding_type bilinear_interpolation \
#     --remove_unpadding True \
#     --attn_norm norm \
#     --stages "-2" "-1" "0" "1" \
#     --verbosity DEBUG \
#     --square 1 \
#     --tta_learning_rate 0.01 \
#     --tta_n_iter 0 \
#     --per_sample_iter 1 \
#     --contrastive_alpha 0.7 \
#     --wandb_args "project=llava1.6_recursive_eval_1219,entity=VLM_Hallucination_Woohyeon,name=tta-topk-90-90-40-bilinear-norm-lr001-iter0-n1-alpha07" \
#     --save_output True \
#     --output_csv_path "./generation_output_pope_tta-topk-90-90-40-bilinear-norm-lr001-iter0-n1-alpha07.csv" \

# CUDA_VISIBLE_DEVICES=3 python3 -m accelerate.commands.launch \
#     --main_process_port 12345 \
#     --num_processes=1 \
#     -m lmms_eval \
#     --model llava \
#     --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
#     --tasks pope \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix llava_v1.6_pope \
#     --output_path ./logs/ \
#     --generation_type recursion \
#     --fix_grid 2x2 \
#     --attention_thresholding_type layer_mean_topk \
#     --attention_threshold "[0.9,0.9,0.5]" \
#     --positional_embedding_type bilinear_interpolation \
#     --remove_unpadding True \
#     --attn_norm norm \
#     --stages "-2" "-1" "0" "1" \
#     --verbosity DEBUG \
#     --square 1 \
#     --tta_learning_rate 0.01 \
#     --tta_n_iter 0 \
#     --per_sample_iter 1 \
#     --contrastive_alpha 0.7 \
#     --wandb_args "project=llava1.6_recursive_eval_1219,entity=VLM_Hallucination_Woohyeon,name=tta-topk-90-90-50-bilinear-norm-lr001-iter0-n1-alpha07" \
#     --save_output True \
#     --output_csv_path "./generation_output_pope_tta-topk-90-90-50-bilinear-norm-lr001-iter0-n1-alpha07.csv" \

# CUDA_VISIBLE_DEVICES=3 python3 -m accelerate.commands.launch \
#     --main_process_port 12345 \
#     --num_processes=1 \
#     -m lmms_eval \
#     --model llava \
#     --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
#     --tasks pope \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix llava_v1.6_pope \
#     --output_path ./logs/ \
#     --generation_type recursion \
#     --fix_grid 2x2 \
#     --attention_thresholding_type layer_mean_topk \
#     --attention_threshold "[0.9,0.9,0.6]" \
#     --positional_embedding_type bilinear_interpolation \
#     --remove_unpadding True \
#     --attn_norm norm \
#     --stages "-2" "-1" "0" "1" \
#     --verbosity DEBUG \
#     --square 1 \
#     --tta_learning_rate 0.01 \
#     --tta_n_iter 0 \
#     --per_sample_iter 1 \
#     --contrastive_alpha 0.7 \
#     --wandb_args "project=llava1.6_recursive_eval_1219,entity=VLM_Hallucination_Woohyeon,name=tta-topk-90-90-60-bilinear-norm-lr001-iter0-n1-alpha07" \
#     --save_output True \
#     --output_csv_path "./generation_output_pope_tta-topk-90-90-60-bilinear-norm-lr001-iter0-n1-alpha07.csv" \

# CUDA_VISIBLE_DEVICES=3 python3 -m accelerate.commands.launch \
#     --main_process_port 12345 \
#     --num_processes=1 \
#     -m lmms_eval \
#     --model llava \
#     --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
#     --tasks pope \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix llava_v1.6_pope \
#     --output_path ./logs/ \
#     --generation_type recursion \
#     --fix_grid 2x2 \
#     --attention_thresholding_type layer_mean_topk \
#     --attention_threshold "[0.9,0.9,0.7]" \
#     --positional_embedding_type bilinear_interpolation \
#     --remove_unpadding True \
#     --attn_norm norm \
#     --stages "-2" "-1" "0" "1" \
#     --verbosity DEBUG \
#     --square 1 \
#     --tta_learning_rate 0.01 \
#     --tta_n_iter 0 \
#     --per_sample_iter 1 \
#     --contrastive_alpha 0.7 \
#     --wandb_args "project=llava1.6_recursive_eval_1219,entity=VLM_Hallucination_Woohyeon,name=tta-topk-90-90-70-bilinear-norm-lr001-iter0-n1-alpha07" \
#     --save_output True \
#     --output_csv_path "./generation_output_pope_tta-topk-90-90-70-bilinear-norm-lr001-iter0-n1-alpha07.csv" \

# CUDA_VISIBLE_DEVICES=3 python3 -m accelerate.commands.launch \
#     --main_process_port 12345 \
#     --num_processes=1 \
#     -m lmms_eval \
#     --model llava \
#     --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
#     --tasks pope \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix llava_v1.6_pope \
#     --output_path ./logs/ \
#     --generation_type recursion \
#     --fix_grid 2x2 \
#     --attention_thresholding_type layer_mean_topk \
#     --attention_threshold "[0.9,0.9,0.4]" \
#     --positional_embedding_type bilinear_interpolation \
#     --remove_unpadding True \
#     --attn_norm norm \
#     --stages "-2" "-1" "0" "1" \
#     --verbosity DEBUG \
#     --square 1 \
#     --tta_learning_rate 0.01 \
#     --tta_n_iter 0 \
#     --per_sample_iter 1 \
#     --contrastive_alpha 0.7 \
#     --wandb_args "project=llava1.6_recursive_eval_1219,entity=VLM_Hallucination_Woohyeon,name=tta-topk-90-90-40-bilinear-norm-lr001-iter0-n1-alpha07" \
#     --save_output True \
#     --output_csv_path "./generation_output_pope_tta-topk-90-90-40-bilinear-norm-lr001-iter0-n1-alpha07.csv" \

# CUDA_VISIBLE_DEVICES=3 python3 -m accelerate.commands.launch \
#     --main_process_port 12345 \
#     --num_processes=1 \
#     -m lmms_eval \
#     --model llava \
#     --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
#     --tasks pope \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix llava_v1.6_pope \
#     --output_path ./logs/ \
#     --generation_type recursion \
#     --fix_grid 2x2 \
#     --attention_thresholding_type layer_mean_topk \
#     --attention_threshold "[0.9,0.9,0.5]" \
#     --positional_embedding_type bilinear_interpolation \
#     --remove_unpadding True \
#     --attn_norm norm \
#     --stages "-2" "-1" "0" "1" \
#     --verbosity DEBUG \
#     --square 1 \
#     --tta_learning_rate 0.01 \
#     --tta_n_iter 0 \
#     --per_sample_iter 1 \
#     --contrastive_alpha 0.7 \
#     --wandb_args "project=llava1.6_recursive_eval_1219,entity=VLM_Hallucination_Woohyeon,name=tta-topk-90-90-50-bilinear-norm-lr001-iter0-n1-alpha07" \
#     --save_output True \
#     --output_csv_path "./generation_output_pope_tta-topk-90-90-50-bilinear-norm-lr001-iter0-n1-alpha07.csv" \

# CUDA_VISIBLE_DEVICES=3 python3 -m accelerate.commands.launch \
#     --main_process_port 12345 \
#     --num_processes=1 \
#     -m lmms_eval \
#     --model llava \
#     --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
#     --tasks pope \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix llava_v1.6_pope \
#     --output_path ./logs/ \
#     --generation_type recursion \
#     --fix_grid 2x2 \
#     --attention_thresholding_type layer_mean_topk \
#     --attention_threshold "[0.9,0.9,0.6]" \
#     --positional_embedding_type bilinear_interpolation \
#     --remove_unpadding True \
#     --attn_norm norm \
#     --stages "-2" "-1" "0" "1" \
#     --verbosity DEBUG \
#     --square 1 \
#     --tta_learning_rate 0.01 \
#     --tta_n_iter 0 \
#     --per_sample_iter 1 \
#     --contrastive_alpha 0.7 \
#     --wandb_args "project=llava1.6_recursive_eval_1219,entity=VLM_Hallucination_Woohyeon,name=tta-topk-90-90-60-bilinear-norm-lr001-iter0-n1-alpha07" \
#     --save_output True \
#     --output_csv_path "./generation_output_pope_tta-topk-90-90-60-bilinear-norm-lr001-iter0-n1-alpha07.csv" \

# CUDA_VISIBLE_DEVICES=3 python3 -m accelerate.commands.launch \
#     --main_process_port 12345 \
#     --num_processes=1 \
#     -m lmms_eval \
#     --model llava \
#     --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
#     --tasks pope \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix llava_v1.6_pope \
#     --output_path ./logs/ \
#     --generation_type recursion \
#     --fix_grid 2x2 \
#     --attention_thresholding_type layer_mean_topk \
#     --attention_threshold "[0.9,0.9,0.7]" \
#     --positional_embedding_type bilinear_interpolation \
#     --remove_unpadding True \
#     --attn_norm norm \
#     --stages "-2" "-1" "0" "1" \
#     --verbosity DEBUG \
#     --square 1 \
#     --tta_learning_rate 0.01 \
#     --tta_n_iter 0 \
#     --per_sample_iter 1 \
#     --contrastive_alpha 0.7 \
#     --wandb_args "project=llava1.6_recursive_eval_1219,entity=VLM_Hallucination_Woohyeon,name=tta-topk-90-90-70-bilinear-norm-lr001-iter0-n1-alpha07" \
#     --save_output True \
#     --output_csv_path "./generation_output_pope_tta-topk-90-90-70-bilinear-norm-lr001-iter0-n1-alpha07.csv" \

CUDA_VISIBLE_DEVICES=3 python3 -m accelerate.commands.launch \
    --main_process_port 12345 \
    --num_processes=1 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
    --tasks pope \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.6_pope \
    --output_path ./logs/ \
    --generation_type recursion \
    --fix_grid 2x2 \
    --attention_thresholding_type layer_mean_topk \
    --attention_threshold "[0.9,0.9,0.8]" \
    --positional_embedding_type bilinear_interpolation \
    --remove_unpadding True \
    --attn_norm norm \
    --stages "-2" "-1" "0" "1" \
    --verbosity DEBUG \
    --square 1 \
    --tta_learning_rate 0.01 \
    --tta_n_iter 0 \
    --per_sample_iter 1 \
    --contrastive_alpha 0.9 \
    --wandb_args "project=llava1.6_recursive_eval_1219,entity=VLM_Hallucination_Woohyeon,name=tta-topk-90-90-80-bilinear-norm-lr001-iter0-n1-alpha09" \
    --save_output True \
    --output_csv_path "./generation_output_pope_tta-topk-90-90-80-bilinear-norm-lr001-iter0-n1-alpha09.csv" \

CUDA_VISIBLE_DEVICES=3 python3 -m accelerate.commands.launch \
    --main_process_port 12345 \
    --num_processes=1 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
    --tasks pope \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.6_pope \
    --output_path ./logs/ \
    --generation_type recursion \
    --fix_grid 2x2 \
    --attention_thresholding_type layer_mean_topk \
    --attention_threshold "[0.9,0.9,0.8]" \
    --positional_embedding_type bilinear_interpolation \
    --remove_unpadding True \
    --attn_norm norm \
    --stages "-2" "-1" "0" "1" \
    --verbosity DEBUG \
    --square 1 \
    --tta_learning_rate 0.01 \
    --tta_n_iter 0 \
    --per_sample_iter 1 \
    --contrastive_alpha 1.0 \
    --wandb_args "project=llava1.6_recursive_eval_1219,entity=VLM_Hallucination_Woohyeon,name=tta-topk-90-90-80-bilinear-norm-lr001-iter0-n1-alpha10" \
    --save_output True \
    --output_csv_path "./generation_output_pope_tta-topk-90-90-80-bilinear-norm-lr001-iter0-n1-alpha10.csv" \

# CUDA_VISIBLE_DEVICES=3 python3 -m accelerate.commands.launch \
#     --main_process_port 12345 \
#     --num_processes=1 \
#     -m lmms_eval \
#     --model llava \
#     --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
#     --tasks pope \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix llava_v1.6_pope \
#     --output_path ./logs/ \
#     --generation_type recursion \
#     --fix_grid 2x2 \
#     --attention_thresholding_type layer_mean_topk \
#     --attention_threshold "[0.9,0.9,0.9]" \
#     --positional_embedding_type bilinear_interpolation \
#     --remove_unpadding True \
#     --attn_norm norm \
#     --stages "-2" "-1" "0" "1" \
#     --verbosity DEBUG \
#     --square 1 \
#     --tta_learning_rate 0.01 \
#     --tta_n_iter 0 \
#     --per_sample_iter 1 \
#     --contrastive_alpha 0.7 \
#     --wandb_args "project=llava1.6_recursive_eval_1219,entity=VLM_Hallucination_Woohyeon,name=tta-topk-90-90-90-bilinear-norm-lr001-iter0-n1-alpha07" \
#     --save_output True \
#     --output_csv_path "./generation_output_pope_tta-topk-90-90-90-bilinear-norm-lr001-iter0-n1-alpha07.csv" \

# CUDA_VISIBLE_DEVICES=3 python3 -m accelerate.commands.launch \
#     --main_process_port 12345 \
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
#     --attention_thresholding_type layer_mean_topk \
#     --attention_threshold "[0.9,0.9,0.4]" \
#     --positional_embedding_type bilinear_interpolation \
#     --remove_unpadding True \
#     --attn_norm norm \
#     --stages "-2" "-1" "0" "1" \
#     --verbosity DEBUG \
#     --square 1 \
#     --tta_learning_rate 0.01 \
#     --tta_n_iter 0 \
#     --per_sample_iter 1 \
#     --contrastive_alpha 0.7 \
#     --wandb_args "project=llava1.6_recursive_eval_1219,entity=VLM_Hallucination_Woohyeon,name=tta-topk-90-90-40-bilinear-norm-lr001-iter0-n1-alpha07" \
#     --save_output True \
#     --output_csv_path "./generation_output_mme_tta-topk-90-90-40-bilinear-norm-lr001-iter0-n1-alpha07.csv" \

# CUDA_VISIBLE_DEVICES=3 python3 -m accelerate.commands.launch \
#     --main_process_port 12345 \
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
#     --attention_thresholding_type layer_mean_topk \
#     --attention_threshold "[0.9,0.9,0.5]" \
#     --positional_embedding_type bilinear_interpolation \
#     --remove_unpadding True \
#     --attn_norm norm \
#     --stages "-2" "-1" "0" "1" \
#     --verbosity DEBUG \
#     --square 1 \
#     --tta_learning_rate 0.01 \
#     --tta_n_iter 0 \
#     --per_sample_iter 1 \
#     --contrastive_alpha 0.7 \
#     --wandb_args "project=llava1.6_recursive_eval_1219,entity=VLM_Hallucination_Woohyeon,name=tta-topk-90-90-50-bilinear-norm-lr001-iter0-n1-alpha07" \
#     --save_output True \
#     --output_csv_path "./generation_output_mme_tta-topk-90-90-50-bilinear-norm-lr001-iter0-n1-alpha07.csv" \

# CUDA_VISIBLE_DEVICES=3 python3 -m accelerate.commands.launch \
#     --main_process_port 12345 \
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
#     --attention_thresholding_type layer_mean_topk \
#     --attention_threshold "[0.9,0.9,0.6]" \
#     --positional_embedding_type bilinear_interpolation \
#     --remove_unpadding True \
#     --attn_norm norm \
#     --stages "-2" "-1" "0" "1" \
#     --verbosity DEBUG \
#     --square 1 \
#     --tta_learning_rate 0.01 \
#     --tta_n_iter 0 \
#     --per_sample_iter 1 \
#     --contrastive_alpha 0.7 \
#     --wandb_args "project=llava1.6_recursive_eval_1219,entity=VLM_Hallucination_Woohyeon,name=tta-topk-90-90-60-bilinear-norm-lr001-iter0-n1-alpha07" \
#     --save_output True \
#     --output_csv_path "./generation_output_mme_tta-topk-90-90-60-bilinear-norm-lr001-iter0-n1-alpha07.csv" \

# CUDA_VISIBLE_DEVICES=3 python3 -m accelerate.commands.launch \
#     --main_process_port 12345 \
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
#     --attention_thresholding_type layer_mean_topk \
#     --attention_threshold "[0.9,0.9,0.7]" \
#     --positional_embedding_type bilinear_interpolation \
#     --remove_unpadding True \
#     --attn_norm norm \
#     --stages "-2" "-1" "0" "1" \
#     --verbosity DEBUG \
#     --square 1 \
#     --tta_learning_rate 0.01 \
#     --tta_n_iter 0 \
#     --per_sample_iter 1 \
#     --contrastive_alpha 0.7 \
#     --wandb_args "project=llava1.6_recursive_eval_1219,entity=VLM_Hallucination_Woohyeon,name=tta-topk-90-90-70-bilinear-norm-lr001-iter0-n1-alpha07" \
#     --save_output True \
#     --output_csv_path "./generation_output_mme_tta-topk-90-90-70-bilinear-norm-lr001-iter0-n1-alpha07.csv" \

# CUDA_VISIBLE_DEVICES=3 python3 -m accelerate.commands.launch \
#     --main_process_port 12345 \
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
#     --attention_thresholding_type layer_mean_topk \
#     --attention_threshold "[0.9,0.9,0.4]" \
#     --positional_embedding_type bilinear_interpolation \
#     --remove_unpadding True \
#     --attn_norm norm \
#     --stages "-2" "-1" "0" "1" \
#     --verbosity DEBUG \
#     --square 1 \
#     --tta_learning_rate 0.01 \
#     --tta_n_iter 0 \
#     --per_sample_iter 1 \
#     --contrastive_alpha 0.7 \
#     --wandb_args "project=llava1.6_recursive_eval_1219,entity=VLM_Hallucination_Woohyeon,name=tta-topk-90-90-40-bilinear-norm-lr001-iter0-n1-alpha07" \
#     --save_output True \
#     --output_csv_path "./generation_output_mme_tta-topk-90-90-40-bilinear-norm-lr001-iter0-n1-alpha07.csv" \

# CUDA_VISIBLE_DEVICES=3 python3 -m accelerate.commands.launch \
#     --main_process_port 12345 \
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
#     --attention_thresholding_type layer_mean_topk \
#     --attention_threshold "[0.9,0.9,0.5]" \
#     --positional_embedding_type bilinear_interpolation \
#     --remove_unpadding True \
#     --attn_norm norm \
#     --stages "-2" "-1" "0" "1" \
#     --verbosity DEBUG \
#     --square 1 \
#     --tta_learning_rate 0.01 \
#     --tta_n_iter 0 \
#     --per_sample_iter 1 \
#     --contrastive_alpha 0.7 \
#     --wandb_args "project=llava1.6_recursive_eval_1219,entity=VLM_Hallucination_Woohyeon,name=tta-topk-90-90-50-bilinear-norm-lr001-iter0-n1-alpha07" \
#     --save_output True \
#     --output_csv_path "./generation_output_mme_tta-topk-90-90-50-bilinear-norm-lr001-iter0-n1-alpha07.csv" \

# CUDA_VISIBLE_DEVICES=3 python3 -m accelerate.commands.launch \
#     --main_process_port 12345 \
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
#     --attention_thresholding_type layer_mean_topk \
#     --attention_threshold "[0.9,0.9,0.6]" \
#     --positional_embedding_type bilinear_interpolation \
#     --remove_unpadding True \
#     --attn_norm norm \
#     --stages "-2" "-1" "0" "1" \
#     --verbosity DEBUG \
#     --square 1 \
#     --tta_learning_rate 0.01 \
#     --tta_n_iter 0 \
#     --per_sample_iter 1 \
#     --contrastive_alpha 0.7 \
#     --wandb_args "project=llava1.6_recursive_eval_1219,entity=VLM_Hallucination_Woohyeon,name=tta-topk-90-90-60-bilinear-norm-lr001-iter0-n1-alpha07" \
#     --save_output True \
#     --output_csv_path "./generation_output_mme_tta-topk-90-90-60-bilinear-norm-lr001-iter0-n1-alpha07.csv" \

CUDA_VISIBLE_DEVICES=3 python3 -m accelerate.commands.launch \
    --main_process_port 12345 \
    --num_processes=1 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
    --tasks mme \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.6_mme \
    --output_path ./logs/ \
    --generation_type recursion \
    --fix_grid 2x2 \
    --attention_thresholding_type layer_mean_topk \
    --attention_threshold "[0.0,0.0,0.0]" \
    --positional_embedding_type bilinear_interpolation \
    --remove_unpadding True \
    --attn_norm norm \
    --stages "-2" "-1" "0" "1" \
    --verbosity DEBUG \
    --square 1 \
    --tta_learning_rate 0.01 \
    --tta_n_iter 0 \
    --per_sample_iter 1 \
    --contrastive_alpha 0.0 \
    --wandb_args "project=llava1.6_recursive_eval_1219,entity=VLM_Hallucination_Woohyeon,name=tta-topk-0-0-0-bilinear-norm-lr001-iter0-n1" \

CUDA_VISIBLE_DEVICES=3 python3 -m accelerate.commands.launch \
    --main_process_port 12345 \
    --num_processes=1 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
    --tasks mme \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.6_mme \
    --output_path ./logs/ \
    --generation_type recursion \
    --fix_grid 2x2 \
    --attention_thresholding_type layer_mean_topk \
    --attention_threshold "[1.0,0.0,0.0]" \
    --positional_embedding_type bilinear_interpolation \
    --remove_unpadding True \
    --attn_norm norm \
    --stages "-2" "-1" "0" "1" \
    --verbosity DEBUG \
    --square 1 \
    --tta_learning_rate 0.01 \
    --tta_n_iter 0 \
    --per_sample_iter 1 \
    --contrastive_alpha 0.0 \
    --wandb_args "project=llava1.6_recursive_eval_1219,entity=VLM_Hallucination_Woohyeon,name=tta-topk-100-0-0-bilinear-norm-lr001-iter0-n1" \  

CUDA_VISIBLE_DEVICES=3 python3 -m accelerate.commands.launch \
    --main_process_port 12345 \
    --num_processes=1 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
    --tasks mme \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.6_mme \
    --output_path ./logs/ \
    --generation_type recursion \
    --fix_grid 2x2 \
    --attention_thresholding_type layer_mean_topk \
    --attention_threshold "[1.0,1.0,0.0]" \
    --positional_embedding_type bilinear_interpolation \
    --remove_unpadding True \
    --attn_norm norm \
    --stages "-2" "-1" "0" "1" \
    --verbosity DEBUG \
    --square 1 \
    --tta_learning_rate 0.01 \
    --tta_n_iter 0 \
    --per_sample_iter 1 \
    --contrastive_alpha 0.0 \
    --wandb_args "project=llava1.6_recursive_eval_1219,entity=VLM_Hallucination_Woohyeon,name=tta-topk-100-100-0-bilinear-norm-lr001-iter0-n1-alpha07" \
    --save_output True \
    --output_csv_path "./generation_output_mme_tta-topk-90-90-80-bilinear-norm-lr001-iter0-n1-alpha07.csv" \

CUDA_VISIBLE_DEVICES=3 python3 -m accelerate.commands.launch \
    --main_process_port 12345 \
    --num_processes=1 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
    --tasks mme \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.6_mme \
    --output_path ./logs/ \
    --generation_type recursion \
    --fix_grid 2x2 \
    --attention_thresholding_type layer_mean_topk \
    --attention_threshold "[1.0,1.0,1.0]" \
    --positional_embedding_type bilinear_interpolation \
    --remove_unpadding True \
    --attn_norm norm \
    --stages "-2" "-1" "0" "1" \
    --verbosity DEBUG \
    --square 1 \
    --tta_learning_rate 0.01 \
    --tta_n_iter 0 \
    --per_sample_iter 1 \
    --contrastive_alpha 0.0 \
    --wandb_args "project=llava1.6_recursive_eval_1219,entity=VLM_Hallucination_Woohyeon,name=tta-topk-100-100-100-bilinear-norm-lr001-iter0-n1-alpha07" \
    --save_output True \
    --output_csv_path "./generation_output_mme_tta-topk-90-90-90-bilinear-norm-lr001-iter0-n1-alpha07.csv" \

CUDA_VISIBLE_DEVICES=3 python3 -m accelerate.commands.launch \
    --main_process_port 12345 \
    --num_processes=1 \
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
    --attention_thresholding_type layer_mean_topk \
    --attention_threshold "[0.9,0.9,0.8]" \
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
    --wandb_args "project=llava1.6_recursive_eval_1219,entity=VLM_Hallucination_Woohyeon,name=tta-topk-90-90-80-bilinear-norm-lr001-iter0-n1-alpha07" \
    --save_output True \
    --output_csv_path "./generation_output_pope_tta-topk-90-90-80-bilinear-norm-lr001-iter0-n1-alpha07.csv" \

CUDA_VISIBLE_DEVICES=3 python3 -m accelerate.commands.launch \
    --main_process_port 12345 \
    --num_processes=1 \
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
    --attention_thresholding_type layer_mean_topk \
    --attention_threshold "[0.9,0.9,0.7]" \
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
    --wandb_args "project=llava1.6_recursive_eval_1219,entity=VLM_Hallucination_Woohyeon,name=tta-topk-90-90-70-bilinear-norm-lr001-iter0-n1-alpha07" \
    --save_output True \
    --output_csv_path "./generation_output_pope_tta-topk-90-90-70-bilinear-norm-lr001-iter0-n1-alpha07.csv" \

CUDA_VISIBLE_DEVICES=3 python3 -m accelerate.commands.launch \
    --main_process_port 12345 \
    --num_processes=1 \
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
    --attention_thresholding_type layer_mean_topk \
    --attention_threshold "[0.9,0.9,0.6]" \
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
    --wandb_args "project=llava1.6_recursive_eval_1219,entity=VLM_Hallucination_Woohyeon,name=tta-topk-90-90-60-bilinear-norm-lr001-iter0-n1-alpha07" \
    --save_output True \
    --output_csv_path "./generation_output_pope_tta-topk-90-90-60-bilinear-norm-lr001-iter0-n1-alpha07.csv" \

CUDA_VISIBLE_DEVICES=3 python3 -m accelerate.commands.launch \
    --main_process_port 12345 \
    --num_processes=1 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
    --tasks pope \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.6_pope \
    --output_path ./logs/ \
    --generation_type recursion \
    --fix_grid 2x2 \
    --attention_thresholding_type layer_mean_topk \
    --attention_threshold "[1.0,1.0,0.6]" \
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
    --wandb_args "project=llava1.6_recursive_eval_1219,entity=VLM_Hallucination_Woohyeon,name=tta-topk-100-100-60-bilinear-norm-lr001-iter0-n1-alpha07" \
    --save_output True \
    --output_csv_path "./generation_output_pope_tta-topk-100-100-60-bilinear-norm-lr001-iter0-n1-alpha07.csv" \

CUDA_VISIBLE_DEVICES=3 python3 -m accelerate.commands.launch \
    --main_process_port 12345 \
    --num_processes=1 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
    --tasks pope \
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
    --output_csv_path "./generation_output_pope_tta-topk-100-100-70-bilinear-norm-lr001-iter0-n1-alpha07.csv" \

CUDA_VISIBLE_DEVICES=3 python3 -m accelerate.commands.launch \
    --main_process_port 12345 \
    --num_processes=1 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
    --tasks pope \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.6_pope \
    --output_path ./logs/ \
    --generation_type recursion \
    --fix_grid 2x2 \
    --attention_thresholding_type layer_mean_topk \
    --attention_threshold "[1.0,1.0,0.8]" \
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
    --wandb_args "project=llava1.6_recursive_eval_1219,entity=VLM_Hallucination_Woohyeon,name=tta-topk-100-100-80-bilinear-norm-lr001-iter0-n1-alpha07" \
    --save_output True \
    --output_csv_path "./generation_output_pope_tta-topk-100-100-80-bilinear-norm-lr001-iter0-n1-alpha07.csv" \




