CUDA_VISIBLE_DEVICES=2 python3 -m accelerate.commands.launch \
    --num_processes=1 \
    --main_process_port 29882 \
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
    --attention_threshold "[1.0,0.7,0.5]" \
    --positional_embedding_type bilinear_interpolation \
    --remove_unpadding True \
    --attn_norm None \
    --stages "-2" "-1" "0" "1" \
    --verbosity DEBUG \
    --square 1 \
    --contrastive_alphas "0.3" "0.3" "0.3" \
    --wandb_args "project=llava1.6_recursive_eval_7b_vicuna_cd,entity=VLM_Hallucination_Woohyeon,name=mmstar_100_70_50_bilinear_s1_cd_code_adaptive_03"

CUDA_VISIBLE_DEVICES=2 python3 -m accelerate.commands.launch \
    --num_processes=1 \
    --main_process_port 29882 \
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
    --attention_threshold "[1.0,0.7,0.5]" \
    --positional_embedding_type bilinear_interpolation \
    --remove_unpadding True \
    --attn_norm None \
    --stages "-2" "-1" "0" "1" \
    --verbosity DEBUG \
    --square 1 \
    --contrastive_alphas "0.4" "0.4" "0.4" \
    --wandb_args "project=llava1.6_recursive_eval_7b_vicuna_cd,entity=VLM_Hallucination_Woohyeon,name=mmstar_100_70_50_bilinear_s1_cd_code_adaptive_04"

CUDA_VISIBLE_DEVICES=2 python3 -m accelerate.commands.launch \
    --num_processes=1 \
    --main_process_port 29882 \
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
    --attention_threshold "[1.0,0.7,0.5]" \
    --positional_embedding_type bilinear_interpolation \
    --remove_unpadding True \
    --attn_norm None \
    --stages "-2" "-1" "0" "1" \
    --verbosity DEBUG \
    --square 1 \
    --contrastive_alphas "0.5" "0.5" "0.5" \
    --wandb_args "project=llava1.6_recursive_eval_7b_vicuna_cd,entity=VLM_Hallucination_Woohyeon,name=mmstar_100_70_50_bilinear_s1_cd_code_adaptive_05"

CUDA_VISIBLE_DEVICES=2 python3 -m accelerate.commands.launch \
    --num_processes=1 \
    --main_process_port 29882 \
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
    --attention_threshold "[1.0,0.7,0.5]" \
    --positional_embedding_type bilinear_interpolation \
    --remove_unpadding True \
    --attn_norm None \
    --stages "-2" "-1" "0" "1" \
    --verbosity DEBUG \
    --square 1 \
    --contrastive_alphas "0.6" "0.6" "0.6" \
    --wandb_args "project=llava1.6_recursive_eval_7b_vicuna_cd,entity=VLM_Hallucination_Woohyeon,name=mmstar_100_70_50_bilinear_s1_cd_code_adaptive_06"

CUDA_VISIBLE_DEVICES=2 python3 -m accelerate.commands.launch \
    --num_processes=1 \
    --main_process_port 29882 \
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
    --attention_threshold "[1.0,0.7,0.5]" \
    --positional_embedding_type bilinear_interpolation \
    --remove_unpadding True \
    --attn_norm None \
    --stages "-2" "-1" "0" "1" \
    --verbosity DEBUG \
    --square 1 \
    --contrastive_alphas "0.1" "0.1" "0.1" \
    --wandb_args "project=llava1.6_recursive_eval_7b_vicuna_cd,entity=VLM_Hallucination_Woohyeon,name=mmstar_100_70_50_bilinear_s1_cd_code_adaptive_01"





