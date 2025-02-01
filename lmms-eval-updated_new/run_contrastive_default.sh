# CUDA_VISIBLE_DEVICES=2 python3 -m accelerate.commands.launch \
#     --num_processes=1 \
#     --main_process_port 29881 \
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
#     --attention_threshold "[0.7,0.7,0.7]" \
#     --positional_embedding_type bilinear_interpolation \
#     --remove_unpadding True \
#     --attn_norm None \
#     --stages "-2" "-1" "0" "1" \
#     --verbosity DEBUG \
#     --square 1 \
#     --contrastive_alphas "1.0" "1.0" "0.9" \
#     --cd_strategy default \
#     --wandb_args "project=llava1.6_recursive_eval_7b_vicuna_cd,entity=VLM_Hallucination_Woohyeon,name=mmstar_07_bilinear_s1_cd_09_10_10_default"

CUDA_VISIBLE_DEVICES=3 python3 -m accelerate.commands.launch \
    --num_processes=1 \
    --main_process_port 29881 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
    --tasks mmbench_en_dev \
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
    --contrastive_alphas "1.1" "1.4" "0.0" \
    --cd_strategy default \
    --wandb_args "project=llava1.6_recursive_eval_7b_vicuna_cd,entity=VLM_Hallucination_Woohyeon,name=mmbench_20_bilinear_s1_cd_0.0_1.4_1.1"

# CUDA_VISIBLE_DEVICES=2 python3 -m accelerate.commands.launch \
#     --num_processes=1 \
#     --main_process_port 29881 \
#     -m lmms_eval \
#     --model llava \
#     --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
#     --tasks pope_pop \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix llava_v1.6_pope \
#     --output_path ./logs/ \
#     --generation_type recursion \
#     --fix_grid 2x2 \
#     --attention_thresholding_type attn_topk \
#     --attention_threshold "[2.5,2.5,2.5]" \
#     --positional_embedding_type bilinear_interpolation \
#     --remove_unpadding True \
#     --attn_norm None \
#     --stages "-2" "-1" "0" "1" \
#     --verbosity DEBUG \
#     --square 1 \
#     --contrastive_alphas "1.0" "0.7" "0.7" \
#     --cd_strategy default \
#     --wandb_args "project=llava1.6_recursive_eval_7b_vicuna_cd,entity=VLM_Hallucination_Woohyeon,name=pope_25_bilinear_s1_cd_07_07_10_default"

# CUDA_VISIBLE_DEVICES=2 python3 -m accelerate.commands.launch \
#     --num_processes=1 \
#     --main_process_port 29881 \
#     -m lmms_eval \
#     --model llava \
#     --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
#     --tasks pope_gqa_pop \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix llava_v1.6_pope \
#     --output_path ./logs/ \
#     --generation_type recursion \
#     --fix_grid 2x2 \
#     --attention_thresholding_type attn_topk \
#     --attention_threshold "[2.5,2.5,2.5]" \
#     --positional_embedding_type bilinear_interpolation \
#     --remove_unpadding True \
#     --attn_norm None \
#     --stages "-2" "-1" "0" "1" \
#     --verbosity DEBUG \
#     --square 1 \
#     --contrastive_alphas "1.0" "0.7" "1.0" \
#     --cd_strategy default \
#     --wandb_args "project=llava1.6_recursive_eval_7b_vicuna_cd,entity=VLM_Hallucination_Woohyeon,name=pope_gqa_25_bilinear_s1_cd_10_07_10_default"

# CUDA_VISIBLE_DEVICES=2 python3 -m accelerate.commands.launch \
#     --num_processes=1 \
#     --main_process_port 29881 \
#     -m lmms_eval \
#     --model llava \
#     --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
#     --tasks pope_aokvqa_pop \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix llava_v1.6_pope \
#     --output_path ./logs/ \
#     --generation_type recursion \
#     --fix_grid 2x2 \
#     --attention_thresholding_type attn_topk \
#     --attention_threshold "[2.5,2.5,2.5]" \
#     --positional_embedding_type bilinear_interpolation \
#     --remove_unpadding True \
#     --attn_norm None \
#     --stages "-2" "-1" "0" "1" \
#     --verbosity DEBUG \
#     --square 1 \
#     --contrastive_alphas "0.4" "0.7" "0.3" \
#     --cd_strategy default \
#     --wandb_args "project=llava1.6_recursive_eval_7b_vicuna_cd,entity=VLM_Hallucination_Woohyeon,name=pope_aokvqa_25_bilinear_s1_cd_03_07_04_default"