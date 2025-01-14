# CUDA_VISIBLE_DEVICES=2,3 python3 -m accelerate.commands.launch \
#     --num_processes=2 \
#     --main_process_port 29885 \
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
#     --contrastive_alphas "2.0" "2.0" "2.0" \
#     --wandb_args "project=llava1.6_recursive_eval_jake_cd_adp,entity=VLM_Hallucination_Woohyeon,name=mmbench_2.0_cd_adaptive_2.0"

CUDA_VISIBLE_DEVICES=2,3 python3 -m accelerate.commands.launch \
    --num_processes=2 \
    --main_process_port 29885 \
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
    --contrastive_alphas "1.0" "1.0" "1.0" \
    --wandb_args "project=llava1.6_recursive_eval_jake_cd_adp,entity=VLM_Hallucination_Woohyeon,name=mmbench_2.0_cd_harmonic_adaptive_1.0"

# CUDA_VISIBLE_DEVICES=2,3 python3 -m accelerate.commands.launch \
#     --num_processes=2 \
#     --main_process_port 29881 \
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
#     --contrastive_alphas "0.6" "0.6" "0.6" \
#     --wandb_args "project=llava1.6_recursive_eval_jake_cd_adp,entity=VLM_Hallucination_Woohyeon,name=mmbench_2.0_cd_adaptive_0.6"

# CUDA_VISIBLE_DEVICES=2,3 python3 -m accelerate.commands.launch \
#     --num_processes=2 \
#     --main_process_port 29881 \
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
#     --contrastive_alphas "0.5" "0.5" "0.5" \
#     --wandb_args "project=llava1.6_recursive_eval_jake_cd_adp,entity=VLM_Hallucination_Woohyeon,name=mmbench_2.0_cd_adaptive_0.5"

# CUDA_VISIBLE_DEVICES=2,3 python3 -m accelerate.commands.launch \
#     --num_processes=2 \
#     --main_process_port 29881 \
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
#     --contrastive_alphas "0.4" "0.4" "0.4" \
#     --wandb_args "project=llava1.6_recursive_eval_jake_cd_adp,entity=VLM_Hallucination_Woohyeon,name=mmbench_2.0_cd_adaptive_0.4"

# CUDA_VISIBLE_DEVICES=2,3 python3 -m accelerate.commands.launch \
#     --num_processes=2 \
#     --main_process_port 29881 \
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
#     --contrastive_alphas "0.3" "0.3" "0.3" \
#     --wandb_args "project=llava1.6_recursive_eval_jake_cd_adp,entity=VLM_Hallucination_Woohyeon,name=mmbench_2.0_cd_adaptive_0.3"

# CUDA_VISIBLE_DEVICES=2,3 python3 -m accelerate.commands.launch \
#     --num_processes=2 \
#     --main_process_port 29881 \
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
#     --contrastive_alphas "0.2" "0.2" "0.2" \
#     --wandb_args "project=llava1.6_recursive_eval_jake_cd_adp,entity=VLM_Hallucination_Woohyeon,name=mmbench_2.0_cd_adaptive_0.2"

# CUDA_VISIBLE_DEVICES=2,3 python3 -m accelerate.commands.launch \
#     --num_processes=2 \
#     --main_process_port 29881 \
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
#     --contrastive_alphas "0.1" "0.1" "0.1" \
#     --wandb_args "project=llava1.6_recursive_eval_jake_cd_adp,entity=VLM_Hallucination_Woohyeon,name=mmbench_2.0_cd_adaptive_0.1"

