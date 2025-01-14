# CUDA_VISIBLE_DEVICES=0,1,2 python3 -m accelerate.commands.launch \
#     --num_processes=3 \
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
#     --attention_threshold "[7.0,7.0,7.0,7.0]" \
#     --positional_embedding_type bilinear_interpolation \
#     --remove_unpadding True \
#     --attn_norm None \
#     --stages "-2" "-1" "0" "1" \
#     --verbosity DEBUG \
#     --square 1 \
#     --contrastive_alphas "1.0" "1.0" "1.0" \
#     --wandb_args "project=llava1.6_recursive_eval_jake_cd_adp,entity=VLM_Hallucination_Woohyeon,name=star_7.0_cd_adaptive_exp_1.0"

CUDA_VISIBLE_DEVICES=0,1,2 python3 -m accelerate.commands.launch \
    --num_processes=3 \
    --main_process_port 29881 \
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
    --attention_threshold "[7.0,7.0,7.0,7.0]" \
    --positional_embedding_type bilinear_interpolation \
    --remove_unpadding True \
    --attn_norm None \
    --stages "-2" "-1" "0" "1" \
    --verbosity DEBUG \
    --square 1 \
    --contrastive_alphas "0.5" "0.5" "0.5" \
    --wandb_args "project=llava1.6_recursive_eval_jake_cd_adp,entity=VLM_Hallucination_Woohyeon,name=star_7.0_cd_adaptive_exp_0.5"

CUDA_VISIBLE_DEVICES=0,1,2 python3 -m accelerate.commands.launch \
    --num_processes=3 \
    --main_process_port 29881 \
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
    --attention_threshold "[7.0,7.0,7.0,7.0]" \
    --positional_embedding_type bilinear_interpolation \
    --remove_unpadding True \
    --attn_norm None \
    --stages "-2" "-1" "0" "1" \
    --verbosity DEBUG \
    --square 1 \
    --contrastive_alphas "0.4" "0.4" "0.4" \
    --wandb_args "project=llava1.6_recursive_eval_jake_cd_adp,entity=VLM_Hallucination_Woohyeon,name=star_7.0_cd_adaptive_exp_0.4"

CUDA_VISIBLE_DEVICES=0,1,2 python3 -m accelerate.commands.launch \
    --num_processes=3 \
    --main_process_port 29881 \
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
    --attention_threshold "[7.0,7.0,7.0,7.0]" \
    --positional_embedding_type bilinear_interpolation \
    --remove_unpadding True \
    --attn_norm None \
    --stages "-2" "-1" "0" "1" \
    --verbosity DEBUG \
    --square 1 \
    --contrastive_alphas "0.3" "0.3" "0.3" \
    --wandb_args "project=llava1.6_recursive_eval_jake_cd_adp,entity=VLM_Hallucination_Woohyeon,name=star_7.0_cd_adaptive_exp_0.3"

CUDA_VISIBLE_DEVICES=0,1,2 python3 -m accelerate.commands.launch \
    --num_processes=3 \
    --main_process_port 29881 \
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
    --attention_threshold "[7.0,7.0,7.0,7.0]" \
    --positional_embedding_type bilinear_interpolation \
    --remove_unpadding True \
    --attn_norm None \
    --stages "-2" "-1" "0" "1" \
    --verbosity DEBUG \
    --square 1 \
    --contrastive_alphas "0.2" "0.2" "0.2" \
    --wandb_args "project=llava1.6_recursive_eval_jake_cd_adp,entity=VLM_Hallucination_Woohyeon,name=star_7.0_cd_adaptive_exp_0.2"

CUDA_VISIBLE_DEVICES=0,1,2 python3 -m accelerate.commands.launch \
    --num_processes=3 \
    --main_process_port 29881 \
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
    --attention_threshold "[7.0,7.0,7.0,7.0]" \
    --positional_embedding_type bilinear_interpolation \
    --remove_unpadding True \
    --attn_norm None \
    --stages "-2" "-1" "0" "1" \
    --verbosity DEBUG \
    --square 1 \
    --contrastive_alphas "0.1" "0.1" "0.1" \
    --wandb_args "project=llava1.6_recursive_eval_jake_cd_adp,entity=VLM_Hallucination_Woohyeon,name=star_7.0_cd_adaptive_exp_0.1"

