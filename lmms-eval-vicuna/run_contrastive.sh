# CUDA_VISIBLE_DEVICES=1 python3 -m accelerate.commands.launch \
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
#     --attention_threshold "[7.0,7.0,7.0]" \
#     --positional_embedding_type bilinear_interpolation \
#     --remove_unpadding True \
#     --attn_norm None \
#     --stages "-2" "-1" "0" "1" \
#     --verbosity DEBUG \
#     --square 1 \
#     --contrastive_alphas "0.8" "0.8" "0.8" \
#     --cd_strategy default \
#     --wandb_args "project=llava1.6_recursive_eval_7b_vicuna_cd,entity=VLM_Hallucination_Woohyeon,name=mmstar_70_bilinear_s1_cd_08_default"

# CUDA_VISIBLE_DEVICES=1 python3 -m accelerate.commands.launch \
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
#     --attention_threshold "[7.0,7.0,7.0]" \
#     --positional_embedding_type bilinear_interpolation \
#     --remove_unpadding True \
#     --attn_norm None \
#     --stages "-2" "-1" "0" "1" \
#     --verbosity DEBUG \
#     --square 1 \
#     --contrastive_alphas "0.8" "0.8" "0.8" \
#     --cd_strategy confidence \
#     --wandb_args "project=llava1.6_recursive_eval_7b_vicuna_cd,entity=VLM_Hallucination_Woohyeon,name=mmstar_70_bilinear_s1_cd_08_confidence_adaptive"

# CUDA_VISIBLE_DEVICES=1 python3 -m accelerate.commands.launch \
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
#     --attention_threshold "[7.0,7.0,7.0]" \
#     --positional_embedding_type bilinear_interpolation \
#     --remove_unpadding True \
#     --attn_norm None \
#     --stages "-2" "-1" "0" "1" \
#     --verbosity DEBUG \
#     --square 1 \
#     --contrastive_alphas "0.8" "0.8" "0.8" \
#     --cd_strategy confidence \
#     --wandb_args "project=llava1.6_recursive_eval_7b_vicuna_cd,entity=VLM_Hallucination_Woohyeon,name=mmstar_70_bilinear_s1_cd_08_code_adaptive"

# CUDA_VISIBLE_DEVICES=1 python3 -m accelerate.commands.launch \
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
#     --attention_threshold "[7.0,7.0,7.0]" \
#     --positional_embedding_type bilinear_interpolation \
#     --remove_unpadding True \
#     --attn_norm None \
#     --stages "-2" "-1" "0" "1" \
#     --verbosity DEBUG \
#     --square 1 \
#     --contrastive_alphas "0.8" "0.8" "0.8" \
#     --cd_strategy entropy \
#     --wandb_args "project=llava1.6_recursive_eval_7b_vicuna_cd,entity=VLM_Hallucination_Woohyeon,name=mmstar_70_bilinear_s1_cd_08_entropy_adaptive"

# CUDA_VISIBLE_DEVICES=1 python3 -m accelerate.commands.launch \
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
#     --attention_threshold "[7.0,7.0,7.0]" \
#     --positional_embedding_type bilinear_interpolation \
#     --remove_unpadding True \
#     --attn_norm None \
#     --stages "-2" "-1" "0" "1" \
#     --verbosity DEBUG \
#     --square 1 \
#     --contrastive_alphas "0.8" "0.8" "0.8" \
#     --cd_strategy jensen_shannon \
#     --wandb_args "project=llava1.6_recursive_eval_7b_vicuna_cd,entity=VLM_Hallucination_Woohyeon,name=mmstar_70_bilinear_s1_cd_08_jensen_shannon_adaptive"

# CUDA_VISIBLE_DEVICES=1 python3 -m accelerate.commands.launch \
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
#     --attention_threshold "[7.0,7.0,7.0]" \
#     --positional_embedding_type bilinear_interpolation \
#     --remove_unpadding True \
#     --attn_norm None \
#     --stages "-2" "-1" "0" "1" \
#     --verbosity DEBUG \
#     --square 1 \
#     --contrastive_alphas "0.8" "0.8" "0.8" \
#     --cd_strategy wasserstein \
#     --wandb_args "project=llava1.6_recursive_eval_7b_vicuna_cd,entity=VLM_Hallucination_Woohyeon,name=mmstar_70_bilinear_s1_cd_08_wasserstein_adaptive"

# CUDA_VISIBLE_DEVICES=1 python3 -m accelerate.commands.launch \
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
#     --attention_threshold "[7.0,7.0,7.0]" \
#     --positional_embedding_type bilinear_interpolation \
#     --remove_unpadding True \
#     --attn_norm None \
#     --stages "-2" "-1" "0" "1" \
#     --verbosity DEBUG \
#     --square 1 \
#     --contrastive_alphas "0.8" "0.8" "0.8" \
#     --cd_strategy hellinger \
#     --wandb_args "project=llava1.6_recursive_eval_7b_vicuna_cd,entity=VLM_Hallucination_Woohyeon,name=mmstar_70_bilinear_s1_cd_08_hellinger_adaptive"

# CUDA_VISIBLE_DEVICES=1 python3 -m accelerate.commands.launch \
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
#     --attention_threshold "[7.0,7.0,7.0]" \
#     --positional_embedding_type bilinear_interpolation \
#     --remove_unpadding True \
#     --attn_norm None \
#     --stages "-2" "-1" "0" "1" \
#     --verbosity DEBUG \
#     --square 1 \
#     --contrastive_alphas "0.8" "0.8" "0.8" \
#     --cd_strategy euclidean \
#     --wandb_args "project=llava1.6_recursive_eval_7b_vicuna_cd,entity=VLM_Hallucination_Woohyeon,name=mmstar_70_bilinear_s1_cd_08_euclidean_adaptive"

CUDA_VISIBLE_DEVICES=1 python3 -m accelerate.commands.launch \
    --num_processes=1 \
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
    --attention_threshold "[10.0,10.0,10.0]" \
    --positional_embedding_type bilinear_interpolation \
    --remove_unpadding True \
    --attn_norm None \
    --stages "-2" "-1" "0" "1" \
    --verbosity DEBUG \
    --square 1 \
    --contrastive_alphas "1.6" "2.0" "0.9" \
    --cd_strategy default \
    --wandb_args "project=llava1.6_recursive_eval_7b_vicuna_cd,entity=VLM_Hallucination_Woohyeon,name=mmstar_0.7_bilinear_s1_cd_0.9-2.0-1.6_default_cut_0.2"