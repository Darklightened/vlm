#!/bin/bash


## generation_type: [recursion, recursion_retain_base, default, base-only, split-only]
# string "recusion" must be in generation_type for recursive generation

## fix_grid: [default, 2x2] - 2x2: force the image to be square
## attention_thresholding_type: [layer_mean]
## remove_unpadding: set True to remove unpadding and set mm_patch_merge_type='spatial' else 'spatial_unpad'
## regenerate_condition: currently "all": always regenerate when type is "recursion"
# python3 -m accelerate.commands.launch \
#     --num_processes=1 \
#     -m lmms_eval \
#     --model llava \
#     --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
#     --tasks pope,mme \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix llava_v1.6_pope \
#     --output_path ./logs/ \
#     --generation_type recursion \
#     --fix_grid 2x2 \
#     --attention_thresholding_type layer_mean_with_top_k\
#     --attention_threshold 0.3 \
#     --remove_unpadding True \
#     --regenerate_condition all \
#     --verbosity DEBUG \
#     --positional_embedding_type reduced_skip_cls \
#     --wandb_args "project=llava1.6_recursive_eval_Jake2,entity=VLM_Hallucination_Woohyeon,name=llava-v1.6-7b-84-168-336-topk30_reduced_skip_cls"

# python3 -m accelerate.commands.launch \
#     --num_processes=1 \
#     -m lmms_eval \
#     --model llava \
#     --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
#     --tasks pope,mme \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix llava_v1.6_pope \
#     --output_path ./logs/ \
#     --generation_type recursion \
#     --fix_grid 2x2 \
#     --attention_thresholding_type layer_mean_with_top_k\
#     --attention_threshold 0.2 \
#     --remove_unpadding True \
#     --regenerate_condition all \
#     --verbosity DEBUG \
#     --positional_embedding_type reduced_skip_cls \
#     --wandb_args "project=llava1.6_recursive_eval_Jake2,entity=VLM_Hallucination_Woohyeon,name=llava-v1.6-7b-84-168-336-topk20_reduced_skip_cls"

# python3 -m accelerate.commands.launch \
#     --num_processes=1 \
#     -m lmms_eval \
#     --model llava \
#     --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
#     --tasks pope,mme \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix llava_v1.6_pope \
#     --output_path ./logs/ \
#     --generation_type recursion \
#     --fix_grid 2x2 \
#     --attention_thresholding_type layer_mean_with_top_k\
#     --attention_threshold 0.4 \
#     --remove_unpadding True \
#     --regenerate_condition all \
#     --verbosity DEBUG \
#     --positional_embedding_type reduced_skip_cls \
#     --wandb_args "project=llava1.6_recursive_eval_Jake2,entity=VLM_Hallucination_Woohyeon,name=llava-v1.6-7b-84-168-336-topk40_reduced_skip_cls"

python3 -m accelerate.commands.launch \
    --num_processes=1 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
    --tasks pope,mme \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.6_pope \
    --output_path ./logs/ \
    --generation_type recursion \
    --fix_grid 2x2 \
    --attention_thresholding_type layer_mean_with_top_k\
    --attention_threshold 0.3 \
    --remove_unpadding True \
    --regenerate_condition all \
    --verbosity DEBUG \
    --positional_embedding_type interpolation \
    --wandb_args "project=llava1.6_recursive_eval_Jake3,entity=VLM_Hallucination_Woohyeon,name=llava-v1.6-7b-84-168-336-topk30_bilinear"

python3 -m accelerate.commands.launch \
    --num_processes=1 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
    --tasks pope,mme \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.6_pope \
    --output_path ./logs/ \
    --generation_type recursion \
    --fix_grid 2x2 \
    --attention_thresholding_type confidence_based_topk\
    --attention_threshold 0.15 \
    --remove_unpadding True \
    --regenerate_condition all \
    --verbosity DEBUG \
    --positional_embedding_type interpolation \
    --wandb_args "project=llava1.6_recursive_eval_Jake3,entity=VLM_Hallucination_Woohyeon,name=llava-v1.6-7b-84-168-336-cnf_topk15_bilinear"

# python3 -m accelerate.commands.launch \
#     --num_processes=1 \
#     -m lmms_eval \
#     --model llava \
#     --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
#     --tasks pope,mme \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix llava_v1.6_pope \
#     --output_path ./logs/ \
#     --generation_type recursion \
#     --fix_grid 2x2 \
#     --attention_thresholding_type layer_mean_with_top_k\
#     --attention_threshold 0.4 \
#     --remove_unpadding True \
#     --regenerate_condition all \
#     --verbosity DEBUG \
#     --positional_embedding_type interpolation \
#     --wandb_args "project=llava1.6_recursive_eval_Jake2,entity=VLM_Hallucination_Woohyeon,name=llava-v1.6-7b-84-168-336-topk40_bilinear"

# python3 -m accelerate.commands.launch \
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
#     --attention_thresholding_type layer_mean_with_top_k\
#     --attention_threshold 0.2 \
#     --remove_unpadding True \
#     --regenerate_condition all \
#     --verbosity DEBUG \
#     --positional_embedding_type reduced \
#     --wandb_args "project=llava1.6_recursive_eval_Jake2,entity=VLM_Hallucination_Woohyeon,name=llava-v1.6-7b-42-84-168-336-topk20"

# python3 -m accelerate.commands.launch \
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
#     --attention_thresholding_type layer_mean_with_top_k\
#     --attention_threshold 0.4 \
#     --remove_unpadding True \
#     --regenerate_condition all \
#     --verbosity DEBUG \
#     --positional_embedding_type reduced \
#     --wandb_args "project=llava1.6_recursive_eval_Jake2,entity=VLM_Hallucination_Woohyeon,name=llava-v1.6-7b-42-84-168-336-topk40"

# python3 -m accelerate.commands.launch \
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
#     --attention_thresholding_type layer_mean_with_top_k\
#     --attention_threshold 0.5 \
#     --remove_unpadding True \
#     --regenerate_condition all \
#     --verbosity DEBUG \
#     --positional_embedding_type reduced \
#     --wandb_args "project=llava1.6_recursive_eval_Jake2,entity=VLM_Hallucination_Woohyeon,name=llava-v1.6-7b-42-84-168-336-topk50"


# python3 -m accelerate.commands.launch \
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
#     --attention_thresholding_type layer_mean_with_top_k\
#     --attention_threshold 0.4 \
#     --remove_unpadding True \
#     --regenerate_condition all \
#     --verbosity DEBUG \
#     --positional_embedding_type reduced \
#     --wandb_args "project=llava1.6_recursive_eval_Jake2,entity=VLM_Hallucination_Woohyeon,name=llava-v1.6-7b-84-168-336-topk40"

# python3 -m accelerate.commands.launch \
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
#     --attention_thresholding_type layer_mean_with_top_k\
#     --attention_threshold 0.5 \
#     --remove_unpadding True \
#     --regenerate_condition all \
#     --verbosity DEBUG \
#     --positional_embedding_type reduced \
#     --wandb_args "project=llava1.6_recursive_eval_Jake2,entity=VLM_Hallucination_Woohyeon,name=llava-v1.6-7b-84-168-336-topk50"

# python3 -m accelerate.commands.launch \
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
#     --attention_thresholding_type layer_mean_with_top_k\
#     --attention_threshold 0.6 \
#     --remove_unpadding True \
#     --regenerate_condition all \
#     --verbosity DEBUG \
#     --positional_embedding_type reduced \
#     --wandb_args "project=llava1.6_recursive_eval_Jake2,entity=VLM_Hallucination_Woohyeon,name=llava-v1.6-7b-84-168-336-topk60"

# python3 -m accelerate.commands.launch \
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
#     --attention_thresholding_type layer_mean_with_top_k\
#     --attention_threshold 0.7 \
#     --remove_unpadding True \
#     --regenerate_condition all \
#     --verbosity DEBUG \
#     --positional_embedding_type reduced \
#     --wandb_args "project=llava1.6_recursive_eval_Jake2,entity=VLM_Hallucination_Woohyeon,name=llava-v1.6-7b-84-168-336-topk70"

# python3 -m accelerate.commands.launch \
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
#     --attention_thresholding_type layer_mean_with_top_k\
#     --attention_threshold 0.8 \
#     --remove_unpadding True \
#     --regenerate_condition all \
#     --verbosity DEBUG \
#     --positional_embedding_type reduced \
#     --wandb_args "project=llava1.6_recursive_eval_Jake2,entity=VLM_Hallucination_Woohyeon,name=llava-v1.6-7b-84-168-336-topk80"

# # python3 -m accelerate.commands.launch \
# #     --num_processes=1 \
# #     -m lmms_eval \
# #     --model llava \
# #     --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
# #     --tasks pope \
# #     --batch_size 1 \
# #     --log_samples \
# #     --log_samples_suffix llava_v1.6_pope \
# #     --output_path ./logs/ \
# #     --generation_type recursion \
# #     --fix_grid 2x2 \
# #     --attention_thresholding_type layer_mean_with_top_k\
# #     --attention_threshold 0.1 \
# #     --remove_unpadding True \
# #     --regenerate_condition all \
# #     --verbosity DEBUG \
# #     --positional_embedding_type reduced \
# #     --wandb_args "project=llava1.6_recursive_eval_Jake,entity=VLM_Hallucination_Woohyeon,name=llava-v1.6-7b-84-168-336-topk10"

# # python3 -m accelerate.commands.launch \
# #     --num_processes=1 \
# #     -m lmms_eval \
# #     --model llava \
# #     --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
# #     --tasks pope \
# #     --batch_size 1 \
# #     --log_samples \
# #     --log_samples_suffix llava_v1.6_pope \
# #     --output_path ./logs/ \
# #     --generation_type recursion \
# #     --fix_grid 2x2 \
# #     --attention_thresholding_type layer_mean_with_top_k\
# #     --attention_threshold 0.3 \
# #     --remove_unpadding True \
# #     --regenerate_condition all \
# #     --verbosity DEBUG \
# #     --positional_embedding_type reduced \
# #     --wandb_args "project=llava1.6_recursive_eval_Jake,entity=VLM_Hallucination_Woohyeon,name=llava-v1.6-7b-84-168-336-topk30"


#     # python3 -m accelerate.commands.launch \
#     # --num_processes=1 \
#     # -m lmms_eval \
#     # --model llava \
#     # --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
#     # --tasks pope \
#     # --batch_size 1 \
#     # --log_samples \
#     # --log_samples_suffix llava_v1.6_pope \
#     # --output_path ./logs/ \
#     # --generation_type recursion \
#     # --fix_grid 2x2 \
#     # --attention_thresholding_type confidence_based_topk \
#     # --attention_threshold 0.1 \
#     # --remove_unpadding True \
#     # --regenerate_condition all \
#     # --verbosity DEBUG \
#     # --positional_embedding_type reduced \
#     # --wandb_args "project=llava1.6_recursive_eval_Jake,entity=VLM_Hallucination_Woohyeon,name=llava-v1.6-7b-168-336-cnf_topk10"

#     # # python3 -m accelerate.commands.launch \
#     # # --num_processes=1 \
#     # # -m lmms_eval \
#     # # --model llava \
#     # # --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
#     # # --tasks pope \
#     # # --batch_size 1 \
#     # # --log_samples \
#     # # --log_samples_suffix llava_v1.6_pope \
#     # # --output_path ./logs/ \
#     # # --generation_type recursion \
#     # # --fix_grid 2x2 \
#     # # --attention_thresholding_type confidence_based_topk \
#     # # --attention_threshold 0.2 \
#     # # --remove_unpadding True \
#     # # --regenerate_condition all \
#     # # --verbosity DEBUG \
#     # # --positional_embedding_type reduced \
#     # # --wandb_args "project=llava1.6_recursive_eval_Jake,entity=VLM_Hallucination_Woohyeon,name=llava-v1.6-7b-168-336-cnf_topk20"
