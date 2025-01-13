#!/bin/bash

# python cd_gridsearch_test.py \
#     --logit_file_path "/workspace/vlm/lmms-eval-updated_new/logit_pope_pop_2.5_2.5_2.5_bilinear_s1.json" \
#     --response_file_path "/workspace/vlm/lmms-eval-updated_new/logs/liuhaotian__llava-v1.6-vicuna-7b/20250112_184958_samples_pope_pop.jsonl" \
#     --output_path_grid_search "./cd_grid_pope_pop_2.5_2.5_2.5.json" \
#     --max 1.0 \
#     --mode pope \

# python cd_gridsearch_test.py \
#     --logit_file_path "/workspace/vlm/lmms-eval-updated_new/logit_pope_gqa_pop_2.5_2.5_2.5_bilinear_s1.json" \
#     --response_file_path "/workspace/vlm/lmms-eval-updated_new/logs/liuhaotian__llava-v1.6-vicuna-7b/20250112_211516_samples_pope_gqa_pop.jsonl" \
#     --output_path_grid_search "./cd_grid_pope_gqa_pop_2.5_2.5_2.5.json" \
#     --max 1.0 \
#     --mode pope_gqa \

# python cd_gridsearch_test.py \
#     --logit_file_path "/workspace/vlm/lmms-eval-updated_new/logit_mmstar_0.7_0.7_0.7_bilinear_s1.json" \
#     --response_file_path "/workspace/vlm/lmms-eval-updated_new/logs/liuhaotian__llava-v1.6-vicuna-7b/20250112_012512_samples_mmstar.jsonl" \
#     --output_path_grid_search "./cd_grid_mmstar_0.7_0.7_0.7.json" \
#     --max 1.0 \
#     --mode mmstar \

# python cd_gridsearch_test.py \
#     --logit_file_path "/workspace/vlm/lmms-eval-updated_new/logit_mmbench_en_dev_lite_2.0_2.0_2.0_bilinear_s1.json" \
#     --response_file_path "/workspace/vlm/lmms-eval-updated_new/logs/liuhaotian__llava-v1.6-vicuna-7b/20250112_165505_samples_mmbench_en_dev_lite.jsonl" \
#     --output_path_grid_search "./cd_grid_mmbench_2.0_2.0_2.0.json" \
#     --max 1.0 \
#     --mode mmbench \

python cd_gridsearch_test.py \
    --logit_file_path "/workspace/vlm/lmms-eval-updated_new/logit_pope_aokvqa_pop_2.5_2.5_2.5_bilinear_s1.json" \
    --response_file_path "/workspace/vlm/lmms-eval-updated_new/logs/liuhaotian__llava-v1.6-vicuna-7b/20250112_234807_samples_pope_aokvqa_pop.jsonl" \
    --output_path_grid_search "./cd_grid_pope_aokvqa_pop_2.5_2.5_2.5.json" \
    --max 1.0 \
    --mode pope_aokvqa \

