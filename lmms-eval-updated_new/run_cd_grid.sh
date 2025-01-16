#!/bin/bash

python cd_gridsearch_test.py \
    --logit_file_path "/workspace/vlm/lmms-eval-updated_new/logit_pope_pop_2.5_2.5_2.5_bilinear_s1.json" \
    --response_file_path "/workspace/vlm/lmms-eval-updated_new/logs/liuhaotian__llava-v1.6-vicuna-7b/20250114_184942_samples_pope_pop.jsonl" \
    --output_path_grid_search "/workspace/vlm/lmms-eval-updated_new/logging/pope_pop_1.0_1.0_1.0_cd_adaptive.json" \
    --max 0.5 \
    --mode pope \

python cd_gridsearch_test.py \
    --logit_file_path "/workspace/vlm/lmms-eval-updated_new/logit_pope_gqa_pop_2.5_2.5_2.5_bilinear_s1.json" \
    --response_file_path "/workspace/vlm/lmms-eval-updated_new/logs/liuhaotian__llava-v1.6-vicuna-7b/20250115_003456_samples_pope_gqa_pop.jsonl" \
    --output_path_grid_search "/workspace/vlm/lmms-eval-updated_new/logging/pope_gqa_pop_2.5_2.5_2.5_cd_adaptive.json" \
    --max 0.5 \
    --mode pope_gqa \

python cd_gridsearch_test.py \
    --logit_file_path "/workspace/vlm/lmms-eval-updated_new/logit_pope_aokvqa_pop_2.5_2.5_2.5_bilinear_s1.json" \
    --response_file_path "/workspace/vlm/lmms-eval-updated_new/logs/liuhaotian__llava-v1.6-vicuna-7b/20250115_015051_samples_pope_aokvqa_pop.jsonl" \
    --output_path_grid_search "/workspace/vlm/lmms-eval-updated_new/logging/pope_vqa_pop_2.5_2.5_2.5_cd_adaptive.json" \
    --max 0.5 \
    --mode pope_aokvqa \

python cd_gridsearch_test.py \
    --logit_file_path "/workspace/vlm/lmms-eval-updated_new/logit_mmbench_en_dev_lite_2.0_2.0_2.0_bilinear_s1.json" \
    --response_file_path "/workspace/vlm/lmms-eval-cd/logs/liuhaotian__llava-v1.6-vicuna-7b/20250114_214458_samples_mmbench_en_dev_lite.jsonl" \
    --output_path_grid_search "/workspace/vlm/lmms-eval-updated_new/logging/mmbench_en_dev_lite_2.0_2.0_2.0__cd_adaptive.json" \
    --max 0.5 \
    --mode mmbench \

python cd_gridsearch_test.py \
    --logit_file_path "/workspace/vlm/lmms-eval-updated_new/logit_mmstar_7.0_7.0_7.0_bilinear_s1.json" \
    --response_file_path "/workspace/vlm/lmms-eval-updated_new/logs/liuhaotian__llava-v1.6-vicuna-7b/20250114_203756_samples_mmstar.jsonl" \
    --output_path_grid_search "/workspace/vlm/lmms-eval-updated_new/logging/mmstar_10.0_10.0_10.0_cd_adaptive.json" \
    --max 0.5 \
    --mode mmstar \

# Bench: 
# 80.3 -> 79.5 (weight ablation),   78.0 (Single CD), 78.0 (Single CD, weight x)
# Star: 
# 38.5 -> 37.8, 37.4, 37.4
# Pope coco: 
# 89.2-> 88.9, 89.1 , 88.9
# Pope GQA:
# 87.2 -> 87.2, 87.0, 87.0
# POPE VQA:
# 90.4 -> 90.4 , 90.2, 90.4
