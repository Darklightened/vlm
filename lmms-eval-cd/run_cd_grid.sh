#!/bin/bash

# python cd_gridsearch_test.py \
#     --logit_file_path "/workspace/vlm/lmms-eval-updated_new/logit_pope_pop_1.0_1.0_1.0_bilinear_s1.json" \
#     --response_file_path "/workspace/vlm/lmms-eval-updated_new/logs/liuhaotian__llava-v1.6-vicuna-7b/20250111_214012_samples_pope_pop.jsonl" \
#     --output_path_grid_search "./cd_grid_pope_pop_1.0_1.0_1.0.json" \
#     --mode pope \

# python cd_gridsearch_test.py \
#     --logit_file_path "/workspace/vlm/lmms-eval-updated_new/logit_mmstar_10.0_10.0_10.0_bilinear_s1.json" \
#     --response_file_path "/workspace/vlm/lmms-eval-updated_new/logs/liuhaotian__llava-v1.6-vicuna-7b/20250112_012512_samples_mmstar.jsonl" \
#     --output_path_grid_search "./cd_grid_mmstar_10.0_10.0_10.0.json" \
#     --mode mmstar \

python cd_gridsearch_test.py \
    --logit_file_path "/workspace/vlm/lmms-eval-updated_new/logit_mmbench_en_dev_lite_1.0_1.0_1.0_bilinear_s1.json" \
    --response_file_path "/workspace/vlm/lmms-eval-updated_new/logs/liuhaotian__llava-v1.6-vicuna-7b/20250112_003837_samples_mmbench_en_dev_lite.jsonl" \
    --output_path_grid_search "./cd_grid_mmbench_1.0_1.0_1.0.json" \
    --mode mmbench \

