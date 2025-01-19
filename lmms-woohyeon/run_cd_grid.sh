# !/bin/bash

# python cd_gridsearch_test.py \
#     --logit_file_path "/workspace/vlm/lmms-woohyeon/logit_pope_pop_2.5_2.5_2.5_bilinear_s1_top100.json" \
#     --response_file_path "/workspace/vlm/lmms-woohyeon/logs/liuhaotian__llava-v1.6-vicuna-7b/20250114_184942_samples_pope_pop.jsonl" \
#     --output_path_grid_search "/workspace/vlm/lmms-woohyeon/logging/pope_pop_2.5_2.5_2.5_cd_adaptive.json" \
#     --max 1.0 \
#     --step 0.01 \
#     --mode pope \

python cd_gridsearch_test.py \
    --logit_file_path "/workspace/vlm/lmms-woohyeon/logit_mmstar_0.5_bilinear_s1_top100.json" \
    --response_file_path "/workspace/vlm/lmms-woohyeon/logs/liuhaotian__llava-v1.6-vicuna-7b/20250114_203756_samples_mmstar.jsonl" \
    --output_path_grid_search "/workspace/vlm/lmms-woohyeon/logging/mmstar_0.5_cd_adaptive.json" \
    --max 1.0 \
    --step 0.1 \
    --mode mmstar \

# python cd_gridsearch_test.py \
#     --logit_file_path "/workspace/vlm/lmms-woohyeon/logit_mmbench_en_dev_lite_2.0_2.0_2.0_bilinear_s1_top100.json" \
#     --response_file_path "/workspace/vlm/lmms-eval-cd/logs/liuhaotian__llava-v1.6-vicuna-7b/20250114_214458_samples_mmbench_en_dev_lite.jsonl" \
#     --output_path_grid_search "/workspace/vlm/lmms-woohyeon/logging/mmbench_en_dev_lite_2.0_2.0_2.0__cd_adaptive.json" \
#     --max 1.0 \
#     --step 0.1 \
#     --mode mmbench \