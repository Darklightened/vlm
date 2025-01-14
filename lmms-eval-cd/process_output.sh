#!/bin/bash

LOG_FILE_PATH=/workspace/vlm/lmms-eval-updated_new/logs/liuhaotian__llava-v1.6-vicuna-7b/20250111_212327_samples_pope_pop.jsonl
CSV_FILE_PATH=/workspace/vlm/lmms-eval-updated_new/generation_output_pope_pop_1.0_1.0_1.0_bilinear_s1.csv
OUTPUT_PATH=./processed_output_pope_pop_1.0_1.0_1.0_bilinear_s1.csv
MODE=pope

python match_res.py --LOG_FILE_PATH $LOG_FILE_PATH --CSV_FILE_PATH $CSV_FILE_PATH --MODE $MODE
python eval_confidence.py --OUTPUT_PATH $OUTPUT_PATH