#!/bin/bash

LOG_FILE_PATH=/workspace/vlm/lmms-eval-updated/logs/liuhaotian__llava-v1.6-vicuna-7b/20250102_023823_samples_mme.jsonl
CSV_FILE_PATH=/workspace/vlm/lmms-eval-updated/generation_output_mme_10_10_1_bilinear_s1_pad.csv
OUTPUT_PATH=./processed_output_mme_10_10_1_bilinear_s1_pad.csv
MODE=mme

python match_res.py --LOG_FILE_PATH $LOG_FILE_PATH --CSV_FILE_PATH $CSV_FILE_PATH --MODE $MODE
python eval_confidence.py --OUTPUT_PATH $OUTPUT_PATH