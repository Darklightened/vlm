#!/bin/bash

LOG_FILE_PATH=/workspace/vlm/lmms-learn-wjk/logs/liuhaotian__llava-v1.6-vicuna-7b/20241220_050832_samples_mme.jsonl
CSV_FILE_PATH=/workspace/vlm/lmms-learn-wjk/csv/generation_output_mme_tta-topk-90-90-30-bilinear-norm-lr001-iter100-n1.csv
OUTPUT_PATH=./processed_output_mme_tta-topk-90-90-30-bilinear-norm-lr001-iter100-n1.csv
MODE=mme

python match_res.py --LOG_FILE_PATH $LOG_FILE_PATH --CSV_FILE_PATH $CSV_FILE_PATH --MODE $MODE
python eval_confidence.py --OUTPUT_PATH $OUTPUT_PATH