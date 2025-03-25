#!/bin/bash

# LOG_FILE_PATH=/workspace/vlm/lmms-eval-updated_new/20250120_144143_samples_pope_aokvqa_pop.jsonl
# CSV_FILE_PATH=/workspace/vlm/lmms-eval-updated_new/generation_output_pope_aokvqa_pop_default_llava_next.csv
# OUTPUT_PATH=./processed_output_pope_aokvqa_pop_default.csv
# MODE=pope

# python match_res.py --LOG_FILE_PATH $LOG_FILE_PATH --CSV_FILE_PATH $CSV_FILE_PATH --MODE $MODE
# python eval_confidence.py --OUTPUT_PATH $OUTPUT_PATH

INPUT_PATHS="/workspace/vlm/lmms-eval-updated_new/matched_output_pope_aokvqa_pop.csv" "/workspace/vlm/lmms-eval-updated_new/matched_output_pope_gqa_pop.csv" "/workspace/vlm/lmms-eval-updated_new/matched_output_pope_pop.csv"

python draw_hallucinated_bins.py --INPUT_PATHS "/workspace/vlm/lmms-eval-updated_new/matched_output_pope_aokvqa_pop.csv" "/workspace/vlm/lmms-eval-updated_new/matched_output_pope_gqa_pop.csv" "/workspace/vlm/lmms-eval-updated_new/matched_output_pope_pop.csv"