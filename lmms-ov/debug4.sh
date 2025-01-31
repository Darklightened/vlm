
CUDA_VISIBLE_DEVICES=3 python3 -m accelerate.commands.launch \
    --num_processes=1 \
    --main_process_port 29835 \
    -m lmms_eval \
    --model llava_onevision \
    --model_args pretrained='lmms-lab/llava-onevision-qwen2-0.5b-ov' \
    --tasks mmstar \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.6_pope \
    --output_path ./logs/ \
    --attention_threshold '0.5' \
    --verbosity DEBUG \
    --save_output True \
    --output_json_path "./logit_mmstar_0.5_bilinear_s1_top100.json" \
    --output_csv_path "./generation_output_mmstar_0.5_bilinear_s1_top100.csv" \
    --wandb_args "project=llava1.6_recursive_eval_ov_0.5_cd,entity=VLM_Hallucination_Woohyeon,name=mmstar_0.5_bilinear_s1" \

CUDA_VISIBLE_DEVICES=3 python3 -m accelerate.commands.launch \
    --num_processes=1 \
    --main_process_port 29836 \
    -m lmms_eval \
    --model llava_onevision \
    --model_args pretrained='lmms-lab/llava-onevision-qwen2-0.5b-ov' \
    --tasks mmbench_en_dev_lite \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.6_pope \
    --output_path ./logs/ \
    --attention_threshold '1.5' \
    --verbosity DEBUG \
    --save_output True \
    --output_json_path "./logit_mmbench_en_dev_lite_1.5_bilinear_s1_top100.json" \
    --output_csv_path "./generation_output_mmbench_en_dev_lite_1.5_bilinear_s1_top100.csv" \
    --wandb_args "project=llava1.6_recursive_eval_ov_0.5_cd,entity=VLM_Hallucination_Woohyeon,name=mmbench_en_dev_lite_1.5_bilinear_s1" \

CUDA_VISIBLE_DEVICES=3 python3 -m accelerate.commands.launch \
    --num_processes=1 \
    --main_process_port 29837 \
    -m lmms_eval \
    --model llava_onevision \
    --model_args pretrained='lmms-lab/llava-onevision-qwen2-0.5b-ov' \
    --tasks mmbench_en_dev \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.6_pope \
    --output_path ./logs/ \
    --attention_threshold '1.5' \
    --verbosity DEBUG \
    --save_output True \
    --output_json_path "./logit_mmbench_en_dev_1.5_bilinear_s1_top100.json" \
    --output_csv_path "./generation_output_mmbench_en_dev_1.5_bilinear_s1_top100.csv" \
    --wandb_args "project=llava1.6_recursive_eval_ov_0.5_cd,entity=VLM_Hallucination_Woohyeon,name=mmbench_en_dev_1.5_bilinear_s1" \
