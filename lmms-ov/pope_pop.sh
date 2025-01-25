CUDA_VISIBLE_DEVICES=0 python3 -m accelerate.commands.launch \
    --num_processes=1 \
    --main_process_port 29832 \
    -m lmms_eval \
    --model llava_onevision \
    --model_args pretrained='lmms-lab/llava-onevision-qwen2-0.5b-ov' \
    --tasks pope_pop \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.6_pope \
    --output_path ./logs/ \
    --attention_threshold '1.0' \
    --verbosity DEBUG \
    --contrastive_alphas '0.4' '0.6' '0.8' \
    --wandb_args "project=llava1.6_recursive_eval_ov_0.5_cd,entity=VLM_Hallucination_Woohyeon,name=new_f16_1.0_bilinear_s1_[0.4-0.6-0.8]" \
    # --save_output True \
    # --output_json_path "./ov_logit_pope_pop_1.0_bilinear_s1_top100.json" \
    # --output_csv_path "./ov_generation_output_pope_pop_1.0_bilinear_s1_top100.csv" \

CUDA_VISIBLE_DEVICES=0 python3 -m accelerate.commands.launch \
    --num_processes=1 \
    --main_process_port 29832 \
    -m lmms_eval \
    --model llava_onevision \
    --model_args pretrained='lmms-lab/llava-onevision-qwen2-0.5b-ov' \
    --tasks pope_aokvqa_pop \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.6_pope \
    --output_path ./logs/ \
    --attention_threshold '1.0' \
    --verbosity DEBUG \
    --contrastive_alphas '0.5' '1.0' '0.7' \
    --wandb_args "project=llava1.6_recursive_eval_ov_0.5_cd,entity=VLM_Hallucination_Woohyeon,name=new_f16_1.0_bilinear_s1_[0.5-1.0-0.7]" \
    # --save_output True \
    # --output_json_path "./ov_logit_pope_pop_1.0_bilinear_s1_top100.json" \
    # --output_csv_path "./ov_generation_output_pope_pop_1.0_bilinear_s1_top100.csv" \

CUDA_VISIBLE_DEVICES=0 python3 -m accelerate.commands.launch \
    --num_processes=1 \
    --main_process_port 29832 \
    -m lmms_eval \
    --model llava_onevision \
    --model_args pretrained='lmms-lab/llava-onevision-qwen2-0.5b-ov' \
    --tasks pope_gqa_pop \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.6_pope \
    --output_path ./logs/ \
    --attention_threshold '1.0' \
    --verbosity DEBUG \
    --contrastive_alphas '0.5' '0.8' '0.6' \
    --wandb_args "project=llava1.6_recursive_eval_ov_0.5_cd,entity=VLM_Hallucination_Woohyeon,name=new_f16_1.0_bilinear_s1_[0.5-0.8-0.6]" \
    # --save_output True \
    # --output_json_path "./ov_logit_pope_pop_1.0_bilinear_s1_top100.json" \
    # --output_csv_path "./ov_generation_output_pope_pop_1.0_bilinear_s1_top100.csv" \
