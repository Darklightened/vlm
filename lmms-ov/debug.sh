CUDA_VISIBLE_DEVICES=0,1,2 python3 -m accelerate.commands.launch \
    --num_processes=3 \
    -m lmms_eval \
    --model llava_onevision \
    --model_args pretrained="lmms-lab/llava-onevision-qwen2-7b-si" \
    --tasks pope_pop \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.5_mme \
    --output_path ./logs/ \
    # --wandb_args "project=llava1.6_recursive_eval_ov,entity=VLM_Hallucination_Woohyeon,name=[1.0]-[0.0]" \