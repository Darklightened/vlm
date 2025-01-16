CUDA_VISIBLE_DEVICES=0,1,2 python3 -m accelerate.commands.launch \
    --num_processes=3 \
    -m lmms_eval \
    --model llava_onevision \
    --model_args pretrained="AI-Safeguard/Ivy-VL-llava",model_name="llava_qwen" \
    --tasks pope_pop \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.5_mme \
    --output_path ./logs/