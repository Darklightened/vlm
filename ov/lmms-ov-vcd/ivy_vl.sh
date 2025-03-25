
python3 -m accelerate.commands.launch \
    --num_processes=4 \
    --main_process_port 29832 \
    -m lmms_eval \
    --model llava_onevision \
    --model_args pretrained='AI-Safeguard/Ivy-VL-llava',model_name='llava_qwen' \
    --tasks mmstar \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.6_pope \
    --output_path ./logs/ \
    --attention_threshold '5.0' \
    --verbosity DEBUG \