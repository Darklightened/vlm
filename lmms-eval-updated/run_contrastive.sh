CUDA_VISIBLE_DEVICES=3 python3 -m accelerate.commands.launch \
    --num_processes=1 \
    --main_process_port 29882 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
    --tasks vqav2_val_lite \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.6_pope \
    --output_path ./logs/ \
    --generation_type recursion \
    --fix_grid 2x2 \
    --attention_thresholding_type layer_mean_topk \
    --attention_threshold "[1.0,0.7,0.1]" \
    --positional_embedding_type bilinear \
    --remove_unpadding True \
    --attn_norm None \
    --stages "-2" "-1" "0" "1" \
    --verbosity DEBUG \
    --square 1 \
    --contrastive_alphas "1.0" "1.0" "1.0"  





