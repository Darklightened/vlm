python3 -m accelerate.commands.launch \
    --num_processes=1 \
    --main_process_port 29881 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.6-mistral-7b" \
    --tasks pope_pop,pope_aokvqa_pop,pope_gqa_pop \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.6_pope \
    --output_path ./logs/ \
    --generation_type recursion \
    --fix_grid 2x2 \
    --attention_thresholding_type attn_topk \
    --attention_threshold "1.0" \
    --positional_embedding_type bilinear_interpolation \
    --remove_unpadding True \
    --attn_norm None \
    --stages "-2" "-1" "0" "1" \
    --verbosity DEBUG \
    --square 1 \
    --contrastive_alphas "0.0" "0.0" "0.0" \
    # --wandb_args "project=llava1.6_recursive_eval_mistral-7b,entity=VLM_Hallucination_Woohyeon,name=mistal-[1.0]-[0.7]" \

