
CUDA_VISIBLE_DEVICES=3 python3 -m accelerate.commands.launch \
    --num_processes=1 \
    --main_process_port 29827 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained='liuhaotian/llava-v1.6-mistral-7b' \
    --tasks pope_gqa_pop \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.6_pope \
    --output_path ./logs/ \
    --generation_type recursion \
    --fix_grid 2x2 \
    --attention_thresholding_type attn_topk \
    --attention_threshold '1.0' \
    --positional_embedding_type bilinear_interpolation \
    --remove_unpadding True \
    --attn_norm None \
    --stages '-2' '-1' '0' '1' \
    --verbosity DEBUG \
    --square 1 \
    --contrastive_alphas '0.0' '0.0' '0.0' \
    --save_output True \
    --output_json_path './logit_mistral_7b_pope_gqa_pop_1.0_bilinear_s1_top100.json' \
    --output_csv_path './generation_output_pope_gqa_pop_1.0_bilinear_s1_top100.csv' \
    --wandb_args 'project=llava1.6_recursive_eval_mistral-7b,entity=VLM_Hallucination_Woohyeon,name=pope_gqa_pop_1.0_bilinear_s1' \
