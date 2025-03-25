import os
import itertools

# Define the base command template
base_command = """
python3 -m accelerate.commands.launch \
    --num_processes=1 \
    -m lmms_eval \
    --device cuda:1 \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b" \
    --tasks pope_pop \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.6_pope \
    --output_path ./logs/ \
    --generation_type recursion \
    --fix_grid 2x2 \
    --attention_thresholding_type layer_mean \
    --attention_threshold "{threshold}" \
    --positional_embedding_type bilinear_interpolation \
    --remove_unpadding True \
    --attn_norm norm_relu \
    --stages "-2" "-1" "0" "1" \
    --verbosity DEBUG \
    --square 1 \
    --tta_learning_rate {lr} \
    --tta_n_iter 2500 \
    --wandb_args "project=llava1.6_recursive_eval_search_TTA,entity=VLM_Hallucination_Woohyeon,name={wandb_name}"
"""

# Define the grid search space for attention_threshold
attention_threshold_values = [0.05, 0.01, 0.00]  # Updated values for grid search
attention_threshold_combinations = list(itertools.product(attention_threshold_values, repeat=3))  # All combinations (125 total)

# Define the grid search space for learning rate
learning_rates = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]  # 5 values

# Generate and execute commands for all combinations
for i, thresholds in enumerate(attention_threshold_combinations):
    # Convert threshold tuple to string
    threshold_str = "[" + ",".join(map(str, thresholds)) + "]"
    
    for lr in learning_rates:
        # Generate a unique wandb name
        wandb_name = f"grid_{i+1:03d}_th_{'_'.join(map(str, thresholds))}_lr_{lr:.0e}"
        
        # Format the command
        command = base_command.format(threshold=threshold_str, lr=lr, wandb_name=wandb_name)
        
        # Print and execute the command
        print(f"Running experiment: {wandb_name}")
        os.system(command)