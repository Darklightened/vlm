import os
import numpy as np
import itertools

def generate_command(attention_threshold, experiment_name_suffix):
    """
    Generate the bash command to run the experiment.
    
    Args:
        attention_threshold (str): The attention threshold value for this run.
        experiment_name_suffix (str): Unique suffix for the wandb experiment name.
        device (str): Device to run the experiment on (default: cuda:2).
    
    Returns:
        str: The generated bash command.
    """
    base_command = (
        f"python3 -m accelerate.commands.launch "
        f"--num_processes=1 "
        f"--main_process_port 29831 "
        f"-m lmms_eval "
        f"--model llava "
        f"--model_args pretrained='Yi-VL-6B' "
        f"--tasks pope_pop,pope_aokvqa_pop,pope_gqa_pop,mmstar,mmbench_en_dev_lite,vqav2_val_lite "
        f"--batch_size 1 "
        f"--log_samples "
        f"--log_samples_suffix llava_v1.6_pope "
        f"--output_path ./logs/ "
        f"--generation_type recursion "
        f"--fix_grid 2x2 "
        f"--attention_thresholding_type attn_topk "
        f"--attention_threshold '{attention_threshold}' "
        f"--positional_embedding_type bilinear_interpolation "
        f"--remove_unpadding True "
        f"--attn_norm None "
        f"--stages '-2' '-1' '0' "
        f"--verbosity DEBUG "
        f"--square 1 "
        f"--contrastive_alphas '0.0' '0.0' "
        f"--wandb_args 'project=llava1.6_recursive_eval_Yi_VL_6B,entity=VLM_Hallucination_Woohyeon,name=exp_attn_{experiment_name_suffix}'"
    )
    return base_command

# def run_experiments():
#     """
#     Run experiments for specified attention_threshold values in reverse order.
#     """
#     # Define the specific threshold values
#     first_stage_values = [0.7, 1.0]
#     other_stage_values = [0.1,0.3,0.5,0.7,1.0]
    
#     # Generate combinations for other stages
#     other_stage_combinations = list(itertools.product(other_stage_values, repeat=2))
    
#     experiment_idx = 0
#     total_experiments = len(first_stage_values) * len(other_stage_combinations) + 0 
    
#     for first_stage in first_stage_values:
#         for other_stages in other_stage_combinations:
#             # Combine thresholds
#             thresholds = [first_stage] + list(other_stages)
            
#             # Create the unique name for the wandb experiment
#             experiment_name_suffix = f"run_{experiment_idx}_thresh_{'_'.join(map(lambda x: f'{x:.2f}', thresholds))}"
            
#             # Convert the thresholds to string format
#             attention_threshold = f"[{','.join(map(str, thresholds))}]"
            
#             # Generate the command
#             command = generate_command(attention_threshold, experiment_name_suffix)
            
#             # Print and run the command
#             print(f"Running experiment {experiment_idx + 1}/{total_experiments}: {command}")
#             os.system(command)
            
#             experiment_idx += 1

# # Run the experiments with the defined thresholds
# run_experiments()

def run_experiments():
    """
    Run experiments for specified attention_threshold values with identical thresholds across all stages.
    """
    # Define the specific identical threshold values
    uniform_threshold_values = [2.5, 5.0, 7.0, 10.0]
    
    experiment_idx = 0
    total_experiments = len(uniform_threshold_values)
    
    for threshold in uniform_threshold_values:
        # Use the same threshold for all stages
        thresholds = [threshold] * 3  # Assuming 3 stages; adjust as needed
        
        # Create the unique name for the wandb experiment
        experiment_name_suffix = f"run_{experiment_idx}_thresh_{'_'.join(map(lambda x: f'{x:.2f}', thresholds))}"
        
        # Convert the thresholds to string format
        attention_threshold = f"[{','.join(map(str, thresholds))}]"
        
        # Generate the command
        command = generate_command(attention_threshold, experiment_name_suffix)
        
        # Print and run the command
        print(f"Running experiment {experiment_idx + 1}/{total_experiments}: {command}")
        os.system(command)
        
        experiment_idx += 1

# Run the experiments with uniform thresholds
run_experiments()