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
        f"CUDA_VISIBLE_DEVICES=3 "
        f"python3 -m accelerate.commands.launch "
        f"--num_processes=1 "
        f"--main_process_port 29840 "
        f"-m lmms_eval "
        f"--model llava_onevision "
        f"--model_args pretrained='lmms-lab/llava-onevision-qwen2-0.5b-ov' "
        f"--tasks vqav2_val_lite "
        f"--batch_size 1 "
        f"--log_samples "
        f"--log_samples_suffix llava_v1.6_pope "
        f"--output_path ./logs/ "
        f"--attention_threshold '{attention_threshold}' "
        f"--verbosity DEBUG "
        f"--wandb_args 'project=llava1.6_recursive_eval_ov_0.5_cd,entity=VLM_Hallucination_Woohyeon,name=new_f16_attn_{experiment_name_suffix}'"
    )
    return base_command

def run_experiments():
    """
    Run experiments for specified attention_threshold values with identical thresholds across all stages.
    """
    # Define the specific identical threshold values
    uniform_threshold_values = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    
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