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
        f"--num_processes=2 "
        f"--main_process_port 29823 "
        f"-m lmms_eval "
        f"--model llava "
        f"--model_args pretrained='liuhaotian/llava-v1.6-vicuna-7b' "
        f"--tasks mme,vqav2_val_lite "
        f"--batch_size 1 "
        f"--log_samples "
        f"--log_samples_suffix llava_v1.6_pope "
        f"--output_path ./logs/ "
        f"--generation_type recursion "
        f"--fix_grid 2x2 "
        f"--attention_thresholding_type layer_mean_topk "
        f"--attention_threshold '{attention_threshold}' "
        f"--positional_embedding_type bilinear_interpolation "
        f"--remove_unpadding True "
        f"--attn_norm None "
        f"--stages '-2' '-1' '0' '1' "
        f"--verbosity DEBUG "
        f"--square 1 "
        f"--wandb_args 'project=llava1.6_recursive_eval_gridsearch,entity=VLM_Hallucination_Woohyeon,name=exp_{experiment_name_suffix}'"
    )
    return base_command

def run_experiments(start=0.1, stop=1.0, step=0.1):
    """
    Run experiments for all combinations of attention_threshold values, in reverse order.
    
    Args:
        start (float): Start value for the range.
        stop (float): Stop value for the range.
        step (float): Step size for the range.
    """
    # Generate all values for thresholds
    threshold_values = np.arange(start, stop + step, step)
    threshold_combinations = list(itertools.product(threshold_values, repeat=3))
    
    # Reverse the order of combinations
    threshold_combinations = threshold_combinations[::-1]
    
    for idx, thresholds in enumerate(threshold_combinations):
        # Create the unique name for the wandb experiment
        experiment_name_suffix = f"run_{idx}_thresh_{'_'.join(map(lambda x: f'{x:.2f}', thresholds))}"
        
        # Convert the thresholds to string format
        attention_threshold = f"[{','.join(map(str, thresholds))}]"
        
        # Generate the command
        command = generate_command(attention_threshold, experiment_name_suffix)
        
        # Print and run the command
        print(f"Running experiment {idx + 1}/{len(threshold_combinations)}: {command}")
        os.system(command)

# Run experiments with thresholds from 1.0 to 0.1 in reverse order
run_experiments(start=0.1, stop=1.0, step=0.1)