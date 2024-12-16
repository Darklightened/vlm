import torch

torch.backends.cuda.matmul.allow_tf32 = True


import copy
import warnings
from datetime import timedelta
from typing import List, Optional, Tuple, Union

from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from packaging import version
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.utils import stop_sequences_criteria
from lmms_eval.recursion_utils import (
    calculate_entropy_for_attn_threshold, 
    entropy_based_threshold, 
    confidence_based_threshold, 
    calculate_entropy_and_all_confidences
)

import numpy as np
import torch.nn.functional as F
import csv
from pathlib import Path

warnings.filterwarnings("ignore")

from loguru import logger as eval_logger

try:
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import (
        get_model_name_from_path,
        process_images,
        tokenizer_image_token,
        show_mask_on_image,
        get_heatmap,
        make_square,
        process_anyres_image,
        get_heatmap_with_layer_visualization
    )
    from llava.model.builder import load_pretrained_model
except Exception as e:
    eval_logger.debug("LLaVA is not installed. Please install LLaVA to use this model.\nError: %s" % e)

# inference implementation for attention, can be "sdpa", "eager", "flash_attention_2". Seems FA2 is not effective during inference: https://discuss.huggingface.co/t/flash-attention-has-no-effect-on-inference/73453/5
# if is_flash_attn_2_available:
#     best_fit_attn_implementation = "flash_attention_2" # flash_attn has a bug that says: ERROR Error query and key must have the same dtype in generating

if version.parse(torch.__version__) >= version.parse("2.1.2"):
    best_fit_attn_implementation = "sdpa"
else:
    best_fit_attn_implementation = "eager"


@register_model("llava")
class Llava(lmms):
    """
    Llava Model
    """

    def __init__(
        self,
        pretrained: str = "liuhaotian/llava-v1.5-7b",
        truncation: Optional[bool] = True,
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        model_name=None,
        attn_implementation=best_fit_attn_implementation,
        device_map="cuda:0",
        conv_template="vicuna_v1",
        use_cache=True,
        tie_weights: bool = True,
        truncate_context=False,  # whether to truncate the context in generation, set it False for LLaVA-1.6
        customized_config=None,  # ends in json        
        ## new args for recursive generation
        generation_type="default",
        fix_grid="default",
        attention_thresholding_type="layer_mean",
        attention_threshold="0.1",
        remove_unpadding=False,    
        detection_strategy=None,
        detection_threshold=0.8,
        save_output=True,
        output_csv_path = "generation_output.csv",
        target_token_selection_strategy="first",
        all_image_sizes=[168, 336, 672],
        positional_embedding_type="default",
        recursed_image_sizes=[336],   
        save_heatmap_image=False,
        save_heatmap_image_path="./image_heatmap",

        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        llava_model_args = {
            "multimodal": True,
        }
        if customized_config is not None:
            llava_model_args["customized_config"] = customized_config
        if attn_implementation is not None:
            llava_model_args["attn_implementation"] = attn_implementation
        if "use_flash_attention_2" in kwargs:
            llava_model_args["use_flash_attention_2"] = kwargs["use_flash_attention_2"]
        model_name = model_name if model_name is not None else get_model_name_from_path(pretrained)
        try:
            # Try to load the model with the multimodal argument
            self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(pretrained, None, model_name, device_map=self.device_map, **llava_model_args)
        except TypeError:
            # for older versions of LLaVA that don't have multimodal argument
            llava_model_args.pop("multimodal", None)
            self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(pretrained, None, model_name, device_map=self.device_map, **llava_model_args)
        self._config = self._model.config
        self.model.eval()
        if tie_weights:
            self.model.tie_weights()

        self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template
        self.use_cache = use_cache
        self.truncate_context = truncate_context

        # additional parameters for recursion
        self.generation_type = generation_type        
        self.fix_grid = fix_grid
        self.attention_thresholding_type = attention_thresholding_type
        self.attention_threshold = attention_threshold        
        self.detection_strategy = detection_strategy
        self.detection_threshold = detection_threshold 
        self.save_output = save_output
        self.output_csv_path = output_csv_path
        self.target_token_selection_strategy = target_token_selection_strategy
        self.all_image_sizes = all_image_sizes
        self.positional_embedding_type = positional_embedding_type
        self.recursed_image_sizes = recursed_image_sizes
        self.save_heatmap_image = save_heatmap_image
        self.save_heatmap_image_path = save_heatmap_image_path
        
        print(f"generation_type: {generation_type}")
        print(f"fix_grid: {fix_grid}")
        print(f"attention_thresholding_type: {attention_thresholding_type}")
        print(f"attention_threshold: {attention_threshold}")
        print(f"detection_strategy: {detection_strategy}")        
        print(f"detection_threshold: {detection_threshold}")
        print(f"save_output: {save_output}")
        print(f"save_output_csv_path: {output_csv_path}")
        print(f"target_token_selection_strategy: {target_token_selection_strategy}")
        print(f"all_image_sizes: {all_image_sizes}")
        print(f"positional_embedding_type: {positional_embedding_type}")
        print(f"recursed_image_sizes: {recursed_image_sizes}")
        print(f"save_heatmap_image: {save_heatmap_image}")
        print(f"save_heatmap_image_path: {save_heatmap_image_path}")

        ## use all when the image size is not in recursed_image_sizes (until first recursed)
        self.use_all = []
        for image_size in self.all_image_sizes:
            if image_size not in self.recursed_image_sizes:
                self.use_all.append(image_size)
            else:
                break
        
        for image_size in self.recursed_image_sizes:
            assert image_size in self.all_image_sizes, f"Image size {image_size} in recursed_image_sizes is not in all_image_sizes"    
        
        self.downsampled_image_sizes = [image_size for image_size in self.all_image_sizes if image_size not in [336, 672]]            
        
        print(f"Use for recursion: {recursed_image_sizes}, Use all: {self.use_all}")
        
        ## default = "spatial_unpad" for llava1.6
        ## To remove unpadding, set remove_unpadding=True -> mm.path_merge_type will be 'spatial'
        if remove_unpadding == True:
            print("remove unpadding=True, change to 'spatial'")
            self._model.config.mm_patch_merge_type = "spatial"      
        
        ## save output confidences to csv
        if self.save_output:
            assert self.output_csv_path is not None, "Output CSV path is not provided."

            self.output_csv_path = Path(self.output_csv_path)

        if not self.output_csv_path.exists():
            with open(self.output_csv_path, mode="w", newline="") as file:
                writer = csv.writer(file)
                headers = ["Doc ID", "Stage", "Text Output"]   
                headers += [f"Cumulative Confidence {i+1}" for i in range(10)]
                writer.writerow(headers)

        ## modify positional embedding for resized image sizes
        if self.positional_embedding_type == "default":
            assert all_image_sizes==[336], "default embedding only allows size of 336"
        else:
            print(f"change positional embedding to {positional_embedding_type}")
            ## except for default size of 336
            self.model.model.downsampled_vision_towers = [copy.deepcopy(self.model.model.vision_tower) for _ in range(len(self.all_image_sizes)-1)]

            # Default configurations of model position embedding
            patch_size = 14

            for idx, image_size in enumerate(self.downsampled_image_sizes):                
                num_patches = (image_size // patch_size) ** 2
                num_positions = num_patches + 1
                embed_dim = self.model.model.downsampled_vision_tower.vision_tower.vision_model.embeddings.embed_dim

                self.model.model.downsampled_vision_towers[idx].vision_tower.vision_model.embeddings.image_size = image_size
                self.model.model.downsampled_vision_towers[idx].vision_tower.vision_model.embeddings.num_patches = num_patches
                self.model.model.downsampled_vision_towers[idx].vision_tower.vision_model.embeddings.num_positions = num_positions
                self.model.model.downsampled_vision_towers[idx].vision_tower.vision_model.embeddings.register_buffer("position_ids", torch.arange(num_positions).expand((1, -1)), persistent=False)
            
                # Modify positional embedding to match the resized image size
                if positional_embedding_type == "zero":       
                    self.model.model.downsampled_vision_towers[idx].vision_tower.vision_model.embeddings.position_embedding = torch.nn.Embedding(num_positions, embed_dim).to(dtype=torch.float16, device=device)
                    torch.nn.init.constant_(self.model.model.downsampled_vision_towers[idx].vision_tower.vision_model.embeddings.position_embedding.weight, 0)
                elif positional_embedding_type == "interpolation":
                    # Interpolate from the pretrained positional embedding
                    original_embedding = self.model.model.downsampled_vision_towers[idx].vision_tower.vision_model.embeddings.position_embedding.weight.data
                    original_num_positions = original_embedding.size(0)
                    new_embedding = torch.nn.functional.interpolate(
                        original_embedding.unsqueeze(0).transpose(1, 2), 
                        size=(num_positions,), 
                        mode='linear', 
                        align_corners=False
                    ).transpose(1, 2).squeeze(0)
                    self.model.model.downsampled_vision_towers[idx].vision_tower.vision_model.embeddings.position_embedding = torch.nn.Embedding(num_positions, embed_dim).to(dtype=torch.float16, device=device)
                    self.model.model.downsampled_vision_towers[idx].vision_tower.vision_model.embeddings.position_embedding.weight.data.copy_(new_embedding)
                elif positional_embedding_type == "reduced":
                    print("Reduced embedding type.")
                    # Reduce the pretrained embedding by truncating
                    original_embedding = self.model.model.downsampled_vision_towers[idx].vision_tower.vision_model.embeddings.position_embedding.weight.data
                    self.model.model.downsampled_vision_towers[idx].vision_tower.vision_model.embeddings.position_embedding = torch.nn.Embedding(num_positions, embed_dim).to(dtype=torch.float16, device=device)
                    self.model.model.downsampled_vision_towers[idx].vision_tower.vision_model.embeddings.position_embedding.weight.data.copy_(original_embedding[:num_positions])
                                
            self.model.to(device)

        # assert self.batch_size_per_gpu == 1, "Llava currently does not support batched generation. See https://github.com/haotian-liu/LLaVA/issues/754. HF Llava also has this issue."
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")

            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(f"Using {accelerator.num_processes} devices with tensor parallelism")
            self._rank = 0
            self._word_size = 1
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1
    
    # Method to log each stage's results
    def save_stage_to_csv(self, doc_id, stage, text_output, cumulative_confidences):
        with open(self.output_csv_path, mode="a", newline="") as file:
            writer = csv.writer(file)

            # Prepare the row with doc_id, stage, and text_output, followed by each cumulative confidence as separate columns
            row = [doc_id, stage, text_output] + cumulative_confidences
            writer.writerow(row)

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        """ """
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        try:
            return self.tokenizer.decode(tokens)
        except:
            return self.tokenizer.decode([tokens])

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # TODO
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, doc_to_target, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            # encode, pad, and truncate contexts for this batch
            if type(doc_to_target) == str:
                continuation = doc_to_target
            else:
                continuation = doc_to_target(self.task_dict[task][split][doc_id])
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            image_sizes = [[visual.size[0], visual.size[1]] for visual in visuals]
            if visuals:
                image = process_images(visuals, self._image_processor, self._config)
                if type(image) is list:
                    image = [_image.to(dtype=torch.float16, device=self.device) for _image in image]
                else:
                    image = image.to(dtype=torch.float16, device=self.device)
            else:
                image = None

            prompts_input = contexts[0] if isinstance(contexts, list) else contexts

            if image is not None and len(image) != 0 and DEFAULT_IMAGE_TOKEN not in prompts_input:
                """
                Three senarios:
                1. No image, and there for, no image token should be added.
                2. image token is already specified in the context, so we don't need to add it.
                3. image token is not specified in the context and there is image inputs, so we need to add it. In this case, we add the image token at the beginning of the context and add a new line.
                """
                image_tokens = [DEFAULT_IMAGE_TOKEN] * len(visuals)
                image_tokens = " ".join(image_tokens)
                prompts_input = image_tokens + "\n" + (contexts[0] if isinstance(contexts, list) else contexts)

            # This is much safer for llama3, as we now have some object type in it
            if "llama_3" in self.conv_template:
                conv = copy.deepcopy(conv_templates[self.conv_template])
            else:
                conv = conv_templates[self.conv_template].copy()
            conv.append_message(conv.roles[0], prompts_input)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            contxt_id = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
            # Add the answer of the second role
            conv.messages[1][1] = continuation

            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
            labels = input_ids.clone()
            # Context part no need to calculate for loss
            labels[0, : contxt_id.shape[1]] = -100
            with torch.inference_mode():
                outputs = self.model(input_ids=input_ids, labels=labels, images=image, use_cache=True, image_sizes=image_sizes)
            loss = outputs["loss"]
            # loss = torch.exp(loss)
            logits = outputs["logits"]
            greedy_tokens = logits.argmax(dim=-1)
            cont_toks = input_ids[:, contxt_id.shape[1] :]  # [1, seq]
            greedy_tokens = greedy_tokens[:, contxt_id.shape[1] : input_ids.shape[1]]  # [1, seq]
            max_equal = (greedy_tokens == cont_toks).all()
            res.append((float(loss.item()), bool(max_equal)))
            pbar.update(1)
        pbar.close()
        return res

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list
    
    ### Modification for recursive generation

    ## select target tokens based on the strategy (for multiple target tokens)
    ## default: select first token (POPE, VQA, ...)
    def select_target_tokens(self, text_outputs, scores=None, target_token_selection_strategy="first"):
        ## target token selection strategy
        target_output_indices = []
        if self.target_token_selection_strategy == "first":
            target_output_indices.append(0)

        elif self.target_token_selection_strategy == "all":
            target_output_indices = [i for i, _ in enumerate(text_outputs)]

        elif self.target_token_selection_strategy == "min_confidence":
            # Find the index of the token with the minimum confidence score
            assert scores is not None, "Confidence scores must be provided for min_confidence target token selection strategy"
            min_conf_index = scores.index(min(scores))
            target_output_indices.append(min_conf_index)
        elif self.target_token_selection_strategy == "classifier_based":
            pass
        elif self.target_token_selection_strategy == "max_attn_variance":
            pass
        elif self.target_token_selection_strategy == "prompt_attn":
            pass

        else:
            # Default or fallback behavior if strategy is unspecified
            target_output_indices = [0]
        
        return target_output_indices
    
    ## get offset for the current stage (non-recursed tokens)
    def get_offset(self, default_patch_size=14):
        offset = 0
        for image_size in self.use_all:
            offset += (image_size // default_patch_size) ** 2
        
        return offset

    ## initialize patch list and mask list for recursive generation
    ## mask_list = [size of first stage, size of second stage, ...]
    def create_patch_list_and_mask(self, patch_size=14):
        patch_list = []
        mask_list = []
        for idx, image_size in enumerate(self.all_image_sizes):
            num_patches = (image_size // patch_size) ** 2
            patch_list.append(num_patches)
            # Create mask for the current stage
            # first stage is fully visible
            if idx == 0:
                mask_list.append(torch.ones(num_patches, 2))
            else:
                mask_list.append(torch.zeros(num_patches, 2))
        return patch_list, mask_list
    
    ## stage starts from 1
    ## update_indices = list with (row,col)
    def update_mask(self, mask_list, stage, update_indices, mode="retain_base"):
        # Base mask for the current stage
        current_mask = mask_list[stage - 1]
        next_mask = mask_list[stage]
        
        # Sizes for the current and next resolutions
        current_size = current_mask.size(0)
        next_size = next_mask.size(0)
        scale_factor = next_size // current_size  # Assume square grids
        
        if mode == "remove_base":
            # Update the current resolution mask
            for row, col in update_indices:                
                current_mask[row, col] = 0  # Set corresponding index to 0            
        
        # Update the next resolution mask
        for row, col in update_indices:    
            start_row, start_col = row * scale_factor, col * scale_factor
            end_row, end_col = start_row + scale_factor, start_col + scale_factor
            next_mask[start_row:end_row, start_col:end_col] = 1  
        
        return mask_list
    
    def get_image_mask(self, ret_attn, cumulative_confidences, index=0):
        ## averages attention over the layers to determine threshold
        if self.attention_thresholding_type == "layer_mean":
            med = torch.stack(ret_attn, dim=0)
            med = med.mean(dim=0)
            # [0] indicates the first token generated (change to [1] if output includes <s>)
            attn = ret_attn[index] - med
            attn = torch.relu(attn)
            attn = attn / attn.max()

            image_mask_list = []
            for row in range(attn.shape[0]):
                for col in range(attn.shape[1]):
                    #print(f"attention value: {attn[row,col]}")
                    if attn[row, col] > self.attention_threshold:
                        #print(f"attention value: {attn[row,col]}")
                        image_mask_list.append(torch.LongTensor([[row, col]]))
            #print(image_mask_list)
            image_mask = torch.cat(image_mask_list)
            #print(f"num_divided: {len(image_mask_list)}")
        elif self.attention_thresholding_type == "layer_mean_with_top_k":  
            attn = ret_attn[index]
            print(self.attention_threshold)

            top_k_percent = self.attention_threshold
            ## Setting Threshold (Top 20%)
            flattened_attn = attn.view(-1) 
            flattened_attn = flattened_attn.float()
            threshold_index = int(len(flattened_attn) * top_k_percent) 
            threshold_value = torch.topk(flattened_attn, threshold_index).values[-1]

            image_mask_list = []
            for row in range(attn.shape[0]):
                for col in range(attn.shape[1]):
                    if attn[row, col] > threshold_value:
                        image_mask_list.append(torch.LongTensor([[row, col]]))
            image_mask = torch.cat(image_mask_list)
            
        elif self.attention_thresholding_type == "entropy_based_topk":  
            print(f"Entropy-based Top-K threshold (Base topk={self.attention_threshold}).")
            attn = ret_attn[index]
            calculated_threshold = entropy_based_threshold(attn, base_threshold=0.2, scaling_factor=0.5)
            print(f"Calculated Threshold: {calculated_threshold}")
            flattened_attn = attn.view(-1).float()
            threshold_index = int(len(flattened_attn) * calculated_threshold)
            threshold_value = torch.topk(flattened_attn, threshold_index).values[-1]

            image_mask_list = []
            for row in range(attn.shape[0]):
                for col in range(attn.shape[1]):
                    if attn[row, col] > threshold_value:
                        image_mask_list.append(torch.LongTensor([[row, col]]))
            image_mask = torch.cat(image_mask_list)
        elif self.attention_thresholding_type == "confidence_based_topk":  
            print(f"confidence-based Top-K threshold (Base topk={self.attention_threshold}).")
            attn = ret_attn[index]
            calculated_threshold = confidence_based_threshold(cumulative_confidences, base_threshold=self.attention_threshold)
            print(f"Calculated Threshold: {calculated_threshold}")
            flattened_attn = attn.view(-1).float()
            threshold_index = int(len(flattened_attn) * calculated_threshold)
            print(f"threshold_index: {threshold_index}")
            threshold_value = torch.topk(flattened_attn, threshold_index).values[-1]

            image_mask_list = []
            for row in range(attn.shape[0]):
                for col in range(attn.shape[1]):
                    if attn[row, col] > threshold_value:
                        image_mask_list.append(torch.LongTensor([[row, col]]))
            image_mask = torch.cat(image_mask_list)            
        else:
            image_mask = None
        
        return image_mask
    
    #### recursive generation end ####

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            batched_visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]  # [B, N]
            flattened_visuals = self.flatten(batched_visuals)  # [B*N]      

            ## original model accepts several diffrent grids (e.g. 1x2, 1x3, 2x2)
            ## for recursive implementation, we only use 2x2 grid (might be updated in future)
            ## Set grid to 2x2 for recursive generation, else default
            if self.fix_grid == "2x2":
                flattened_visuals = [make_square(visual, min_size=336) for visual in flattened_visuals]
            else:
                pass

            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]

            # Set default values for until and max_new_tokens
            until = [self.tok_decode(self.eot_token_id)]

            # Update values from gen_kwargs if present
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")

            if "image_aspect_ratio" in gen_kwargs.keys() and "image_aspect_ratio" not in self._config.__dict__:
                # here we should pop it out of gen_kwargs so that it doesn't get passed to the model for next step of generation
                self._config.image_aspect_ratio = gen_kwargs.pop("image_aspect_ratio")
                eval_logger.info(f"Setting image aspect ratio: {self._config.image_aspect_ratio}")
            # encode, pad, and truncate contexts for this batch
            if flattened_visuals:
                image_tensor = process_images(flattened_visuals, self._image_processor, self._config)
                if type(image_tensor) is list:
                    image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
                else:                    
                    image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)
                    if len(self.downsampled_image_sizes) > 1:
                        downsampled_image_tensors = []
                        for idx, image_size in enumerate(self.all_image_sizes):
                            downsampled_image_tensor = F.interpolate(image_tensor[0], size=(image_size, image_size), mode='bilinear', align_corners=False)
                            downsampled_image_tensor = downsampled_image_tensor.unsqueeze(0)
                            downsampled_image_tensor = downsampled_image_tensor.to(dtype=torch.float16, device=self.device)
                            downsampled_image_tensors.append(downsampled_image_tensor)
                    else:
                        downsampled_image_tensors = None
            else:
                image_tensor = None

            # prompts_input = contexts[0]

            question_input = []

            for visual, context in zip(batched_visuals, contexts):
                if image_tensor is not None and len(image_tensor) != 0 and DEFAULT_IMAGE_TOKEN not in context:
                    """
                    Three senarios:
                    1. No image, and there for, no image token should be added.
                    2. image token is already specified in the context, so we don't need to add it.
                    3. image token is not specified in the context and there is image inputs, so we need to add it. In this case, we add the image token at the beginning of the context and add a new line.
                    """
                    image_tokens = [DEFAULT_IMAGE_TOKEN] * len(visual) if isinstance(visual, list) else [DEFAULT_IMAGE_TOKEN]
                    image_tokens = " ".join(image_tokens)
                    question = image_tokens + "\n" + context
                else:
                    question = context
                # This is much safer for llama3, as we now have some object type in it
                if "llama_3" in self.conv_template:
                    conv = copy.deepcopy(conv_templates[self.conv_template])
                else:
                    conv = conv_templates[self.conv_template].copy()
                conv.append_message(conv.roles[0], question)
                conv.append_message(conv.roles[1], None)
                prompt_question = conv.get_prompt()
                question_input.append(prompt_question)

            # input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
            # preconfigure gen_kwargs with defaults
            gen_kwargs["image_sizes"] = [flattened_visuals[idx].size for idx in range(len(flattened_visuals))]
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            input_ids_list = [tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt") for prompt in question_input]
            pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            input_ids = self.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_ids).to(self.device)
            attention_masks = input_ids.ne(pad_token_ids).to(self.device)
            # These steps are not in LLaVA's original code, but are necessary for generation to work
            # TODO: attention to this major generation step...
            
            ## main generation part
            try:  
                if "recursion" in self.generation_type:
                    assert self.recursed_image_sizes is not None, "no recursed image size"
                
                    ## stage starts from 1
                    total_stages = len(self.recursed_image_sizes) + 1
                    
                    print(f"{total_stages} stage recursion")
                
                    print("recursive stage 1")
                    
                    cont = self.model.generate(
                        input_ids,
                        attention_mask=attention_masks,
                        pad_token_id=pad_token_ids,
                        images=image_tensor,
                        image_sizes=gen_kwargs["image_sizes"],
                        do_sample=True if gen_kwargs["temperature"] > 0 else False,
                        temperature=gen_kwargs["temperature"],
                        top_p=gen_kwargs["top_p"],
                        num_beams=gen_kwargs["num_beams"],
                        max_new_tokens=gen_kwargs["max_new_tokens"],
                        use_cache=self.use_cache,
                        generation_type=self.generation_type,
                        return_dict_in_generate=True,
                        output_attentions=True,
                        output_scores=True,
                        downsampled_images = downsampled_image_tensors,
                        all_image_sizes = self.all_image_sizes,
                        recursed_image_sizes = self.recursed_image_sizes,
                        current_stage = 1
                    )
                    
                    text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
                    scores = cont.scores
                    
                    offset = self.get_offset()
                    current_patches = self.get_num_patches()
                    
                    patch_list, mask_list = self.create_patch_list_and_mask
                    
                    # Calculate entropy and all cumulative confidences
                    P_T_given_I_Q_full, entropy_sum, cumulative_confidences = calculate_entropy_and_all_confidences(
                        cont["sequences"][0], scores = scores
                    )                                
                    
                    # Save first stage results to CSV
                    if self.save_output:
                        self.save_stage_to_csv("Stage 1", doc_id, text_outputs, cumulative_confidences)
                    
                    if self.save_heatmap_image:
                        input_image = flattened_visuals[0]
                        img_save_path = self.save_heatmap_image_path
                    else:
                        input_image = None  
                        img_save_path = None             

                    ## returns attention over image tokens
                    ret_attn = get_heatmap(self.model, cont, self.tokenizer, question_input[0], input_ids, current_patches, offset, input_image, img_save_path)
                    updated_indices = self.get_image_mask(ret_attn, cumulative_confidences, 0)
                
                    mask_list = self.update_mask(mask_list, stage, updated_indices, "remove_base" if "remove" in self.generation_type else "retain_base")
                    
                    ## detection
                    if self.detection_strategy and cumulative_confidences[0] > self.detection_threshold:
                        print("score over threshold, do not recurse")
                    else:
                        ## recursive generation  
                        for stage in range(1, total_stages):
                            stage = stage + 1
                            print(f"recursive stage {stage}")

                            last_stage = (stage == total_stages)

                            cont = self.model.generate(
                                input_ids,
                                attention_mask=attention_masks,
                                pad_token_id=pad_token_ids,
                                images=image_tensor,
                                image_sizes=gen_kwargs["image_sizes"],
                                do_sample=True if gen_kwargs["temperature"] > 0 else False,
                                temperature=gen_kwargs["temperature"],
                                top_p=gen_kwargs["top_p"],
                                num_beams=gen_kwargs["num_beams"],
                                max_new_tokens=gen_kwargs["max_new_tokens"],
                                use_cache=self.use_cache,
                                generation_type=self.generation_type,
                                return_dict_in_generate=True,
                                output_attentions=last_stage,
                                output_scores=True,
                                downsampled_images = downsampled_image_tensors,
                                all_image_sizes = self.all_image_sizes,
                                recursed_image_sizes = self.recursed_image_sizes,
                                current_stage = stage,
                                image_mask = mask_list
                            )

                            text_outputs = self.tokenizer.batch_decode(cont['sequences'], skip_special_tokens=True)
                            scores = cont.scores 

                            # Calculate entropy and all cumulative confidences
                            P_T_given_I_Q_full, entropy_sum, cumulative_confidences = calculate_entropy_and_all_confidences(
                                cont["sequences"][0], scores = scores
                            )                                
                            
                            # Save stage results to CSV
                            if self.save_output:
                                self.save_stage_to_csv(f"Stage {stage}", doc_id, text_outputs, cumulative_confidences)  

                            if not last_stage:  
                                ret_attn = get_heatmap(self.model, cont, self.tokenizer, question_input[0], input_ids, current_patches, offset, input_image, img_save_path)
                                updated_indices = self.get_image_mask(ret_attn, cumulative_confidences, 0)
                            
                                mask_list = self.update_mask(mask_list, stage, updated_indices, "remove_base" if "remove" in self.generation_type else "retain_base")                                            

                ## no recursion: remove output_attentions, return_dict params since passing them requires additional memory
                else:
                    #print("no recursion")
                    cont = self.model.generate(
                        input_ids,
                        attention_mask=attention_masks,
                        pad_token_id=pad_token_ids,
                        images=image_tensor,
                        image_sizes=gen_kwargs["image_sizes"],
                        do_sample=True if gen_kwargs["temperature"] > 0 else False,
                        temperature=gen_kwargs["temperature"],
                        top_p=gen_kwargs["top_p"],
                        num_beams=gen_kwargs["num_beams"],
                        max_new_tokens=gen_kwargs["max_new_tokens"],
                        use_cache=self.use_cache,
                        generation_type=self.generation_type,
                        return_dict_in_generate=self.save_output,
                        # output_attentions=True,
                        output_scores=self.save_output
                    )
                    
                    if self.save_output:                    
                        text_outputs = self.tokenizer.batch_decode(cont['sequences'], skip_special_tokens=True)
                        
                        # Calculate entropy and all cumulative confidences
                        P_T_given_I_Q_full, entropy_sum, cumulative_confidences = calculate_entropy_and_all_confidences(
                            cont["sequences"][0], cont.scores
                        )
                        self.save_stage_to_csv("Non-recursive", doc_id, text_outputs, cumulative_confidences)
                    else:
                        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
                   
            except Exception as e:
                cont = self.model.generate(
                        input_ids,
                        attention_mask=attention_masks,
                        pad_token_id=pad_token_ids,
                        images=image_tensor,
                        image_sizes=gen_kwargs["image_sizes"],
                        do_sample=True if gen_kwargs["temperature"] > 0 else False,
                        temperature=gen_kwargs["temperature"],
                        top_p=gen_kwargs["top_p"],
                        num_beams=gen_kwargs["num_beams"],
                        max_new_tokens=gen_kwargs["max_new_tokens"],
                        use_cache=self.use_cache,
                        generation_type="default",                       
                    )
                text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
                #raise e
                eval_logger.error(f"Error {e} in generating, generate with default")
                # cont = ""
                # text_outputs = [""]
                

            # cont_toks_list = cont.tolist()
            # for cont_toks, context in zip(cont_toks_list, contexts):
            # discard context + left-padding toks if using causal decoder-only LMM
            # if self.truncate_context:
            #     cont_toks = cont_toks[input_ids.shape[1] :]
            # use secondary stop seqs to cut off should-have-been-stopped content post-hoc
            # if self.truncate_context:
            #     for term in until:
            #         if len(term) > 0:
            #             # ignore '' separator,
            #             # for seq2seq case where self.tok_decode(self.eot_token_id) = ''
            #             text_outputs = text_outputs.split(term)[0]
            res.extend(text_outputs)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text_outputs)
            pbar.update(1)
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res