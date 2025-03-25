import torch
import sys

torch.backends.cuda.matmul.allow_tf32 = True


import copy
import warnings
from datetime import timedelta
from typing import List, Optional, Tuple, Union

from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from packaging import version
from tqdm import tqdm
import matplotlib.pyplot as plt

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.utils import stop_sequences_criteria
from lmms_eval.recursion_utils import *
import numpy as np
import cv2
import torch.nn.functional as F
import csv
import os
import ast
from pathlib import Path
import math
import json

warnings.filterwarnings("ignore")

def string_to_list(input_string):
    """
    Converts a string representation of a list into an actual Python list.
    
    Args:
        input_string (str): The input string, e.g., '[1, 2, 3]' or "['a', 'b', 'c']"
        
    Returns:
        list: The converted Python list.
        
    Raises:
        ValueError: If the string cannot be converted into a list.
    """
    try:
        # Use ast.literal_eval for safe evaluation of string as Python literal
        if input_string.startswith('['):
            return ast.literal_eval(input_string)
        else:
            raise ValueError("The input string does not start with '['")
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"Failed to convert string to list: {e}")
    
from loguru import logger as eval_logger


from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
    get_heatmap,
    make_square_center,
    make_square_top_left,
    make_square_bot_right,
    init_downsampled_vision_towers,
)
from llava.mm_utils import (
    norm_relu,
    norm_min_max,
)
from llava.model.builder import load_pretrained_model
from llava.model.language_model.llava_recursion import LlavaRecursionConfig, LlavaMistralForRecursion
    
# except Exception as e:
#     eval_logger.debug("LLaVA is not installed. Please install LLaVA to use this model.\nError: %s" % e)

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
        device_map="aidas",
        conv_template="vicuna_v1",
        use_cache=True,
        tie_weights: bool = True,
        truncate_context=False,  # whether to truncate the context in generation, set it False for LLaVA-1.6
        customized_config=None,  # ends in json        
        ## new args for recursive generation
        generation_type="default",
        fix_grid="2x2",
        attention_thresholding_type="layer_mean",
        attn_norm = "norm_relu", 
        attention_threshold=[0.1],
        remove_unpadding=False, 
        merging=None,
        save_output=True,
        output_csv_path = "generation_output.csv",
        output_json_path = "generation_output.json",
        target_token_selection_strategy="first",
        stages=[-1, 0, 1],
        positional_embedding_type="reduced",
        square=1,
        contrastive_alphas=[1.0,1.0,1.0],
        use_noised_for_contrastive=False,
        save_logit=False,       
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        #assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        elif accelerator.num_processes == 1 and device_map == "aidas":
            self._device = torch.device(device)
            self.device_map = device
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        llava_model_args = {
            "multimodal": True,
        }
        
        # Prepare model merging
        self.merging=merging  
        # If merging==None, then single model evaluation.
        if merging == "None":
            self.merging = None
        elif isinstance(merging, str):
            self.merging = merging
        print(f'self.merging= {self.merging}')

        if customized_config is not None:
            llava_model_args["customized_config"] = customized_config
        if attn_implementation is not None:
            llava_model_args["attn_implementation"] = attn_implementation
        if "use_flash_attention_2" in kwargs:
            llava_model_args["use_flash_attention_2"] = kwargs["use_flash_attention_2"]
        model_name = model_name if model_name is not None else get_model_name_from_path(pretrained)

        # default args
        self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template
        self.use_cache = use_cache
        self.truncate_context = truncate_context

        # recursion config
        self.recursion_config = LlavaRecursionConfig()
        self.recursion_config.generation_type = generation_type 
        self.recursion_config.attention_thresholding_type = attention_thresholding_type 
        self.recursion_config.attn_norm = attn_norm
        
        self.recursion_config.save_output = save_output
        self.recursion_config.output_csv_path = output_csv_path
        self.recursion_config.output_json_path = output_json_path
        assert 0 in stages, "stages must have 0, which means stage of 336."
        assert sorted(stages) == stages, "stages must be sorted."
        self.recursion_config.stages = stages
        self.recursion_config.positional_embedding_type = positional_embedding_type
        self.recursion_config.square = square
        self.recursion_config.fix_grid = fix_grid
        self.recursion_config.contrastive_alphas = contrastive_alphas
        self.recursion_config._device = device
        self.recursion_config.use_noised_for_contrastive = use_noised_for_contrastive
        if attention_threshold[0] == '[' :
            self.recursion_config.attention_threshold = string_to_list(attention_threshold)
            
        else:
            self.recursion_config.attention_threshold = [float(attention_threshold) for i in range(len(self.recursion_config.stages)-1)]
            

        print(self.recursion_config)
        
        print(f"device: {device}")
        print(f"generation_type: {generation_type}")
        print(f"fix_grid: {fix_grid}")
        print(f"attention_thresholding_type: {attention_thresholding_type}")
        print(f"attention_norm: {attn_norm}")
        print(f"attention_threshold: {attention_threshold}")        
        print(f"save_output: {save_output}")
        print(f"save_output_csv_path: {output_csv_path}")        
        print(f"stages: {stages}")
        print(f"positional_embedding_type: {positional_embedding_type}")        
        print(f"square: {square}")

        if self.recursion_config.square == 1:
            self.make_square = make_square_center
        elif self.recursion_config.square == 2:
            self.make_square = make_square_top_left
        elif self.recursion_config.square == 3:
            self.make_square = make_square_bot_right

        # try:
        # Try to load the model with the multimodal argument
        self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(pretrained, None, model_name, 
                                                                                                        device_map=self.device_map, 
                                                                                                        merging=self.merging, 
                                                                                                        recursion_config=self.recursion_config,
                                                                                                        **llava_model_args)
    # except TypeError:
        #     # for older versions of LLaVA that don't have multimodal argument
        #     llava_model_args.pop("multimodal", None)
        #     self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(pretrained, None, model_name, device_map=self.device_map, **llava_model_args)
        self._config = self._model.config
        self.model.eval()
        if tie_weights:
            self.model.tie_weights()
        
        if remove_unpadding == True:
            print("remove unpadding=True, change to 'spatial'")
            self._model.config.mm_patch_merge_type = "spatial"
        
        ## save output confidences to csv
        if save_output:
            assert output_csv_path is not None, "Output CSV path is not provided."

            output_csv_path = Path(output_csv_path)

            if not output_csv_path.exists():
                with open(output_csv_path, mode="w", newline="") as file:
                    writer = csv.writer(file)
                    headers = ["Doc ID", "Stage", "Text Output"]   
                    headers += [f"Cumulative Confidence {i+1}" for i in range(10)]
                    writer.writerow(headers)
            
            ## save output confidences to csv
            if save_output:
                assert output_json_path is not None, "Output JSON path is not provided."

                output_json_path = Path(output_json_path)

                with open(output_json_path, mode="w") as file:
                    json.dump({}, file)
        
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
            # self.model.model.vision_tower.to(self._device)
            # if self.stages[0] < 0:
            #     for stage in self.stages:
            #         if stage == 0:
            #             break
            #         self.model.model.downsampled_vision_towers[str(stage)].to(self._device)
            self._rank = 0
            self._world_size = 1
    
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
        for idx_chunk, chunk in enumerate(chunks):
            ## reset image mask and pad mask
            ## self.reset_image_mask()
            
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            batched_visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]  # [B, N]
            flattened_visuals = self.flatten(batched_visuals)  # [B*N]
            if len(flattened_visuals) != 1:
                res.extend(["skipped"])
                pbar.update(1)
                continue
            
            ## original model accepts several diffrent grids (e.g. 1x2, 1x3, 2x2)
            ## for recursive implementation, we only use 2x2 grid (might be updated in future)
            ## Set grid to 2x2 for recursive generation, else default
            if self.recursion_config.fix_grid == "2x2":
                temp_list = []
                for idx, visual in enumerate(flattened_visuals):
                    if idx == 0:
                        squared_img, size, origin_x, origin_y = self.make_square(visual, min_size=384, smallest_grid_size=self.model.smallest_grid_size)
                        # squared_img.save("./test.png")
                        # exit()
                        temp_list.append(squared_img)
                        self.model.set_pad_mask(size, origin_x, origin_y)
                    else:
                        squared_img, _, _ = self.make_square(visual, min_size=384, smallest_grid_size=self.model.smallest_grid_size)
                        temp_list.append(squared_img)
                flattened_visuals = temp_list
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
                    downsampled_image_tensors = dict()
                    for stage in self.recursion_config.stages:
                        if stage == 0:
                            break
                        temp_image_size = int(self.model.image_size * pow(2, stage))
                        downsampled_image_tensor = F.interpolate(image_tensor[0], size=(temp_image_size, temp_image_size), mode='bilinear', align_corners=False)
                        downsampled_image_tensor = downsampled_image_tensor.unsqueeze(0)
                        downsampled_image_tensor = downsampled_image_tensor.to(dtype=torch.float16, device=self.device)
                        downsampled_image_tensors[stage] = downsampled_image_tensor
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

            self.model.half()
            # for name, param in self.model.named_parameters():
            #     print(f"{name}: {param.dtype}")            

            text_outputs = self.model.generate_recursive(
                input_ids,
                attention_mask=attention_masks,
                pad_token_id=pad_token_ids,
                images=image_tensor,
                gen_kwargs = gen_kwargs,               
                max_length = gen_kwargs['max_new_tokens'],
                use_cache=self.use_cache,                 
                downsampled_images = downsampled_image_tensors,
                doc_id = doc_id,
                flattened_visuals = flattened_visuals,
                tokenizer = self.tokenizer,
                question_input = question_input,     
                task = task,        
            )

            res.extend(text_outputs)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text_outputs)
            pbar.update(1)
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res
    
    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for LLaVAHF")