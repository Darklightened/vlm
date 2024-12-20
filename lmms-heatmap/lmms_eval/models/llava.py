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

from torchvision.datasets import CocoDetection
import monai
import numpy as np
from PIL import Image, ImageDraw

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.utils import stop_sequences_criteria
from lmms_eval.recursion_utils import *
import numpy as np
import torch.nn.functional as F
import csv
import os
import ast
from pathlib import Path

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

try:
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import (
        get_model_name_from_path,
        process_images,
        tokenizer_image_token,
        get_heatmap,
        make_square_center,
        make_square_oneside,
        init_downsampled_vision_towers,
        norm_relu,
        norm_min_max,
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
        detection_strategy=None,
        merging=None,
        detection_threshold=0.8,
        save_output=True,
        output_csv_path = "generation_output.csv",
        target_token_selection_strategy="first",
        stages=[-1, 0, 1],
        positional_embedding_type="reduced",
        visualize_heatmap=False,
        square=1,
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
        
        try:
            # Try to load the model with the multimodal argument
            self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(pretrained, None, model_name, 
                                                                                                          device_map=self.device_map, 
                                                                                                          merging=self.merging, 
                                                                                                          **llava_model_args)
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
        self.attn_norm = attn_norm
        self.attention_threshold = attention_threshold      
        self.detection_strategy = detection_strategy
        self.detection_threshold = detection_threshold 
        self.save_output = save_output
        self.output_csv_path = output_csv_path
        self.target_token_selection_strategy = target_token_selection_strategy
        assert 0 in stages, "stages must have 0, which means stage of 336."
        assert sorted(stages) == stages, "stages must be sorted."
        self.stages = stages
        self.positional_embedding_type = positional_embedding_type
        self.visualize_heatmap = visualize_heatmap
        self.square = square
        
        print(f"device: {device}")
        print(f"generation_type: {generation_type}")
        print(f"fix_grid: {fix_grid}")
        print(f"attention_thresholding_type: {attention_thresholding_type}")
        print(f"attention_norm: {attn_norm}")
        print(f"attention_threshold: {attention_threshold}")
        print(f"detection_strategy: {detection_strategy}")        
        print(f"detection_threshold: {detection_threshold}")
        print(f"save_output: {save_output}")
        print(f"save_output_csv_path: {output_csv_path}")
        print(f"target_token_selection_strategy: {target_token_selection_strategy}")
        print(f"stages: {stages}")
        print(f"positional_embedding_type: {positional_embedding_type}")
        print(f"visualize_heatmap: {visualize_heatmap}")
        print(f"square: {square}")
        
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
                    
        if attn_norm == "None":
            self.attn_norm = None
        elif attn_norm == "norm_relu":
            self.attn_norm = norm_relu
        elif attn_norm == "norm_min_max":
            self.attn_norm = norm_min_max
        else:
            eval_logger.info(f"Unsupported norm type. Using norm_relu.")
            self.attn_norm = norm_relu
            
        if attention_threshold[0] == '[' :
            self.attention_threshold = string_to_list(attention_threshold)
            # print(self.attention_threshold)
        else:
            self.attention_threshold = [float(attention_threshold) for i in range(len(self.stages)-1)]
            # print(self.attention_threshold)
            
        if self.square == 1:
            self.make_square = make_square_center
        elif self.square == 2:
            self.make_square = make_square_oneside
        
        self.dice_ce = 0
        self.dice = 0
        self.cnt = 0
        
        ##################################################################################
        ## init downsampled vision towers
        ## stage -2:  84
        ## stage -1: 168
        ## stage  0: 336
        ## stage  1: 672
        if self.stages[0] < 0:
            self.model.model.downsampled_vision_towers = init_downsampled_vision_towers(
                self.model.model.vision_tower,
                self.stages,
                self.positional_embedding_type,
                device,
            )
        
        self.patch_size = 14
        self.image_size = self.model.model.vision_tower.vision_tower.vision_model.embeddings.image_size
        self.smallest_grid_size = int(self.image_size * pow(2, stages[0]) // self.patch_size)
        self.largest_grid_size = int(self.image_size * pow(2, stages[-1]) // self.patch_size)
        self.reset_image_mask()
        ## downsampled vision tower end
        ##################################################################################

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
        # pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        
        data_dir = '/workspace/vlm/val2017/'
        instances_file = '/workspace/vlm/annotations/instances_val2017.json' #Segmentation, category_id, bbox
        
        dataset = CocoDetection(root=data_dir, annFile=instances_file)
        coco = dataset.coco
        
        # Categories
        categories = coco.loadCats(coco.getCatIds())
        categories_list = [cat['name'] for cat in categories]

        for img_idx in tqdm(range(len(dataset)), desc="Processing"):
            if img_idx == 50:break
            image, target = dataset[img_idx]
            
            # Create semantic mask from instance mask.
            mask_dict = {}
            for obj in target:
                # Get category name
                category_id = coco.getCatIds(catIds=[obj["category_id"]])
                name = coco.loadCats(category_id)[0]['name']
                
                # Get instance mask
                segmentation = obj["segmentation"]
                if type(segmentation) != list:
                    continue
                mask = Image.new('L', (image.size), 0)
                draw = ImageDraw.Draw(mask)
                draw.polygon(segmentation[0], fill=255)
                mask = np.array(mask) / 255
                mask = mask.astype(np.uint8)
                
                # Merge to semantic mask
                semantic_mask = mask_dict.get(name)
                if semantic_mask is not None:
                    semantic_mask = semantic_mask | mask
                else:
                    semantic_mask = mask
                
                mask_dict[name] = semantic_mask
            
            # Create sorted list by mask area
            positive_list = sorted(mask_dict, key=lambda x: mask_dict[x].sum(), reverse=True)
            
            for cat_idx, cat_name in enumerate(positive_list):
                self.reset_image_mask()
                mask = mask_dict.get(cat_name, np.zeros((image.height, image.width), dtype=np.uint8))
                prompt_text = f"Does there exist {cat_name} in the image? Answer in 'Yes' or 'No'"
                #print(f"\n{prompt_text}\n")

                # input_ids, prompt = preprocess_prompt(model, model_name, prompt_text, tokenizer)
                # image, image_tensor = preprocess_image(model, image_processor, image)
                # image_size = image.size
                ################################################
                

                ## original model accepts several diffrent grids (e.g. 1x2, 1x3, 2x2)
                ## for recursive implementation, we only use 2x2 grid (might be updated in future)
                ## Set grid to 2x2 for recursive generation, else default
                image_h, image_w = image.size
                if image_h < 360 or image_w < 360: continue
                flattened_visuals = [image]
                
                if self.fix_grid == "2x2":
                    temp_list = []
                    for idx, visual in enumerate(flattened_visuals):
                        if idx == 0:
                            squared_img, bounding_x, bounding_y = self.make_square(visual, min_size=360, smallest_grid_size=self.smallest_grid_size)
                            # squared_img.save("./test.png")
                            # exit()
                            temp_list.append(squared_img)
                            self.set_pad_mask(bounding_x, bounding_y)
                        else:
                            squared_img, _, _ = self.make_square(visual, min_size=360, smallest_grid_size=self.smallest_grid_size)
                            temp_list.append(squared_img)
                    flattened_visuals = temp_list
                else:
                    pass

                if flattened_visuals:
                    image_tensor = process_images(flattened_visuals, self._image_processor, self._config)
                    if type(image_tensor) is list:
                        image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
                    else:                    
                        image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)
                        downsampled_image_tensors = dict()
                        for stage in self.stages:
                            if stage == 0:
                                break
                            temp_image_size = int(self.image_size * pow(2, stage))
                            downsampled_image_tensor = F.interpolate(image_tensor[0][0].unsqueeze(0), size=(temp_image_size, temp_image_size), mode='bilinear', align_corners=False)
                            downsampled_image_tensor = downsampled_image_tensor.unsqueeze(0)
                            downsampled_image_tensor = downsampled_image_tensor.to(dtype=torch.float16, device=self.device)
                            downsampled_image_tensors[stage] = downsampled_image_tensor
                else:
                    image_tensor = None

                context = prompt_text
                image_tokens = [DEFAULT_IMAGE_TOKEN]
                image_tokens = " ".join(image_tokens)
                question = image_tokens + "\n" + context
                
                conv = conv_templates[self.conv_template].copy()
                
                conv.append_message(conv.roles[0], question)
                conv.append_message(conv.roles[1], None)
                prompt_question = conv.get_prompt()
                question_input = [prompt_question]
                
                input_ids_list = [tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt") for prompt in question_input]
                pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
                input_ids = self.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_ids).to(self.device)
                attention_masks = input_ids.ne(pad_token_ids).to(self.device)
                
                image_sizes = [flattened_visuals[idx].size for idx in range(len(flattened_visuals))]
            
                ## main generation part
                # try:
                if "recursion" in self.generation_type:
                    for idx_stage, stage in enumerate(self.stages):
                        
                        ## for debugging with visualization
                        if self.visualize_heatmap:
                            save_path = f"./heatmap_vis/img_{str(img_idx).zfill(4)}/cat_{cat_name}/{str(idx_stage).zfill(2)}_stage_{stage}/"
                            os.makedirs(save_path, exist_ok=True)
                        else:
                            save_path = None

                        ## Necessary for unpad
                        # self.combine_image_and_pad_mask()
                        
                        last_stage = (stage == self.stages[-1])

                        cont = self.model.generate(
                            input_ids,
                            attention_mask=attention_masks,
                            pad_token_id=pad_token_ids,
                            images=image_tensor,
                            image_sizes=image_sizes,
                            do_sample=False,
                            temperature=0,
                            top_p=None,
                            num_beams=1,
                            max_new_tokens=16,
                            use_cache=self.use_cache,
                            generation_type=self.generation_type,
                            return_dict_in_generate=True,
                            output_attentions=True,
                            output_scores=True,
                            downsampled_images = downsampled_image_tensors,
                            image_mask = self.image_mask
                        )
                        
                        ## delete sos
                        if cont["sequences"][0][0] == 1:
                            cont["sequences"] = cont["sequences"][0][1:].unsqueeze(0)
                            
                        text_outputs = self.tokenizer.batch_decode(cont['sequences'], skip_special_tokens=True)
                        scores = cont.scores
                        sequences = cont["sequences"][0]       
                        
                        # For confidence-based topk attention threshold.
                        if last_stage:
                            ret_attn = get_heatmap(
                                        self.model,
                                        cont,
                                        self.tokenizer,
                                        question_input[0],
                                        input_ids,
                                        stage,
                                        self.stages,
                                        self.image_mask,
                                        select_token=None,
                                        image=flattened_visuals[0],
                                        save_path=save_path,
                                        attn_norm=None,
                                    )
                            h, w = mask.shape
                            target = max(h, w)
                            
                            ret_attn = torch.relu(ret_attn)
                            ret_attn /= ret_attn.max()
                            ret_attn -= 0.3
                            ret_attn = torch.relu(ret_attn)
                            ret_attn /= ret_attn.max()
                            ret_attn = torch.nn.functional.interpolate(
                                ret_attn.unsqueeze(0).unsqueeze(0),
                                size=(target, target),
                                mode='bilinear',
                                align_corners=False).squeeze()
                            
                            attn_h, attn_w = ret_attn.size()
                            ret_attn_bin = (ret_attn > 0).float()
                            ret_attn_bin = ret_attn_bin.unsqueeze(0).unsqueeze(0)
                            ret_attn_bin = ret_attn_bin.to("cpu")
                            ret_attn = ret_attn.unsqueeze(0).unsqueeze(0)
                            ret_attn = ret_attn.to("cpu")
                            dice_ce_loss = monai.losses.DiceCELoss()
                            dice_loss = monai.losses.DiceLoss()
                            
                            h, w = mask.shape
                            mask_resize = mask.copy()
                            
                            # Resize mask to square size
                            if h != attn_h:
                                half_h = (attn_h - h) // 2
                                block1 = np.zeros((half_h, w), dtype=np.uint8)
                                block2 = np.zeros((attn_h - h - half_h, w), dtype=np.uint8)
                                mask_resize = np.vstack((block1, mask_resize, block2))
                            if w != attn_w:
                                half_w = (attn_w - w) // 2
                                block1 = np.zeros((h, half_w), dtype=np.uint8)
                                block2 = np.zeros((h, attn_w - w - half_w), dtype=np.uint8)
                                mask_resize = np.hstack((block1, mask_resize, block2))
                            
                            mask_viz = mask_resize * 255
                            mask_viz = Image.fromarray(mask_viz)
                            if self.visualize_heatmap:
                                mask_viz.save(save_path + "mask.png")
                            
                            mask = torch.tensor(mask_resize, dtype=torch.float32, device="cpu").unsqueeze(0).unsqueeze(0)
                            dice_ce = dice_ce_loss(ret_attn, mask)
                            dice = dice_loss(ret_attn_bin, mask)
                            
                            self.dice_ce += dice_ce
                            self.dice += dice
                            self.cnt += 1
                            print(f"[{self.cnt}]  {cat_name}", dice_ce, dice)
                            del cont
                            break
                        
                        ret_attn = get_heatmap(
                                        self.model,
                                        cont,
                                        self.tokenizer,
                                        question_input[0],
                                        input_ids,
                                        stage,
                                        self.stages,
                                        self.image_mask,
                                        select_token=None,
                                        image=flattened_visuals[0],
                                        save_path=save_path,
                                        attn_norm=self.attn_norm,
                                    )

                        ### Threshold-based Recursion ######################################################               
                        if self.attention_thresholding_type == "layer_mean":
                            self.image_mask[stage+1] = layer_mean_based_recursion(attn = ret_attn, # select token index
                                                    attn_threshold = self.attention_threshold[idx_stage], 
                                                    image_mask = self.image_mask[stage+1])
                            
                        elif self.attention_thresholding_type == "layer_mean_topk":
                            self.image_mask[stage+1] = layer_mean_topk_based_recursion(attn = ret_attn, # select token index
                                                    top_k = self.attention_threshold[idx_stage], 
                                                    image_mask = self.image_mask[stage+1])
                        
                        elif self.attention_thresholding_type == "confidence_topk": 
                            self.image_mask[stage+1] = confidence_topk_based_recursion(attn = ret_attn, # select token index
                                                    top_k = self.attention_threshold[idx_stage], 
                                                    sequences = sequences,
                                                    scores = scores, 
                                                    image_mask = self.image_mask[stage+1])
                                
                        else: 
                            self.activate_image_mask(self.stages[idx_stage + 1])
                            
                        del cont

        print(f"Dice CE: {self.dice_ce / self.cnt}")
        print(f"Dice: {self.dice / self.cnt}")
        print(f"Count: {self.cnt}")
        exit()
        res = re_ords.get_original(res)
        return res
    
    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for LLaVAHF")
    
    def reset_image_mask(self):
        self.image_mask = dict()
        for idx, stage in enumerate(self.stages):
            if idx == 0:
                # activate every token of the first stage
                self.image_mask[stage] = torch.ones(self.largest_grid_size, self.largest_grid_size, device=self.device)        
            else:
                self.image_mask[stage] = torch.zeros(self.largest_grid_size, self.largest_grid_size, device=self.device)
        self.pad_mask = torch.ones(self.largest_grid_size, self.largest_grid_size, device=self.device)
    
    def set_pad_mask(self, bounding_x, bounding_y):
        temp_tensor = torch.zeros(self.smallest_grid_size, self.smallest_grid_size, device=self.device)
        temp_tensor[:bounding_y, :bounding_x] = 1
        temp_tensor = F.interpolate(temp_tensor.unsqueeze(0).unsqueeze(0), size=(self.largest_grid_size, self.largest_grid_size), mode='nearest').squeeze()
        self.pad_mask = self.pad_mask * temp_tensor
    
    def combine_image_and_pad_mask(self):
        for stage in self.stages:
            self.image_mask[stage] = self.image_mask[stage] * self.pad_mask
    
    def activate_every_image_masks(self):
        self.image_mask = dict()
        for stage in self.stages:
            self.image_mask[stage] = torch.ones(self.largest_grid_size, self.largest_grid_size, device=self.device)        
    
    def activate_image_mask(self, stage):
        self.image_mask[stage] = torch.ones(self.largest_grid_size, self.largest_grid_size, device=self.device)        

        