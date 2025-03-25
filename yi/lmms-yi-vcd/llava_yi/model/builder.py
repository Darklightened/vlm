from PIL import Image
from io import BytesIO
import base64
import math
import ast
import re
import torch
import copy
from llava_yi.model.constants import IMAGE_TOKEN_INDEX

import torch
from llava_yi.model import LlavaLlamaForCausalLM
from llava_yi.model.constants import IMAGE_TOKEN_INDEX
from llava_yi.model.llava_recursion import LlavaRecursionConfig, LlavaLlamaForRecursion
from transformers import AutoTokenizer, StoppingCriteria
from llava_yi.model.llava_llama import LlavaConfig

## Additional imports for utils
import requests
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
from matplotlib import cm

def load_pretrained_model(model_path, device_map="auto", recursion_config=None, **kwargs):
    kwargs = {"device_map": device_map}
    kwargs["torch_dtype"] = torch.bfloat16

    # model_path = '/root/.cache/huggingface/hub/models--01-ai--Yi-VL-6B/snapshots/dab34dabd32b391e4e870b7985180f90f79ad9a0'
    model_path = '/home/aidas_2/.cache/huggingface/hub/models--01-ai--Yi-VL-6B/snapshots/dab34dabd32b391e4e870b7985180f90f79ad9a0'
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = LlavaLlamaForCausalLM.from_pretrained(
        model_path, low_cpu_mem_usage=True, **kwargs
    )
    model = LlavaLlamaForRecursion.from_pretrained(
                pretrained_model_name_or_path=model_path,
                low_cpu_mem_usage=True,
                config=model.config,
                **kwargs
            )
    image_processor = None
    model.resize_token_embeddings(len(tokenizer))
    vision_tower = model.get_vision_tower()

    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device="cuda", dtype=torch.bfloat16)
    image_processor = vision_tower.image_processor

    model.init_all(recursion_config)
    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
