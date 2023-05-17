import json
import logging
import os
import pathlib
import re
from copy import deepcopy
from pathlib import Path
from typing import Optional, Tuple

import torch

from .model import CLIP
from misc.transforms import create_transforms
from misc.constants import IMAGENET_COLOR_MEAN, IMAGENET_COLOR_STD

_MODEL_CONFIG_PATHS = [Path(__file__).parent.parent / f"model_configs/"]
_MODEL_CONFIGS = {}  # directory (model_name: config) of model architecture configs

_IMAGE_TRANSFORM_TRAIN = [
    'pad_to_minimum',
    'horizontal_flip',
    'vertical_flip',
    'normalize'
]

_IMAGE_TRANSFORM_VAL = [
    'pad_to_minimum',
    'normalize'
]

def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def _rescan_model_configs():
    global _MODEL_CONFIGS

    config_ext = ('.json',)
    config_files = []
    for config_path in _MODEL_CONFIG_PATHS:
        if config_path.is_file() and config_path.suffix in config_ext:
            config_files.append(config_path)
        elif config_path.is_dir():
            for ext in config_ext:
                config_files.extend(config_path.glob(f'*{ext}'))

    for cf in config_files:
        with open(cf, 'r') as f:
            model_cfg = json.load(f)
            if all(a in model_cfg for a in ('embed_dim', 'vision_cfg', 'text_cfg')):
                _MODEL_CONFIGS[cf.stem] = model_cfg

    _MODEL_CONFIGS = {k: v for k, v in sorted(_MODEL_CONFIGS.items(), key=lambda x: _natural_key(x[0]))}


_rescan_model_configs()  # initial populate of model config registry


def load_state_dict(checkpoint_path: str, map_location='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    if next(iter(state_dict.items()))[0].startswith('module'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    return state_dict


def load_checkpoint(model, checkpoint_path, strict=True):
    state_dict = load_state_dict(checkpoint_path)
    # resize_pos_embed(state_dict, model)
    incompatible_keys = model.load_state_dict(state_dict, strict=strict)
    return incompatible_keys


def create_model(
        model_name: str,
        device: torch.device = torch.device('cpu'),
        override_image_size = None,
):
    model_name = model_name.replace('/', '-')  # for callers using old naming with / in ViT names

    if model_name in _MODEL_CONFIGS:
        logging.info(f'Loading {model_name} model config.')
        model_cfg = deepcopy(_MODEL_CONFIGS[model_name])
    else:
        logging.error(f'Model config for {model_name} not found; available models {list_models()}.')
        raise RuntimeError(f'Model config for {model_name} not found.')
    
    if override_image_size:
        model_cfg['vision_cfg']['image_size'] = override_image_size
        logging.info(f'Created model {model_name} with image size of {override_image_size} instead')
        
    model = CLIP(**model_cfg)
    model.to(device=device)
    model.visual.image_mean = IMAGENET_COLOR_MEAN
    model.visual.image_std = IMAGENET_COLOR_STD
    return model


def create_model_and_transforms(
        model_name: str,
        device: torch.device = torch.device('cpu'),
        image_mean: Optional[Tuple[float, ...]] = None,
        image_std: Optional[Tuple[float, ...]] = None):
    
    model = create_model(model_name, device)

    image_mean = image_mean or getattr(model.visual, 'image_mean', None)
    image_std = image_std or getattr(model.visual, 'image_std', None)
    preprocess_train = create_transforms(_IMAGE_TRANSFORM_TRAIN, img_size = model.visual.image_size, mean=image_mean, std=image_std)
    preprocess_val = create_transforms(_IMAGE_TRANSFORM_VAL, img_size = model.visual.image_size, mean=image_mean, std=image_std)

    return model, preprocess_train, preprocess_val

def list_models():
    """ enumerate available model architectures based on config files """
    return list(_MODEL_CONFIGS.keys())
