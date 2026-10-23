#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to fine-tune Stable Diffusion for InstructPix2Pix."""

import argparse
import logging
import math
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sub in ("models", "dataset"):
    _path = os.path.join(_ROOT, _sub)
    if _path not in sys.path:
        sys.path.insert(0, _path)
import shutil
from contextlib import nullcontext
from pathlib import Path

import accelerate
import datasets
import numpy as np
import PIL
from PIL import Image
import imageio
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from torch.utils.data import ConcatDataset

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionInstructPix2PixPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

from relighting_datasets import VIDITRelightingDataset,SyntheticPortritRelightingDataset,ObjectRelightingDataset,ObjectSyntheticRelightingDataset
from decoupled_attention import inject_harmolight_conditioning, load_harmolight_unet, read_diffusion_state_dict
from multimodal_conditions import curriculum_embedding_from_one_hot, inference_condition_kwargs

from pipeline_harmolight import StableDiffusionInstructPix2PixPipeline

import json

from transformers import AutoModelForImageSegmentation
from torchvision import transforms

seg_model = AutoModelForImageSegmentation.from_pretrained(os.environ.get("SEG_MODEL_PATH", "pretrained/RMBG-2.0"), trust_remote_code=True)
seg_model.to('cuda')
seg_model.eval()

transform_image = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

if is_wandb_available():
    import wandb

logger = get_logger(__name__, log_level="INFO")

def _validation_condition_kwargs(item, default_curriculum="relighted"):
    try:
        return inference_condition_kwargs(item, default_curriculum=default_curriculum)
    except Exception as error:
        logger.warning(f"Skip decoupled validation conditions: {error}")
        return {}

def encode_foreground_clip(model, images):
    images = (images + 1.0) * 0.5
    images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
    mean = images.new_tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
    std = images.new_tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
    images = (images - mean) / std
    features = model.get_image_features(pixel_values=images)
    if not torch.is_tensor(features):
        features = getattr(features, "image_embeds", None)
        if features is None:
            raise RuntimeError("CLIP image encoder did not return image embeddings.")
    return features

DATASET_NAME_MAPPING = {
    "fusing/instructpix2pix-1000-samples": ("input_image", "edit_prompt", "edited_image"),
}
WANDB_TABLE_COL_NAMES = ["original_image", "edited_image", "edit_prompt"]

def log_validation1(
    pipeline,
    args,
    accelerator,
    generator,
    epoch,
):
    logger.info(
        f"Running validation... a test synthetic portrait image"
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    data = []
    with open(args.val_syn_portrait_image_url, 'rt') as f:
        for line in tqdm(f, desc='Loading data'):
            data.append(json.loads(line))
    item = data[0]

    target_filename = item['target']
    source_filename = item['fg']
    light_filename = item['bg']
    mask_filename = item['mask']

    mask = imageio.v3.imread(mask_filename) / 255.
    non_mask = 1 - mask
    source = imageio.v3.imread(source_filename)[..., :3] / 255.
    light = imageio.v3.imread(light_filename)[..., :3] / 255.

    source = source[:, :, :3] * mask[:, :, np.newaxis]
    light = light[:, :, :3] * non_mask[:, :, np.newaxis]

    target = imageio.v3.imread(target_filename)[..., :3] / 255.

    conds = [source, light]
    conds = np.concatenate(conds, axis=2).astype(np.float32)[None]
    conds = torch.from_numpy(conds).to(torch.float16).permute(0, 3, 1, 2).to(accelerator.device)

    original_image = conds

    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)

    prompt = args.validation_prompt

    condition_kwargs = _validation_condition_kwargs(item)
    with autocast_ctx:
        edited_images=pipeline(
            prompt,
            image=original_image,
            num_inference_steps=50,
            image_guidance_scale=1.5,
            guidance_scale=7,
            generator=generator,
            **condition_kwargs,
        ).images[0]

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            images = edited_images

            src_image = source

            target_image = target

            formatted_images = []

            images = np.array(images) / 255.

            formatted_images.append(src_image)

            formatted_images.append(light)

            formatted_images.append(images)

            formatted_images.append(target_image)

            formatted_images = np.stack(formatted_images)

            tracker.writer.add_images(prompt+'test synthetic portrait image', formatted_images,epoch, dataformats="NHWC")

def log_validation2(
    pipeline,
    args,
    accelerator,
    generator,
    epoch,
):
    logger.info(
        f"Running validation... a training synthetic portrait image"
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    data = []
    with open(args.dataset_name, 'rt') as f:
        for line in tqdm(f, desc='Loading data'):
            data.append(json.loads(line))
    item = data[0]

    target_filename = item['target']
    source_filename = item['fg']
    light_filename = item['bg']
    mask_filename = item['mask']

    mask = imageio.v3.imread(mask_filename) / 255.
    non_mask = 1 - mask
    source = imageio.v3.imread(source_filename)[..., :3] / 255.
    light = imageio.v3.imread(light_filename)[..., :3] / 255.

    light = Image.fromarray((light * 255).astype(np.uint8))
    light = light.resize((512, 512))
    light = np.array(light) / 255.

    source = source[:, :, :3] * mask[:, :, np.newaxis]
    light = light[:, :, :3] * non_mask[:, :, np.newaxis]

    target = imageio.v3.imread(target_filename)[..., :3] / 255.

    conds = [source, light]
    conds = np.concatenate(conds, axis=2).astype(np.float32)[None]
    conds = torch.from_numpy(conds).to(torch.float16).permute(0, 3, 1, 2).to(accelerator.device)

    original_image = conds

    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)

    prompt = args.validation_prompt

    condition_kwargs = _validation_condition_kwargs(item)
    with autocast_ctx:
        edited_images=pipeline(
            prompt,
            image=original_image,
            num_inference_steps=50,
            image_guidance_scale=1.5,
            guidance_scale=7,
            generator=generator,
            **condition_kwargs,
        ).images[0]

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            images = edited_images
            src_image = source

            target_image = target

            formatted_images = []

            images = np.array(images) / 255.

            formatted_images.append(src_image)

            formatted_images.append(light)

            formatted_images.append(images)

            formatted_images.append(target_image)

            formatted_images = np.stack(formatted_images)

            tracker.writer.add_images(prompt+'train synthetic portrait image', formatted_images,epoch, dataformats="NHWC")

def log_validation3(
    pipeline,
    args,
    accelerator,
    generator,
    epoch,
):
    logger.info(
        f"Running validation... a test object image"
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    data = []
    with open(args.val_object_image_url, 'rt') as f:
        for line in tqdm(f, desc='Loading data'):
            data.append(json.loads(line))
    item = data[0]

    target_filename = item['target']
    source_filename = item['fg']
    light_filename = item['bg']
    mask_filename = item['mask']

    mask = imageio.v3.imread(mask_filename) / 255.
    non_mask = 1 - mask
    source = imageio.v3.imread(source_filename)[..., :3] / 255.
    light = imageio.v3.imread(light_filename)[..., :3] / 255.

    source = source[:, :, :3] * mask[:, :, np.newaxis]
    light = light[:, :, :3] * non_mask[:, :, np.newaxis]

    target = imageio.v3.imread(target_filename)[..., :3] / 255.

    conds = [source, light]
    conds = np.concatenate(conds, axis=2).astype(np.float32)[None]
    conds = torch.from_numpy(conds).to(torch.float16).permute(0, 3, 1, 2).to(accelerator.device)

    original_image = conds

    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)

    prompt = item['lighting_description']

    condition_kwargs = _validation_condition_kwargs(item)
    with autocast_ctx:
        edited_images=pipeline(
            prompt,
            image=original_image,
            num_inference_steps=50,
            image_guidance_scale=1.5,
            guidance_scale=7,
            generator=generator,
            **condition_kwargs,
        ).images[0]

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            images = edited_images

            src_image = source

            target_image = target

            formatted_images = []

            images = np.array(images) / 255.

            formatted_images.append(src_image)

            formatted_images.append(light)

            formatted_images.append(images)

            formatted_images.append(target_image)

            formatted_images = np.stack(formatted_images)

            tracker.writer.add_images(prompt+'test object image', formatted_images,epoch, dataformats="NHWC")

def log_validation4(
    pipeline,
    args,
    accelerator,
    generator,
    epoch,
):
    logger.info(
        f"Running validation... a training object image"
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    data = []
    with open(args.object_dataset_name, 'rt') as f:
        for line in tqdm(f, desc='Loading data'):
            data.append(json.loads(line))
    item = data[0]

    target_filename = item['target']
    source_filename = item['fg']
    light_filename = item['bg']
    mask_filename = item['mask']

    mask = imageio.v3.imread(mask_filename) / 255.
    non_mask = 1 - mask
    source = imageio.v3.imread(source_filename)[..., :3] / 255.
    light = imageio.v3.imread(light_filename)[..., :3] / 255.

    source = source[:, :, :3] * mask[:, :, np.newaxis]
    light = light[:, :, :3] * non_mask[:, :, np.newaxis]

    target = imageio.v3.imread(target_filename)[..., :3] / 255.

    conds = [source, light]
    conds = np.concatenate(conds, axis=2).astype(np.float32)[None]
    conds = torch.from_numpy(conds).to(torch.float16).permute(0, 3, 1, 2).to(accelerator.device)

    original_image = conds

    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)

    prompt = item['lighting_description']

    condition_kwargs = _validation_condition_kwargs(item)
    with autocast_ctx:
        edited_images=pipeline(
            prompt,
            image=original_image,
            num_inference_steps=50,
            image_guidance_scale=1.5,
            guidance_scale=7,
            generator=generator,
            **condition_kwargs,
        ).images[0]

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            images = edited_images

            src_image = source

            target_image = target

            formatted_images = []

            images = np.array(images) / 255.

            formatted_images.append(src_image)

            formatted_images.append(light)

            formatted_images.append(images)

            formatted_images.append(target_image)

            formatted_images = np.stack(formatted_images)

            tracker.writer.add_images(prompt+'train object image', formatted_images,epoch, dataformats="NHWC")

def log_validation5(
    pipeline,
    args,
    accelerator,
    generator,
    epoch,
):
    logger.info(
        f"Running validation... a test real portrait image"
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    data = []
    with open(args.val_real_portrait_image_url, 'rt') as f:
        for line in tqdm(f, desc='Loading data'):
            data.append(json.loads(line))
    item = data[0]

    source_filename = item['fg']
    light_filename = item['bg']

    source_image = imageio.v3.imread(source_filename)[..., :3] / 255.
    light_image = imageio.v3.imread(light_filename)[..., :3] / 255.

    light_image = Image.fromarray((light_image * 255).astype(np.uint8))
    light_image = light_image.resize((512, 512))
    light_image = np.array(light_image) / 255.

    input_images = transform_image(source_image).unsqueeze(0).to('cuda', dtype=torch.float32)
    with torch.no_grad():
        preds = seg_model(input_images)[-1].sigmoid().cpu()
    mask = preds[0].squeeze().numpy()
    non_mask = 1 - mask

    source = source_image[:, :, :3] * mask[:, :, np.newaxis]
    light = light_image[:, :, :3] * non_mask[:, :, np.newaxis]

    conds = [source, light]
    conds = np.concatenate(conds, axis=2).astype(np.float32)[None]
    conds = torch.from_numpy(conds).to(torch.float16).permute(0, 3, 1, 2).to(accelerator.device)

    original_image = conds

    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)

    prompt = args.validation_prompt

    condition_kwargs = _validation_condition_kwargs(item)
    with autocast_ctx:
        edited_images=pipeline(
            prompt,
            image=original_image,
            num_inference_steps=50,
            image_guidance_scale=1.5,
            guidance_scale=7,
            generator=generator,
            **condition_kwargs,
        ).images[0]

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            images = edited_images

            src_image = source

            formatted_images = []

            images = np.array(images) / 255.

            formatted_images.append(source_image)

            formatted_images.append(src_image)

            formatted_images.append(light_image)

            formatted_images.append(images)

            formatted_images = np.stack(formatted_images)

            tracker.writer.add_images(prompt+'test real portrait image', formatted_images,epoch, dataformats="NHWC")

def log_validation(
    pipeline,
    args,
    accelerator,
    generator,
    epoch,
):
    logger.info(
        f"Running validation... a training image and a test image"
    )

    log_validation1(
        pipeline,
        args,
        accelerator,
        generator,
        epoch,
    )

    log_validation2(
        pipeline,
        args,
        accelerator,
        generator,
        epoch,
    )

    log_validation3(
        pipeline,
        args,
        accelerator,
        generator,
        epoch,
    )

    log_validation4(
        pipeline,
        args,
        accelerator,
        generator,
        epoch,
    )

    log_validation5(
        pipeline,
        args,
        accelerator,
        generator,
        epoch,
    )

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script for InstructPix2Pix.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_unet",
        type=str,
        default=None,
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--object_dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--original_image_column",
        type=str,
        default="input_image",
        help="The column of the dataset containing the original image on which edits where made.",
    )
    parser.add_argument(
        "--edited_image_column",
        type=str,
        default="edited_image",
        help="The column of the dataset containing the edited image.",
    )
    parser.add_argument(
        "--edit_prompt_column",
        type=str,
        default="edit_prompt",
        help="The column of the dataset containing the edit instruction.",
    )
    parser.add_argument(
        "--val_syn_portrait_image_url",
        type=str,
        default=None,
        help="URL to the original image that you would like to edit (used during inference for debugging purposes).",
    )
    parser.add_argument(
        "--val_object_image_url",
        type=str,
        default=None,
        help="URL to the original image that you would like to edit (used during inference for debugging purposes).",
    )
    parser.add_argument(
        "--val_real_portrait_image_url",
        type=str,
        default=None,
        help="URL to the original image that you would like to edit (used during inference for debugging purposes).",
    )
    parser.add_argument(
        "--validation_prompt", type=str, default=None, help="A prompt that is sampled during training for inference."
    )
    parser.add_argument(
        "--content_image_encoder",
        type=str,
        default=None,
        help="CLIP model used to encode the foreground when the dataset item has no content_embed path.",
    )
    parser.add_argument("--content_embed_dim", type=int, default=768, help="Dimension of the content image vector.")
    parser.add_argument("--lighting_embed_dim", type=int, default=768, help="Dimension of the lighting image vector.")
    parser.add_argument(
        "--num_image_tokens",
        type=int,
        default=4,
        help="Number of cross-attention tokens projected from each image condition vector.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=2,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="instruct-pix2pix-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--conditioning_dropout_prob",
        type=float,
        default=None,
        help="Conditioning dropout probability. Drops out the conditionings (image and edit prompt) used in training InstructPix2Pix. See section 3.2.1 in the paper: https://arxiv.org/abs/2211.09800.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args

def convert_to_np(image, resolution):
    image = image.convert("RGB").resize((resolution, resolution))
    return np.array(image).transpose(2, 0, 1)

def download_image(url):
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image

def main():
    args = parse_args()
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:

        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:

        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )

    if args.pretrained_unet==None:
        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="unet",
            revision=args.non_ema_revision,
            class_embed_type='projection',
            projection_class_embeddings_input_dim=6,
        )

        logger.info("Initializing the InstructPix2Pix UNet from the pretrained UNet.")
        in_channels = 12
        out_channels = unet.conv_in.out_channels
        unet.register_to_config(in_channels=in_channels)

        with torch.no_grad():
            new_conv_in = nn.Conv2d(
                in_channels, out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding
            )
            new_conv_in.weight.zero_()
            new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
            unet.conv_in = new_conv_in
    else:
        unet = load_harmolight_unet(args.pretrained_unet)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    if args.pretrained_unet is None:
        harmolight_info = inject_harmolight_conditioning(
            unet,
            content_embed_dim=args.content_embed_dim,
            lighting_embed_dim=args.lighting_embed_dim,
            num_image_tokens=args.num_image_tokens,
        )
    else:
        harmolight_info = {
            "num_blocks": sum(1 for module in unet.modules() if getattr(module, "_harmolight_patched", False)),
            "curriculum": "loaded",
            "num_image_tokens": unet.lighting_image_projector.num_tokens,
        }
    logger.info(
        "HarmoLight conditioning enabled: "
        f"{harmolight_info['num_blocks']} blocks, curriculum embedding={harmolight_info['curriculum']}, "
        f"image tokens={harmolight_info['num_image_tokens']}."
    )

    if args.use_ema:
        ema_unet = EMAModel(unet.parameters(), model_cls=UNet2DConditionModel, model_config=unet.config)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):

        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    if weights:
                        weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for i in range(len(models)):

                model = models.pop()

                unet_dir = os.path.join(input_dir, "unet")
                state_dict = read_diffusion_state_dict(unet_dir)
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                logger.info(
                    f"Loaded UNet checkpoint from {unet_dir}. missing={len(missing)} unexpected={len(unexpected)}"
                )
                del state_dict

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataset = ObjectRelightingDataset(
        args.object_dataset_name,
        args.pretrained_model_name_or_path,
    )

    def tokenize_captions(captions):
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    if args.use_ema:
        ema_unet.to(accelerator.device)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    content_image_encoder = None
    if args.content_image_encoder:
        from transformers import CLIPModel

        content_image_encoder = CLIPModel.from_pretrained(args.content_image_encoder)
        content_image_encoder.requires_grad_(False)
        content_image_encoder.to(accelerator.device, dtype=weight_dtype)
        content_image_encoder.eval()
        projection_dim = getattr(content_image_encoder.config, "projection_dim", None)
        if projection_dim is not None and projection_dim != args.content_embed_dim:
            raise ValueError(
                f"CLIP projection dim is {projection_dim}, but --content_embed_dim is {args.content_embed_dim}."
            )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )

    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers("instruct-pix2pix", config=vars(args))

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:

            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):

            with accelerator.accumulate(unet):

                latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                content_text_embeds = text_encoder(batch["content_input_ids"].to(accelerator.device))[0]
                lighting_text_embeds = text_encoder(batch["lighting_input_ids"].to(accelerator.device))[0]
                encoder_hidden_states = content_text_embeds

                original_image_embeds1 = vae.encode(batch["conditioning_pixel_values"][:,:3,:,:].to(weight_dtype)).latent_dist.mode()

                original_image_embeds2 = vae.encode(
                    batch["conditioning_pixel_values"][:, 3:, :, :].to(weight_dtype)).latent_dist.mode()

                concatenated_noisy_latents = torch.cat([noisy_latents, original_image_embeds1,original_image_embeds2], dim=1)

                has_content_image = bool(batch["has_content_image_embed"].flatten()[0].item())
                if has_content_image:
                    content_image_embed = batch["content_image_embed"].to(
                        device=latents.device, dtype=weight_dtype
                    )
                else:
                    if content_image_encoder is None:
                        raise RuntimeError(
                            "Dataset item has no content image vector. Provide content_embed, "
                            "or set --content_image_encoder to encode the foreground with CLIP."
                        )
                    foreground = batch["conditioning_pixel_values"][:, :3].to(
                        device=latents.device, dtype=weight_dtype
                    )
                    with torch.no_grad():
                        content_image_embed = encode_foreground_clip(content_image_encoder, foreground)

                lighting_image_embed = batch["lighting_image_embed"].to(
                    device=latents.device, dtype=weight_dtype
                )

                curriculum = batch["curriculum"].to(device=latents.device, dtype=weight_dtype)
                class_embedding = curriculum_embedding_from_one_hot(curriculum)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                model_pred = unet(
                    concatenated_noisy_latents,
                    timesteps,
                    encoder_hidden_states,
                    class_labels=class_embedding,
                    cross_attention_kwargs={
                        "content_text_embeds": content_text_embeds,
                        "content_image_embeds": content_image_embed,
                        "lighting_text_embeds": lighting_text_embeds,
                        "lighting_image_embeds": lighting_image_embed,
                    },
                    return_dict=False,
                )[0]
                noise_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                loss = noise_loss

                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:

                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            if (
                (args.val_syn_portrait_image_url is not None)
                and (args.validation_prompt is not None)
                and (epoch % args.validation_epochs == 0)
            ):
                if args.use_ema:

                    ema_unet.store(unet.parameters())
                    ema_unet.copy_to(unet.parameters())

                pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    unet=unwrap_model(unet),
                    text_encoder=unwrap_model(text_encoder),
                    vae=unwrap_model(vae),
                    revision=args.revision,
                    variant=args.variant,
                    torch_dtype=weight_dtype,
                    safety_checker=None,
                    requires_safety_checker=False,
                )

                log_validation(
                    pipeline,
                    args,
                    accelerator,
                    generator,
                    epoch,
                )

                if args.use_ema:

                    ema_unet.restore(unet.parameters())

                del pipeline
                torch.cuda.empty_cache()

    accelerator.end_training()

if __name__ == "__main__":
    main()
