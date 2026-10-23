import numpy as np
import imageio
import torch
from diffusers.utils import get_class_from_dynamic_module
from diffusers import StableDiffusionPipeline, DDIMInverseScheduler, AutoencoderKL, DDIMScheduler,DDPMScheduler, DPMSolverMultistepScheduler
import json
from tqdm import tqdm
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler
import os
import sys
from torchvision import transforms
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models"))
import safetensors.torch as sf
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch.nn.functional as F
import gradio as gr
import cv2

from pipeline_harmolight import StableDiffusionInstructPix2PixPipeline
from decoupled_attention import load_harmolight_unet

device = 'cuda'

from transformers import AutoModelForImageSegmentation

seg_model = AutoModelForImageSegmentation.from_pretrained(os.environ.get("SEG_MODEL_PATH", "pretrained/RMBG-2.0"), trust_remote_code=True)
seg_model.to('cuda')
seg_model.eval()

sd15_name = os.environ.get("SD15_MODEL_PATH", "pretrained/realistic-vision-v51")
tokenizer = CLIPTokenizer.from_pretrained(sd15_name, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(sd15_name, subfolder="text_encoder")
vae = AutoencoderKL.from_pretrained(sd15_name, subfolder="vae")

unet_path = os.environ.get("UNET_PATH", "checkpoints/unet")

unet = load_harmolight_unet(unet_path)

device = torch.device('cuda')
text_encoder = text_encoder.to(device=device)
vae = vae.to(device=device)
unet = unet.to(device=device)

ddim_scheduler = DDIMScheduler.from_pretrained(sd15_name, subfolder="scheduler")

pipe = StableDiffusionInstructPix2PixPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler= ddim_scheduler,
    safety_checker=None,
    requires_safety_checker=False,
    feature_extractor=None,
    image_encoder=None
)

pipe = pipe.to(device)

transform_image = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def resize_to_multiple_of_64(image):
    _, _, h, w = image.shape
    new_h = (h // 64) * 64 if h % 64 == 0 else ((h // 64) + 1) * 64
    new_w = (w // 64) * 64 if w % 64 == 0 else ((w // 64) + 1) * 64
    return F.interpolate(image, size=(new_h, new_w), mode='bilinear', align_corners=False)

source_image = imageio.v3.imread(source_filename)[..., :3]
light_image = imageio.v3.imread(light_filename)[..., :3]

light_image = Image.fromarray((light_image).astype(np.uint8))
light_image = np.array(light_image) / 255.

source_image = Image.fromarray((source_image).astype(np.uint8))
source = source_image
source_image = np.array(source_image) / 255.

input_images = transform_image(source).unsqueeze(0).to('cuda', dtype=torch.float32)
input_images_resized = resize_to_multiple_of_64(input_images)
with torch.no_grad():
    preds = seg_model(input_images_resized)[-1].sigmoid().cpu()
mask = preds[0].squeeze().numpy()
original_size = input_images.shape[2:]
mask_resized = torch.tensor(mask).unsqueeze(0).unsqueeze(0)
mask = F.interpolate(mask_resized, size=original_size, mode='bilinear',
                             align_corners=False).squeeze().numpy()
non_mask = 1 - mask

conds = [source, light]

conds = np.concatenate(conds, axis=2).astype(np.float32)[None]
conds = torch.from_numpy(conds).to(torch.float16).permute(0, 3, 1, 2).to(device)

rng = torch.Generator(device=device).manual_seed(seed)

image = pipe(prompt, image=conds,num_inference_steps=steps,generator=rng,num_images_per_prompt=num_samples).images[0]

image = np.array(image) / 255.
image = image[:, :, :3] * mask[:, :, np.newaxis]
image = image + light

result = Image.fromarray((image).astype(np.uint8))

result.save(os.path.join(my_output_dir, new_name))
