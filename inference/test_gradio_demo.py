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

def resize_and_center_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    original_width, original_height = pil_image.size
    scale_factor = max(target_width / original_width, target_height / original_height)
    resized_width = int(round(original_width * scale_factor))
    resized_height = int(round(original_height * scale_factor))
    resized_image = pil_image.resize((resized_width, resized_height), Image.LANCZOS)
    left = (resized_width - target_width) / 2
    top = (resized_height - target_height) / 2
    right = (resized_width + target_width) / 2
    bottom = (resized_height + target_height) / 2
    cropped_image = resized_image.crop((left, top, right, bottom))
    return np.array(cropped_image)

def pad_with_black(source, new_h, new_w):

    h, w, c = source.shape

    padded = np.zeros((new_h, new_w, c), dtype=source.dtype)

    top = (new_h - h) // 2
    left = (new_w - w) // 2

    padded[top:top + h, left:left + w] = source

    return padded

def mask_pad_with_white(source, new_h, new_w):

    h, w = source.shape

    padded = np.ones((new_h, new_w), dtype=source.dtype)

    top = (new_h - h) // 2
    left = (new_w - w) // 2

    padded[top:top + h, left:left + w] = source

    return padded

def mask_pad_with_black(source, new_h, new_w):

    h, w = source.shape

    padded = np.zeros((new_h, new_w), dtype=source.dtype)

    top = (new_h - h) // 2
    left = (new_w - w) // 2

    padded[top:top + h, left:left + w] = source

    return padded

def resize_to_multiple_of_64(image):
    _, _, h, w = image.shape
    new_h = (h // 64) * 64 if h % 64 == 0 else ((h // 64) + 1) * 64
    new_w = (w // 64) * 64 if w % 64 == 0 else ((w // 64) + 1) * 64
    return F.interpolate(image, size=(new_h, new_w), mode='bilinear', align_corners=False)

@torch.inference_mode()
def process_relight(input_fg, input_bg, prompt, image_width, image_height, fg_width, fg_height,num_samples=1, seed=12345, steps=50):

    source_image = resize_and_center_crop(input_fg, fg_width, fg_height)
    light_image = resize_and_center_crop(input_bg, image_width, image_height)

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

    if image_width!=fg_width:

        source = source_image[:, :, :3] * mask[:, :, np.newaxis]

        source = pad_with_black(source, image_width, image_height)

        mask = mask_pad_with_black(mask, image_width, image_height)
        non_mask = mask_pad_with_white(non_mask, image_width, image_height)
        light = light_image[:, :, :3] * non_mask[:, :, np.newaxis]
    else:

        source = source_image[:, :, :3] * mask[:, :, np.newaxis]
        light = light_image[:, :, :3] * non_mask[:, :, np.newaxis]

    conds = [source, light]

    conds = np.concatenate(conds, axis=2).astype(np.float32)[None]
    conds = torch.from_numpy(conds).to(torch.float16).permute(0, 3, 1, 2).to(device)

    rng = torch.Generator(device=device).manual_seed(seed)

    images = pipe(prompt, image=conds,num_inference_steps=steps,generator=rng,num_images_per_prompt=num_samples).images

    results = []

    for image in images:
        image = np.array(image) / 255.
        image = image[:, :, :3] * mask[:, :, np.newaxis]
        image = image + light
        results.append(image)

    results = [(x*255).astype(np.uint8) for x in results]

    output = [source,light]

    output = [(x*255).astype(np.uint8) for x in output]

    return output[0],output[1],results

block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## Diff-Light (Relighting with Background Condition)")
    with gr.Row():
        with gr.Column():
            with gr.Row():
                input_fg = gr.Image(sources='upload', type="numpy", label="Foreground", height=512)
                input_bg = gr.Image(sources='upload', type="numpy", label="Background", height=512)
            with gr.Row():
                output_fg = gr.Image(type="numpy", label="Preprocessed Foreground", height=512)
                output_bg = gr.Image(type="numpy", label="Preprocessed Background", height=512)
            with gr.Row():
                result_gallery = gr.Gallery(object_fit='contain', label="Result")
            prompt = gr.Textbox(label="Prompt")
            relight_button = gr.Button(value="Relight")

            with gr.Group():
                with gr.Row():
                    num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                    seed = gr.Number(label="Seed", value=12345, precision=0)

                with gr.Row():

                    image_width = gr.Slider(label="Bg Width", minimum=256, maximum=4096, value=512, step=64)
                    image_height = gr.Slider(label="Bg Height", minimum=256, maximum=4096, value=512, step=64)

                    fg_width = gr.Slider(label="Fg Width", minimum=256, maximum=4096, value=512, step=64)
                    fg_height = gr.Slider(label="Fg Height", minimum=256, maximum=4096, value=512, step=64)

            with gr.Accordion("Advanced options", open=False):
                steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)

    ips = [input_fg, input_bg, prompt, image_width, image_height, fg_width, fg_height, num_samples, seed, steps]
    relight_button.click(fn=process_relight, inputs=ips, outputs=[output_fg, output_bg, result_gallery])

block.launch(server_name='127.0.0.1')
