import json
import random
import cv2
import imageio
import numpy as np
from tqdm import tqdm
import imageio
from PIL import Image, ImageEnhance
import os

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from multimodal_conditions import MissingConditionError, build_multimodal_condition_fields

imageio.plugins.freeimage.download()

def random_enhance(image):

    color_factor = random.uniform(0.9, 1.1)
    contrast_factor = random.uniform(0.9, 1.1)
    brightness_factor = random.uniform(0.9, 1.1)

    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(color_factor)

    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast_factor)

    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness_factor)

    return image

class DelightingDataset(Dataset):
    def __init__(
            self,
            data_jsonl=None,
            pretrained_model=os.environ.get("SD15_MODEL_PATH", "pretrained/realistic-vision-v51"),
            channel_aug_ratio=0.0,
            empty_prompt_ratio=0.0,
            self_ref_ratio=0.0,
            eval_mode=False,
            direct_data=None,
    ):
        self.data = []
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model,
            subfolder="tokenizer",
            use_fast=False,
        )
        self.channel_aug_ratio = channel_aug_ratio
        self.empty_prompt_ratio = empty_prompt_ratio
        self.eval_mode = eval_mode

        if direct_data is not None:
            self.data = direct_data

        elif data_jsonl is not None:
            with open(data_jsonl, 'rt') as f:
                for line in tqdm(f, desc='Loading data'):
                    self.data.append(json.loads(line))
                    self.data = self.data
        else:
            raise ValueError("No data source specified.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            item = self.data[idx]

            parts = [part for part in item['target'].replace('\\', '/').split('/') if part]
            model_id = '_'.join(parts[-4:-1]) if len(parts) >= 4 else '_'.join(parts)

            target_filename = item['target']
            source_filename = item['fg']
            light_filename = item['bg']
            mask_filename = item['mask']

            mask = imageio.v3.imread(mask_filename) / 255.
            non_mask = 1 - mask
            source = imageio.v3.imread(target_filename)[..., :3] / 255.

            light = imageio.v3.imread(light_filename)[..., :3] / 255.

            source = source[:, :, :3] * mask[:, :, np.newaxis]

            light = light[:, :, :3] * non_mask[:, :, np.newaxis]

            target = imageio.v3.imread(target_filename)[..., :3] / 127.5 - 1.

            source = source * 2.0 - 1.0
            light = light * 2.0 - 1.0

            conds = np.concatenate([source, light], axis=2)
            conds = conds.astype(np.float32)
            target = target.astype(np.float32)

            fields = build_multimodal_condition_fields(
                item,
                self.tokenizer,
                self.empty_prompt_ratio,
                default_curriculum="delighted",
            )
        except MissingConditionError:
            raise
        except Exception as e:
            if self.eval_mode:
                raise e
            print(idx, repr(e), e)
            return self.__getitem__(np.random.randint(0, len(self.data)))

        return dict(
            pixel_values=target.transpose(2, 0, 1),
            conditioning_pixel_values=conds.transpose(2, 0, 1),
            target_file=target_filename,
            ref_file=source_filename,
            model_id=model_id,
            **fields,
        )
