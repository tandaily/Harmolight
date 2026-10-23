import json
import random
import cv2
import imageio
import numpy as np
from tqdm import tqdm
from PIL import Image
import imageio
import os

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from multimodal_conditions import MissingConditionError, build_multimodal_condition_fields

imageio.plugins.freeimage.download()

class VIDITRelightingDataset(Dataset):
    def __init__(
            self,
            data_jsonl=None,
            pretrained_model=os.environ.get("SD21_MODEL_PATH", "pretrained/stable-diffusion-2-1"),
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
            model_id = os.path.splitext(os.path.basename(item['gt']))[0]

            target_filename = item['gt']
            source_filename = item['src']

            light_filename = item['light']

            source = cv2.imread(source_filename)
            if source is None:
                print("Failed to load source image")
            source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
            source = cv2.resize(source, (512,512))
            source = source / 127.5 - 1.

            target = cv2.imread(target_filename)
            if target is None:
                print("Failed to load target image")
            target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
            target = cv2.resize(target, (512, 512))
            target = target / 127.5 - 1.

            light = cv2.imread(light_filename)

            if light is None:
                print("Failed to load target image")
            light = cv2.cvtColor(light, cv2.COLOR_BGR2RGB)
            source_size = (source.shape[1], source.shape[0])

            light = cv2.resize(light, source_size)
            light = light / 127.5 - 1.

            conds = np.concatenate([source, light], axis=2)
            conds = conds.astype(np.float32)
            target = target.astype(np.float32)

            fields = build_multimodal_condition_fields(
                item,
                self.tokenizer,
                self.empty_prompt_ratio,
                default_curriculum="relighted",
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

class SyntheticPortritRelightingDataset(Dataset):
    def __init__(
            self,
            data_jsonl=None,
            pretrained_model=os.environ.get("SD21_MODEL_PATH", "pretrained/stable-diffusion-2-1"),
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

            model_id = os.path.splitext(os.path.basename(item['fg']))[0]

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
                default_curriculum="relighted",
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

class ObjectRelightingDataset(Dataset):
    def __init__(
            self,
            data_jsonl=None,
            pretrained_model=os.environ.get("SD21_MODEL_PATH", "pretrained/stable-diffusion-2-1"),
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
            source = imageio.v3.imread(source_filename)[..., :3] / 255.
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
                default_curriculum="relighted",
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

class ObjectSyntheticRelightingDataset(Dataset):
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

            new_name = os.path.splitext(os.path.basename(item['target']))[0]
            model_id = new_name

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
                default_curriculum="relighted",
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
