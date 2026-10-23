import os

import numpy as np
import torch

class MissingConditionError(RuntimeError):
    """Dataset item is missing a required multimodal condition."""

CURRICULUM_NAME_TO_INDEX = {
    "filtered": 0,
    "retouch": 0,
    "delighted": 1,
    "delight": 1,
    "relighted": 2,
    "relight": 2,
}

CONTENT_TEXT_KEYS = (
    "content_description",
    "foreground_description",
    "content_prompt",
    "fg_description",
    "overall_description",
)
LIGHTING_TEXT_KEYS = (
    "lighting_description",
    "lighting_prompt",
    "light_prompt",
    "light_description",
)
LIGHTING_EMBED_KEYS = (
    "light_embed",
    "lighting_embed",
    "lighting_image_embed",
    "light_embedding",
    "lighting_embedding_path",
    "le_embed",
)
CONTENT_EMBED_KEYS = (
    "content_embed",
    "content_image_embed",
    "foreground_embed",
    "ce_embed",
)

def _first_present(item, keys):
    for key in keys:
        value = item.get(key)
        if value is not None and value != "":
            return value
    return None

def curriculum_one_hot(item, default_curriculum="relighted"):
    if "curriculum" in item:
        value = item["curriculum"]
    elif "difficulty" in item:
        value = item["difficulty"]
    elif "domain" in item:
        value = item["domain"]
    else:
        value = default_curriculum

    if isinstance(value, str):
        name = value.lower()
        if name not in CURRICULUM_NAME_TO_INDEX:
            raise MissingConditionError(
                f"Unknown curriculum '{value}'. Expected one of {sorted(CURRICULUM_NAME_TO_INDEX)}."
            )
        vector = np.zeros(3, dtype=np.float32)
        vector[CURRICULUM_NAME_TO_INDEX[name]] = 1.0
        return vector

    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size == 3:
        return array
    if array.size == 1:
        vector = np.zeros(3, dtype=np.float32)
        vector[int(array[0])] = 1.0
        return vector
    raise MissingConditionError(f"Curriculum value must be a name, an index, or a 3-d one-hot, got shape {array.shape}.")

def curriculum_name(item, default_curriculum="relighted"):
    vector = curriculum_one_hot(item, default_curriculum=default_curriculum)
    index = int(np.argmax(vector))
    return ("filtered", "delighted", "relighted")[index]

def curriculum_embedding_from_one_hot(one_hot: torch.Tensor) -> torch.Tensor:
    """Paper CDE input: cat[sin(i), cos(i)] before the MLP that is added like the timestep."""
    return torch.cat([torch.sin(one_hot), torch.cos(one_hot)], dim=-1)

def _tensor_from_object(obj):
    if torch.is_tensor(obj):
        return obj.detach().float().cpu()
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(np.asarray(obj, dtype=np.float32))
    if isinstance(obj, dict):
        for key in ("embed", "embedding", "light_embed", "content_embed", "feat", "feature", "vector"):
            if key in obj:
                return _tensor_from_object(obj[key])
        for value in obj.values():
            if torch.is_tensor(value) or isinstance(value, np.ndarray):
                return _tensor_from_object(value)
    raise MissingConditionError(f"Cannot read a condition vector from object of type {type(obj)}.")

def load_condition_vector(path: str) -> torch.Tensor:
    """Load a lighting or content condition vector saved beside the dataset item.

    Sequences are averaged into one vector. The diffusion model projects that vector
    into cross-attention tokens.
    """
    if not path or not os.path.isfile(path):
        raise MissingConditionError(f"Condition vector file does not exist: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        array = np.load(path)
        tensor = torch.from_numpy(np.asarray(array, dtype=np.float32))
    elif ext == ".npz":
        data = np.load(path)
        tensor = torch.from_numpy(np.asarray(data[data.files[0]], dtype=np.float32))
    elif ext == ".safetensors":
        import safetensors.torch as sf

        tensor = _tensor_from_object(sf.load_file(path))
    elif ext in {".pt", ".pth"}:
        try:
            obj = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            obj = torch.load(path, map_location="cpu")
        tensor = _tensor_from_object(obj)
    else:
        raise MissingConditionError(f"Unsupported condition vector file: {path}")

    tensor = tensor.float().cpu().reshape(-1, tensor.shape[-1]) if tensor.ndim >= 2 else tensor.float().cpu().reshape(1, -1)
    vector = tensor.mean(dim=0).contiguous()
    if vector.ndim != 1:
        vector = vector.reshape(-1)
    return vector

def _tokenize(tokenizer, text):
    inputs = tokenizer(
        text,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs.input_ids[0]

def build_multimodal_condition_fields(
    item,
    tokenizer,
    empty_prompt_ratio=0.0,
    default_curriculum="relighted",
    require_lighting_embed=True,
):
    import random

    content_text = _first_present(item, CONTENT_TEXT_KEYS) or ""
    lighting_text = _first_present(item, LIGHTING_TEXT_KEYS) or ""
    if random.random() < empty_prompt_ratio:
        content_text = ""
        lighting_text = ""

    lighting_path = _first_present(item, LIGHTING_EMBED_KEYS)
    if require_lighting_embed and not lighting_path:
        raise MissingConditionError(
            "Dataset item is missing a lighting image condition path. "
            f"Add one of {LIGHTING_EMBED_KEYS} pointing to a .npy/.pt/.safetensors vector."
        )
    if lighting_path:
        lighting_image_embed = load_condition_vector(lighting_path)
    else:
        lighting_image_embed = torch.zeros(1, dtype=torch.float32)

    content_path = _first_present(item, CONTENT_EMBED_KEYS)
    if content_path:
        content_image_embed = load_condition_vector(content_path)
        has_content_image_embed = torch.tensor(1, dtype=torch.long)
    else:
        content_image_embed = torch.zeros(1, dtype=torch.float32)
        has_content_image_embed = torch.tensor(0, dtype=torch.long)

    content_input_ids = _tokenize(tokenizer, content_text)
    lighting_input_ids = _tokenize(tokenizer, lighting_text)
    return {
        "input_ids": content_input_ids,
        "content_input_ids": content_input_ids,
        "lighting_input_ids": lighting_input_ids,
        "text": content_text,
        "content_text": content_text,
        "lighting_text": lighting_text,
        "lighting_image_embed": lighting_image_embed,
        "content_image_embed": content_image_embed,
        "has_content_image_embed": has_content_image_embed,
        "curriculum": torch.from_numpy(curriculum_one_hot(item, default_curriculum=default_curriculum)),
    }

def inference_condition_kwargs(item, default_curriculum="relighted"):
    kwargs = {"domain": curriculum_name(item, default_curriculum=default_curriculum)}
    content_text = _first_present(item, CONTENT_TEXT_KEYS)
    lighting_text = _first_present(item, LIGHTING_TEXT_KEYS)
    if content_text:
        kwargs["content_prompt"] = content_text
    if lighting_text:
        kwargs["lighting_prompt"] = lighting_text
    lighting_path = _first_present(item, LIGHTING_EMBED_KEYS)
    if lighting_path:
        kwargs["lighting_image_embed_path"] = lighting_path
    content_path = _first_present(item, CONTENT_EMBED_KEYS)
    if content_path:
        kwargs["content_image_embed_path"] = content_path
    return kwargs
