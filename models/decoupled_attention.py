import copy
import glob
import os

import torch
from torch import nn
from diffusers import UNet2DConditionModel
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.embeddings import TimestepEmbedding

def read_diffusion_state_dict(model_dir: str) -> dict:
    safetensor_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if safetensor_files:
        import safetensors.torch as sf

        state_dict = {}
        for path in safetensor_files:
            state_dict.update(sf.load_file(path))
        return state_dict

    for name in ("diffusion_pytorch_model.bin", "pytorch_model.bin"):
        bin_path = os.path.join(model_dir, name)
        if os.path.exists(bin_path):
            return torch.load(bin_path, map_location="cpu")
    raise FileNotFoundError(f"No diffusion weights found in {model_dir}")

def _apply_norm(norm, hidden_states, timestep):
    try:
        return norm(hidden_states, timestep)
    except TypeError:
        return norm(hidden_states)

def _chunked_feed_forward(ff, hidden_states, chunk_dim, chunk_size):
    if hidden_states.shape[chunk_dim] % chunk_size != 0:
        raise ValueError(
            f"Hidden size {hidden_states.shape[chunk_dim]} is not divisible by chunk size {chunk_size}."
        )
    pieces = hidden_states.chunk(hidden_states.shape[chunk_dim] // chunk_size, dim=chunk_dim)
    return torch.cat([ff(piece) for piece in pieces], dim=chunk_dim)

def _align_batch(reference, cond):
    if cond is None:
        return None
    if cond.ndim == 2:
        cond = cond.unsqueeze(1)
    if cond.ndim != 3:
        raise ValueError(f"Cross-attention condition must be [batch, tokens, dim], got {tuple(cond.shape)}.")

    batch = reference.shape[0]
    if cond.shape[0] == 1 and batch != 1:
        cond = cond.expand(batch, -1, -1)
    elif cond.shape[0] != batch:
        if cond.shape[0] < batch and batch % cond.shape[0] == 0:
            cond = cond.repeat_interleave(batch // cond.shape[0], dim=0)
        else:
            cond = cond[:batch]
    return cond.to(device=reference.device, dtype=reference.dtype).contiguous()

def _zero_out_projection(attn):
    projection = attn.to_out[0] if isinstance(attn.to_out, nn.ModuleList) else attn.to_out
    nn.init.zeros_(projection.weight)
    if getattr(projection, "bias", None) is not None:
        nn.init.zeros_(projection.bias)

def _patch_block(block: BasicTransformerBlock) -> None:
    if getattr(block, "_harmolight_patched", False):
        return

    def forward(
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        timestep=None,
        cross_attention_kwargs=None,
        class_labels=None,
        added_cond_kwargs=None,
        **kwargs,
    ):
        del class_labels, added_cond_kwargs, kwargs
        cross_attention_kwargs = dict(cross_attention_kwargs) if cross_attention_kwargs is not None else {}
        content_text = cross_attention_kwargs.pop("content_text_embeds", None)
        content_image = cross_attention_kwargs.pop("content_image_embeds", None)
        lighting_text = cross_attention_kwargs.pop("lighting_text_embeds", None)
        lighting_image = cross_attention_kwargs.pop("lighting_image_embeds", None)

        if content_text is None:
            content_text = encoder_hidden_states
        if lighting_text is None:
            lighting_text = encoder_hidden_states

        content_text = _align_batch(hidden_states, content_text)
        content_image = _align_batch(hidden_states, content_image)
        lighting_text = _align_batch(hidden_states, lighting_text)
        lighting_image = _align_batch(hidden_states, lighting_image)

        norm_hidden_states = _apply_norm(block.norm1, hidden_states, timestep)
        attn_output = block.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if getattr(block, "only_cross_attention", False) else None,
            attention_mask=attention_mask,
        )
        hidden_states = hidden_states + attn_output

        if block.attn2 is not None:
            norm_hidden_states = _apply_norm(block.norm2, hidden_states, timestep)
            if content_image is not None and hasattr(block, "attn2_image"):
                hidden_states = hidden_states + block.attn2_image(
                    norm_hidden_states,
                    encoder_hidden_states=content_image,
                )
            if content_text is not None:
                text_output = block.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=content_text,
                    attention_mask=encoder_attention_mask,
                )
                beta = block.content_text_scale.to(dtype=text_output.dtype)
                hidden_states = hidden_states + beta * text_output

        if getattr(block, "attn3", None) is not None:
            norm_hidden_states = _apply_norm(block.norm_lighting, hidden_states, timestep)
            if lighting_image is not None and hasattr(block, "attn3_image"):
                hidden_states = hidden_states + block.attn3_image(
                    norm_hidden_states,
                    encoder_hidden_states=lighting_image,
                )
            if lighting_text is not None:
                text_output = block.attn3(
                    norm_hidden_states,
                    encoder_hidden_states=lighting_text,
                    attention_mask=encoder_attention_mask,
                )
                alpha = block.lighting_text_scale.to(dtype=text_output.dtype)
                hidden_states = hidden_states + alpha * text_output

        if getattr(block, "norm3", None) is None:
            norm_hidden_states = hidden_states
        else:
            norm_hidden_states = _apply_norm(block.norm3, hidden_states, timestep)
        chunk_size = getattr(block, "_chunk_size", None)
        if chunk_size is not None:
            ff_output = _chunked_feed_forward(block.ff, norm_hidden_states, getattr(block, "_chunk_dim", 0), chunk_size)
        else:
            ff_output = block.ff(norm_hidden_states)
        hidden_states = hidden_states + ff_output
        return hidden_states

    block.forward = forward
    block._harmolight_patched = True

def inject_decoupled_cross_attention(unet: UNet2DConditionModel) -> int:
    """Replace each text cross-attention with DCCA and DLCA.

    DCCA(X, c_c, t_c) = Attention(X, c_c) + beta * Attention(X, t_c)
    DLCA(X, c_l, t_l) = Attention(X, c_l) + alpha * Attention(X, t_l)
    """
    num_blocks = 0
    for module in unet.modules():
        if not isinstance(module, BasicTransformerBlock) or module.attn2 is None:
            continue
        if not hasattr(module, "attn2_image"):
            module.attn2_image = copy.deepcopy(module.attn2)
            module.attn3 = copy.deepcopy(module.attn2)
            module.attn3_image = copy.deepcopy(module.attn2)
            module.norm_lighting = copy.deepcopy(module.norm2)
            module.content_text_scale = nn.Parameter(torch.ones(()))
            module.lighting_text_scale = nn.Parameter(torch.zeros(()))
            _zero_out_projection(module.attn2_image)
            _zero_out_projection(module.attn3_image)
        _patch_block(module)
        num_blocks += 1
    return num_blocks

class ConditionTokenProjector(nn.Module):
    """Map one condition vector to a short token sequence for cross-attention."""

    def __init__(self, in_dim: int, cross_dim: int, num_tokens: int):
        super().__init__()
        self.num_tokens = num_tokens
        self.cross_dim = cross_dim
        self.proj = nn.Linear(in_dim, num_tokens * cross_dim)
        self.norm = nn.LayerNorm(cross_dim)

    def forward(self, embeds: torch.Tensor) -> torch.Tensor:
        if embeds.ndim == 3:
            embeds = embeds.mean(dim=1)
        if embeds.ndim != 2:
            raise ValueError(f"Condition vector must be [batch, dim], got {tuple(embeds.shape)}.")
        tokens = self.proj(embeds)
        tokens = tokens.reshape(embeds.shape[0], self.num_tokens, self.cross_dim)
        return self.norm(tokens)

def ensure_curriculum_embedding(unet: UNet2DConditionModel, input_dim: int = 6) -> str:
    """Add CDE = MLP(cat[sin(i), cos(i)]) and let UNet add it to the timestep embedding."""
    if getattr(unet, "class_embedding", None) is not None:
        expected = getattr(unet.config, "projection_class_embeddings_input_dim", None)
        if expected not in (None, input_dim):
            raise ValueError(
                f"UNet class embedding expects input dim {expected}, but curriculum embedding uses {input_dim}."
            )
        return "existing"

    time_embed_dim = unet.time_embedding.linear_2.out_features
    unet.class_embedding = TimestepEmbedding(input_dim, time_embed_dim)
    unet.register_to_config(
        class_embed_type="projection",
        projection_class_embeddings_input_dim=input_dim,
    )
    return "created"

def _maybe_project_condition(projector, embeds):
    if embeds is None or projector is None or embeds.ndim == 3:
        return embeds
    if embeds.ndim == 1:
        embeds = embeds.unsqueeze(0)
    return projector(embeds.to(dtype=projector.proj.weight.dtype))

def install_condition_projector_forward(unet: UNet2DConditionModel) -> None:
    """Project image vectors inside UNet.forward so the new layers stay in the training graph."""
    if getattr(unet, "_harmolight_forward_wrapped", False):
        return

    original_forward = unet.forward

    def forward(*args, **kwargs):
        cross_attention_kwargs = dict(kwargs.get("cross_attention_kwargs") or {})
        if cross_attention_kwargs:
            if "content_image_embeds" in cross_attention_kwargs:
                cross_attention_kwargs["content_image_embeds"] = _maybe_project_condition(
                    getattr(unet, "content_image_projector", None),
                    cross_attention_kwargs["content_image_embeds"],
                )
            if "lighting_image_embeds" in cross_attention_kwargs:
                cross_attention_kwargs["lighting_image_embeds"] = _maybe_project_condition(
                    getattr(unet, "lighting_image_projector", None),
                    cross_attention_kwargs["lighting_image_embeds"],
                )
            kwargs["cross_attention_kwargs"] = cross_attention_kwargs
        return original_forward(*args, **kwargs)

    unet.forward = forward
    unet._harmolight_forward_wrapped = True

def inject_harmolight_conditioning(
    unet: UNet2DConditionModel,
    curriculum_input_dim: int = 6,
    content_embed_dim: int = 768,
    lighting_embed_dim: int = 768,
    num_image_tokens: int = 4,
) -> dict:
    num_blocks = inject_decoupled_cross_attention(unet)
    curriculum_status = ensure_curriculum_embedding(unet, input_dim=curriculum_input_dim)
    cross_dim = unet.config.cross_attention_dim
    if not hasattr(unet, "content_image_projector"):
        unet.add_module(
            "content_image_projector",
            ConditionTokenProjector(content_embed_dim, cross_dim, num_image_tokens),
        )
    if not hasattr(unet, "lighting_image_projector"):
        unet.add_module(
            "lighting_image_projector",
            ConditionTokenProjector(lighting_embed_dim, cross_dim, num_image_tokens),
        )
    install_condition_projector_forward(unet)
    return {
        "num_blocks": num_blocks,
        "curriculum": curriculum_status,
        "cross_attention_dim": cross_dim,
        "num_image_tokens": num_image_tokens,
    }

def conditioning_settings_from_state(state_dict: dict, cross_dim: int) -> dict:
    """Read projector and curriculum sizes stored in a HarmoLight checkpoint."""
    settings = {
        "content_embed_dim": 768,
        "lighting_embed_dim": 768,
        "num_image_tokens": 4,
        "curriculum_input_dim": 6,
    }
    content_weight = state_dict.get("content_image_projector.proj.weight")
    lighting_weight = state_dict.get("lighting_image_projector.proj.weight")
    if content_weight is not None:
        settings["content_embed_dim"] = int(content_weight.shape[1])
        if cross_dim and content_weight.shape[0] % cross_dim == 0:
            settings["num_image_tokens"] = int(content_weight.shape[0] // cross_dim)
    if lighting_weight is not None:
        settings["lighting_embed_dim"] = int(lighting_weight.shape[1])
    curriculum_weight = state_dict.get("class_embedding.linear_1.weight")
    if curriculum_weight is not None:
        settings["curriculum_input_dim"] = int(curriculum_weight.shape[1])
    return settings

def load_harmolight_unet(unet_path: str, **from_pretrained_kwargs) -> UNet2DConditionModel:
    """Load a UNet and restore decoupled-attention weights when the checkpoint has them."""
    state_dict = read_diffusion_state_dict(unet_path)
    try:
        unet = UNet2DConditionModel.from_pretrained(unet_path, **from_pretrained_kwargs)
    except Exception:
        config = UNet2DConditionModel.load_config(unet_path)
        init_kwargs = {
            key: value
            for key, value in from_pretrained_kwargs.items()
            if key in UNet2DConditionModel.__init__.__code__.co_varnames
        }
        unet = UNet2DConditionModel.from_config(config, **init_kwargs)
        unet.load_state_dict(state_dict, strict=False)
    settings = conditioning_settings_from_state(state_dict, unet.config.cross_attention_dim)
    inject_harmolight_conditioning(unet, **settings)

    unet.load_state_dict(state_dict, strict=False)
    return unet
