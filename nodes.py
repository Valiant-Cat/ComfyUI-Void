import gc
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple

import imageio
import numpy as np
import torch
import torch.nn.functional as F

import folder_paths

try:
    import comfy.model_management as mm
except Exception:
    mm = None

try:
    from comfy.utils import ProgressBar
except Exception:
    ProgressBar = None


_HERE = Path(__file__).resolve().parent
_VOID_RUNTIME_ROOT = _HERE / "void_runtime"
_VOID_MODELS_DIR = Path(folder_paths.models_dir) / "void"
_VOID_CACHE_DIR = Path(folder_paths.get_temp_directory()) / "void"
_VOID_MODEL_TYPE = "void"
_MODEL_LOCK = threading.Lock()
_MODEL_CACHE: Dict[Tuple[str, str, str, str], dict] = {}

BASE_MODEL_CHOICES = ["CogVideoX-Fun-V1.5-5b-InP"]

if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

try:
    from safetensors.torch import load_file as load_safetensors
except Exception:
    load_safetensors = None

try:
    from scipy.ndimage import binary_fill_holes
except Exception:
    binary_fill_holes = None


SCHEDULER_NAMES = ["DDIM_Origin", "DDIM_Cog", "Euler", "Euler A", "DPM++", "PNDM"]

PRECISIONS = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def _register_void_model_dir() -> None:
    os.makedirs(_VOID_MODELS_DIR, exist_ok=True)
    if _VOID_MODEL_TYPE not in folder_paths.folder_names_and_paths:
        folder_paths.folder_names_and_paths[_VOID_MODEL_TYPE] = (
            [str(_VOID_MODELS_DIR)],
            set(folder_paths.supported_pt_extensions),
        )
    else:
        paths, exts = folder_paths.folder_names_and_paths[_VOID_MODEL_TYPE]
        folder_paths.folder_names_and_paths[_VOID_MODEL_TYPE] = (
            paths,
            set(exts) | set(folder_paths.supported_pt_extensions),
        )
    try:
        folder_paths.add_model_folder_path(_VOID_MODEL_TYPE, str(_VOID_MODELS_DIR), is_default=True)
    except TypeError:
        folder_paths.add_model_folder_path(_VOID_MODEL_TYPE, str(_VOID_MODELS_DIR))


def _soft_empty_cache() -> None:
    if mm is not None:
        try:
            mm.soft_empty_cache()
        except Exception:
            pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _cleanup_loaded_bundle(bundle: Optional[dict]) -> None:
    if not bundle:
        return
    pipeline = bundle.get("pipeline")
    if pipeline is not None:
        try:
            pipeline.maybe_free_model_hooks()
        except Exception:
            pass
        for module_name in ("transformer", "vae", "text_encoder"):
            module = getattr(pipeline, module_name, None)
            if module is not None:
                try:
                    module.to("cpu")
                except Exception:
                    pass
    bundle.clear()


def _clear_model_cache() -> None:
    for key in list(_MODEL_CACHE.keys()):
        _cleanup_loaded_bundle(_MODEL_CACHE.pop(key, None))
    gc.collect()
    _soft_empty_cache()


def _device() -> torch.device:
    if mm is not None:
        try:
            return mm.get_torch_device()
        except Exception:
            pass
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_base_model_dir(base_model_name: str) -> str:
    candidate = _VOID_MODELS_DIR / base_model_name
    if (candidate / "transformer").is_dir() and (candidate / "vae").is_dir():
        return str(candidate)

    raise FileNotFoundError(
        "VOID base model not found. Expected exactly: "
        f"'models/void/{base_model_name}'."
    )


def _checkpoint_choices():
    choices = {"void_pass1.safetensors", "void_pass2.safetensors"}
    try:
        choices.update(
            name for name in folder_paths.get_filename_list(_VOID_MODEL_TYPE) if name.endswith(".safetensors")
        )
    except Exception:
        pass
    return sorted(choices)


def _import_void_runtime():
    try:
        from diffusers import (
            CogVideoXDDIMScheduler,
            DDIMScheduler,
            DPMSolverMultistepScheduler,
            EulerAncestralDiscreteScheduler,
            EulerDiscreteScheduler,
            PNDMScheduler,
        )
    except ImportError as exc:
        raise ImportError(
            "ComfyUI-Void requires diffusers. Install the packages listed in "
            "custom_nodes/ComfyUI-Void/requirements.txt."
        ) from exc

    try:
        from void_runtime.models import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel, T5EncoderModel, T5Tokenizer
        from void_runtime.pipeline import CogVideoXFunInpaintPipeline
    except ImportError as exc:
        raise ImportError(
            "ComfyUI-Void could not import its vendored runtime package. "
            "Check that custom_nodes/ComfyUI-Void/void_runtime exists and its dependencies are installed."
        ) from exc

    schedulers = {
        "DDIM_Origin": DDIMScheduler,
        "DDIM_Cog": CogVideoXDDIMScheduler,
        "Euler": EulerDiscreteScheduler,
        "Euler A": EulerAncestralDiscreteScheduler,
        "DPM++": DPMSolverMultistepScheduler,
        "PNDM": PNDMScheduler,
    }

    return {
        "AutoencoderKLCogVideoX": AutoencoderKLCogVideoX,
        "CogVideoXTransformer3DModel": CogVideoXTransformer3DModel,
        "T5EncoderModel": T5EncoderModel,
        "T5Tokenizer": T5Tokenizer,
        "CogVideoXFunInpaintPipeline": CogVideoXFunInpaintPipeline,
        "schedulers": schedulers,
    }


def _resolve_checkpoint_path(model_name: str) -> str:
    full_path = None
    try:
        full_path = folder_paths.get_full_path(_VOID_MODEL_TYPE, model_name)
    except Exception:
        full_path = None
    if full_path and os.path.isfile(full_path):
        return full_path
    fallback = _VOID_MODELS_DIR / model_name
    if fallback.is_file():
        return str(fallback)
    raise FileNotFoundError(f"VOID checkpoint not found: {model_name}")


def _load_state_dict(path: str) -> dict:
    if path.endswith(".safetensors"):
        if load_safetensors is None:
            raise ImportError("safetensors is required to load VOID checkpoints.")
        state_dict = load_safetensors(path)
    else:
        state_dict = torch.load(path, map_location="cpu")
    return state_dict["state_dict"] if "state_dict" in state_dict else state_dict


def _scheduler_for_name(name: str, variant: str):
    runtime = _import_void_runtime()
    schedulers = runtime["schedulers"]
    if name == "model_default":
        name = "DDIM_Origin" if variant == "pass1" else "DDIM_Cog"
    return schedulers[name], name


def _load_void_pipeline(
    base_model_name: str,
    checkpoint_name: str,
    variant: str,
    precision: str,
    memory_mode: str,
    scheduler_name: str,
) -> dict:
    runtime = _import_void_runtime()
    AutoencoderKLCogVideoX = runtime["AutoencoderKLCogVideoX"]
    CogVideoXTransformer3DModel = runtime["CogVideoXTransformer3DModel"]
    T5EncoderModel = runtime["T5EncoderModel"]
    T5Tokenizer = runtime["T5Tokenizer"]
    CogVideoXFunInpaintPipeline = runtime["CogVideoXFunInpaintPipeline"]

    base_model_dir = _resolve_base_model_dir(base_model_name)
    checkpoint_path = _resolve_checkpoint_path(checkpoint_name)
    device = _device()
    weight_dtype = PRECISIONS[precision]
    scheduler_cls, resolved_scheduler_name = _scheduler_for_name(scheduler_name, variant)

    transformer = CogVideoXTransformer3DModel.from_pretrained(
        base_model_dir,
        subfolder="transformer",
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
        use_vae_mask=True,
        stack_mask=False,
    ).to(weight_dtype)

    state_dict = _load_state_dict(checkpoint_path)
    param_name = "patch_embed.proj.weight"
    if param_name in state_dict and param_name in transformer.state_dict():
        loaded_channels = state_dict[param_name].size(1)
        expected_channels = transformer.state_dict()[param_name].size(1)
        if loaded_channels != expected_channels:
            feat_dim = 16 * 8
            new_weight = transformer.state_dict()[param_name].clone()
            new_weight[:, :feat_dim] = state_dict[param_name][:, :feat_dim]
            new_weight[:, -feat_dim:] = state_dict[param_name][:, -feat_dim:]
            state_dict[param_name] = new_weight

    transformer.load_state_dict(state_dict, strict=False)

    vae = AutoencoderKLCogVideoX.from_pretrained(base_model_dir, subfolder="vae").to(weight_dtype)
    text_encoder = T5EncoderModel.from_pretrained(
        base_model_dir, subfolder="text_encoder", torch_dtype=weight_dtype
    )
    tokenizer = T5Tokenizer.from_pretrained(base_model_dir, subfolder="tokenizer")
    scheduler = scheduler_cls.from_pretrained(base_model_dir, subfolder="scheduler")

    pipeline = CogVideoXFunInpaintPipeline(
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        transformer=transformer,
        scheduler=scheduler,
    )

    if memory_mode == "sequential_cpu_offload":
        pipeline.enable_sequential_cpu_offload(device=device)
    elif memory_mode == "model_cpu_offload":
        pipeline.enable_model_cpu_offload(device=device)
    else:
        pipeline.to(device=device)

    defaults = {
        "use_trimask": variant == "pass1",
        "guidance_scale": 1.0 if variant == "pass1" else 6.0,
        "num_inference_steps": 30 if variant == "pass1" else 50,
        "negative_prompt": (
            "The video is not of a high quality, it has a low resolution. "
            "Watermark present in each frame. The background is solid. "
            "Strange body and strange trajectory. Distortion."
        ),
    }

    return {
        "pipeline": pipeline,
        "variant": variant,
        "precision": precision,
        "memory_mode": memory_mode,
        "scheduler_name": resolved_scheduler_name,
        "dtype": weight_dtype,
        "device": device,
        "base_model_dir": base_model_dir,
        "checkpoint_path": checkpoint_path,
        "base_model_name": base_model_name,
        "defaults": defaults,
    }


def _temporal_fit_sequence(seq: torch.Tensor, target_frames: int) -> torch.Tensor:
    if seq.shape[0] >= target_frames:
        return seq[:target_frames]
    if seq.shape[0] == 0:
        raise ValueError("Received an empty sequence.")
    pad = seq[-1:].repeat(target_frames - seq.shape[0], *([1] * (seq.ndim - 1)))
    return torch.cat([seq, pad], dim=0)


def _prepare_image_sequence(images: torch.Tensor, height: int, width: int, max_video_length: int):
    if images.ndim != 4:
        raise ValueError(f"`images` must be [frames, height, width, channels], got {tuple(images.shape)}")
    images = images[..., :3].float().clamp(0.0, 1.0)
    original_frames = min(images.shape[0], max_video_length)
    images = images[:max_video_length]
    images_chw = images.permute(0, 3, 1, 2)
    if images_chw.shape[-2:] != (height, width):
        images_chw = F.interpolate(images_chw, size=(height, width), mode="bilinear", align_corners=False)
    return images_chw, original_frames


def _prepare_video_tensor(
    images: torch.Tensor,
    height: int,
    width: int,
    max_video_length: int,
    temporal_window_size: int,
) -> Tuple[torch.Tensor, int]:
    images_chw, original_frames = _prepare_image_sequence(images, height, width, max_video_length)
    images_chw = _temporal_fit_sequence(images_chw, max(temporal_window_size, images_chw.shape[0]))
    video = images_chw.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
    return video, original_frames


def _validate_temporal_window_size(temporal_window_size: int, pipeline) -> None:
    patch_size_t = getattr(pipeline.transformer.config, "patch_size_t", None)
    if patch_size_t is None:
        return

    latent_frames = (temporal_window_size - 1) // pipeline.vae_scale_factor_temporal + 1
    if latent_frames % patch_size_t != 0:
        raise ValueError(
            "Invalid temporal_window_size for this VOID model. "
            f"Got {temporal_window_size}, which produces {latent_frames} latent frames; "
            f"that must be divisible by patch_size_t={patch_size_t}. "
            "Use values like 85, 93, 101, ..., 197."
        )


def _prepare_mask_sequence(mask: torch.Tensor, frame_count: int, height: int, width: int) -> torch.Tensor:
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    elif mask.ndim == 4 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    elif mask.ndim != 3:
        raise ValueError(f"`mask` must be [frames, height, width] or [height, width], got {tuple(mask.shape)}")

    mask = mask.float().clamp(0.0, 1.0)
    mask = _temporal_fit_sequence(mask, frame_count)
    if mask.shape[-2:] != (height, width):
        mask = F.interpolate(mask.unsqueeze(1), size=(height, width), mode="nearest").squeeze(1)
    return mask


def _mask_dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    kernel = radius * 2 + 1
    return F.max_pool2d(mask.unsqueeze(1), kernel_size=kernel, stride=1, padding=radius).squeeze(1)


def _fill_holes(mask: torch.Tensor) -> torch.Tensor:
    if binary_fill_holes is None:
        return mask
    filled = []
    for frame in mask.cpu().numpy():
        filled.append(binary_fill_holes(frame > 0.5).astype(np.float32))
    return torch.from_numpy(np.stack(filled, axis=0))


def _quadmask_to_ready_mask(quadmask_uint8: np.ndarray) -> torch.Tensor:
    ready = 255 - quadmask_uint8.astype(np.uint8)
    return torch.from_numpy(ready.astype(np.float32) / 255.0)


def _quantize_quadmask_from_images(quadmask_images: torch.Tensor) -> np.ndarray:
    if quadmask_images.ndim != 4:
        raise ValueError(
            f"`quadmask_images` must be [frames, height, width, channels], got {tuple(quadmask_images.shape)}"
        )
    gray = quadmask_images.float().clamp(0.0, 1.0)
    if gray.shape[-1] > 1:
        gray = gray.mean(dim=-1)
    else:
        gray = gray[..., 0]
    gray = (gray * 255.0).round().to(torch.uint8).cpu().numpy()
    quantized = np.where(gray <= 31, 0, gray)
    quantized = np.where((quantized > 31) & (quantized <= 95), 63, quantized)
    quantized = np.where((quantized > 95) & (quantized <= 191), 127, quantized)
    quantized = np.where(quantized > 191, 255, quantized)
    return quantized.astype(np.uint8)


def _build_quadmask_from_masks(
    remove_mask: torch.Tensor,
    affected_mask: Optional[torch.Tensor],
    threshold: float,
    remove_expand: int,
    affected_expand: int,
    auto_affected_expand: int,
    fill_holes: bool,
) -> np.ndarray:
    remove_bin = (remove_mask >= threshold).float()
    if fill_holes:
        remove_bin = _fill_holes(remove_bin)
    remove_bin = (_mask_dilate(remove_bin, remove_expand) >= 0.5).float()

    if affected_mask is not None:
        affected_bin = (affected_mask >= threshold).float()
        if fill_holes:
            affected_bin = _fill_holes(affected_bin)
        affected_bin = (_mask_dilate(affected_bin, affected_expand) >= 0.5).float()
        overlap = (remove_bin > 0.5) & (affected_bin > 0.5)
        pure_affected = (affected_bin > 0.5) & (~overlap)
        pure_remove = (remove_bin > 0.5) & (~overlap)
    else:
        overlap = torch.zeros_like(remove_bin, dtype=torch.bool)
        pure_remove = remove_bin > 0.5
        if auto_affected_expand > 0:
            auto_affected = (_mask_dilate(remove_bin, auto_affected_expand) >= 0.5) & (~pure_remove)
            pure_affected = auto_affected
        else:
            pure_affected = torch.zeros_like(remove_bin, dtype=torch.bool)

    quadmask = np.full(remove_bin.shape, 255, dtype=np.uint8)
    quadmask[pure_affected.cpu().numpy()] = 127
    quadmask[overlap.cpu().numpy()] = 63
    quadmask[pure_remove.cpu().numpy()] = 0
    return quadmask


def _mask_sequence_from_images(mask_images: torch.Tensor) -> torch.Tensor:
    if mask_images.ndim != 4:
        raise ValueError(
            f"`mask_images` must be [frames, height, width, channels], got {tuple(mask_images.shape)}"
        )
    mask_gray = mask_images.float().clamp(0.0, 1.0)
    if mask_gray.shape[-1] > 1:
        mask_gray = mask_gray.mean(dim=-1)
    else:
        mask_gray = mask_gray[..., 0]
    return mask_gray


def _preview_from_quadmask(quadmask_uint8: np.ndarray) -> torch.Tensor:
    preview = torch.from_numpy(quadmask_uint8.astype(np.float32) / 255.0)
    return preview.unsqueeze(-1).repeat(1, 1, 1, 3)


def _warped_noise_cache_key(pass1_images: torch.Tensor) -> str:
    sample_count = min(pass1_images.shape[0], 4)
    sample = pass1_images[:sample_count].detach().cpu().numpy().astype(np.float16)
    digest = str(abs(hash(sample.tobytes())))
    return f"{pass1_images.shape[0]}_{pass1_images.shape[1]}_{pass1_images.shape[2]}_{digest}"


def _generate_warped_noise_from_images(pass1_images: torch.Tensor, fps: int) -> str:
    script_path = _VOID_RUNTIME_ROOT / "make_warped_noise.py"
    if not script_path.is_file():
        raise FileNotFoundError(f"Warped-noise script not found: {script_path}")

    cache_root = _VOID_CACHE_DIR / "warped_noise"
    os.makedirs(cache_root, exist_ok=True)
    cache_key = _warped_noise_cache_key(pass1_images)
    output_dir = cache_root / cache_key
    noise_path = output_dir / "noises.npy"
    if noise_path.is_file():
        return str(noise_path)

    temp_video = cache_root / f"{cache_key}.mp4"
    frames_uint8 = (pass1_images.float().clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
    imageio.mimsave(temp_video, frames_uint8, fps=fps, codec="libx264", quality=8, pixelformat="yuv420p")

    if output_dir.exists():
        shutil.rmtree(output_dir, ignore_errors=True)

    result = subprocess.run(
        [sys.executable, str(script_path), str(temp_video), str(output_dir)],
        capture_output=True,
        text=True,
        cwd=str(_VOID_RUNTIME_ROOT),
    )
    if result.returncode != 0 or not noise_path.is_file():
        raise RuntimeError(
            "Failed to generate warped noise for VOID pass2.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return str(noise_path)


def _load_and_resize_warped_noise(noise_path: str, target_shape: Tuple[int, int, int, int], device, dtype) -> torch.Tensor:
    latent_t, latent_h, latent_w, latent_c = target_shape
    warped_noise_np = np.load(noise_path)
    if warped_noise_np.dtype == np.float16:
        warped_noise_np = warped_noise_np.astype(np.float32)
    if warped_noise_np.ndim == 4 and warped_noise_np.shape[1] == latent_c:
        warped_noise_np = warped_noise_np.transpose(0, 2, 3, 1)

    warped_noise = torch.from_numpy(warped_noise_np).float()
    if warped_noise.ndim != 4:
        raise ValueError(f"Warped noise must be 4D, got {tuple(warped_noise.shape)}")
    warped_noise = warped_noise.permute(0, 3, 1, 2)

    if warped_noise.shape[0] != latent_t:
        indices = torch.linspace(0, warped_noise.shape[0] - 1, steps=latent_t).round().long()
        warped_noise = warped_noise.index_select(0, indices)

    if warped_noise.shape[-2:] != (latent_h, latent_w):
        warped_noise = F.interpolate(warped_noise, size=(latent_h, latent_w), mode="bilinear", align_corners=False)

    warped_noise = warped_noise.unsqueeze(0).to(device=device, dtype=dtype)
    return warped_noise


def _loader_input_types(default_checkpoint: str):
    return {
        "required": {
            "base_model": (BASE_MODEL_CHOICES, {"default": BASE_MODEL_CHOICES[0]}),
            "checkpoint_name": (_checkpoint_choices(), {"default": default_checkpoint}),
            "precision": (list(PRECISIONS.keys()), {"default": "bf16"}),
            "memory_mode": (
                ["model_cpu_offload", "sequential_cpu_offload", "model_full_load"],
                {"default": "model_cpu_offload"},
            ),
            "scheduler": (["model_default"] + SCHEDULER_NAMES, {"default": "model_default"}),
            "unload_cached_void": ("BOOLEAN", {"default": False}),
            "unload_all_comfy_models": ("BOOLEAN", {"default": False}),
        }
    }


def _load_fixed_variant_model(
    *,
    variant: str,
    base_model: str,
    checkpoint_name: str,
    precision: str,
    memory_mode: str,
    scheduler: str,
    unload_cached_void: bool,
    unload_all_comfy_models: bool,
):
    if variant == "pass1" and "pass2" in checkpoint_name:
        checkpoint_name = "void_pass1.safetensors"
    if variant == "pass2" and "pass1" in checkpoint_name:
        checkpoint_name = "void_pass2.safetensors"

    cache_key = (base_model, variant, checkpoint_name, precision, memory_mode, scheduler)
    with _MODEL_LOCK:
        if unload_all_comfy_models and mm is not None:
            try:
                mm.unload_all_models()
                mm.cleanup_models()
            except Exception:
                pass
            _soft_empty_cache()

        if unload_cached_void:
            _clear_model_cache()
        elif cache_key in _MODEL_CACHE:
            return _MODEL_CACHE[cache_key]

        bundle = _load_void_pipeline(
            base_model_name=base_model,
            checkpoint_name=checkpoint_name,
            variant=variant,
            precision=precision,
            memory_mode=memory_mode,
            scheduler_name=scheduler,
        )
        _MODEL_CACHE[cache_key] = bundle
        return bundle


class VOIDPass1ModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return _loader_input_types("void_pass1.safetensors")

    RETURN_TYPES = ("VOID_PASS1_MODEL",)
    RETURN_NAMES = ("void_pass1_model",)
    FUNCTION = "load_model"
    CATEGORY = "VOID"

    def load_model(
        self,
        base_model: str,
        checkpoint_name: str,
        precision: str,
        memory_mode: str,
        scheduler: str,
        unload_cached_void: bool,
        unload_all_comfy_models: bool,
    ):
        _register_void_model_dir()
        bundle = _load_fixed_variant_model(
            variant="pass1",
            base_model=base_model,
            checkpoint_name=checkpoint_name,
            precision=precision,
            memory_mode=memory_mode,
            scheduler=scheduler,
            unload_cached_void=unload_cached_void,
            unload_all_comfy_models=unload_all_comfy_models,
        )
        return (bundle,)


class VOIDPass2ModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return _loader_input_types("void_pass2.safetensors")

    RETURN_TYPES = ("VOID_PASS2_MODEL",)
    RETURN_NAMES = ("void_pass2_model",)
    FUNCTION = "load_model"
    CATEGORY = "VOID"

    def load_model(
        self,
        base_model: str,
        checkpoint_name: str,
        precision: str,
        memory_mode: str,
        scheduler: str,
        unload_cached_void: bool,
        unload_all_comfy_models: bool,
    ):
        _register_void_model_dir()
        bundle = _load_fixed_variant_model(
            variant="pass2",
            base_model=base_model,
            checkpoint_name=checkpoint_name,
            precision=precision,
            memory_mode=memory_mode,
            scheduler=scheduler,
            unload_cached_void=unload_cached_void,
            unload_all_comfy_models=unload_all_comfy_models,
        )
        return (bundle,)


class VOIDMaskProcessor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "remove_expand": ("INT", {"default": 0, "min": 0, "max": 256, "step": 1}),
                "affected_expand": ("INT", {"default": 0, "min": 0, "max": 256, "step": 1}),
                "auto_affected_expand": ("INT", {"default": 0, "min": 0, "max": 256, "step": 1}),
                "fill_mask_holes": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "mask_video": ("MASK",),
                "mask_video_images": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("void_mask", "quadmask_preview")
    FUNCTION = "process_mask"
    CATEGORY = "VOID"

    def process_mask(
        self,
        threshold: float,
        remove_expand: int,
        affected_expand: int,
        auto_affected_expand: int,
        fill_mask_holes: bool,
        mask_video: Optional[torch.Tensor] = None,
        mask_video_images: Optional[torch.Tensor] = None,
    ):
        if mask_video is not None:
            remove_seq = mask_video if mask_video.ndim != 2 else mask_video.unsqueeze(0)
            frame_count = remove_seq.shape[0]
            height, width = remove_seq.shape[-2:]
            remove_seq = _prepare_mask_sequence(remove_seq, frame_count, height, width)
            quadmask = _build_quadmask_from_masks(
                remove_mask=remove_seq,
                affected_mask=None,
                threshold=threshold,
                remove_expand=remove_expand,
                affected_expand=affected_expand,
                auto_affected_expand=auto_affected_expand,
                fill_holes=fill_mask_holes,
            )
        elif mask_video_images is not None:
            remove_seq = _mask_sequence_from_images(mask_video_images)
            frame_count = remove_seq.shape[0]
            height, width = remove_seq.shape[-2:]
            remove_seq = _prepare_mask_sequence(remove_seq, frame_count, height, width)
            quadmask = _build_quadmask_from_masks(
                remove_mask=remove_seq,
                affected_mask=None,
                threshold=threshold,
                remove_expand=remove_expand,
                affected_expand=affected_expand,
                auto_affected_expand=auto_affected_expand,
                fill_holes=fill_mask_holes,
            )
        else:
            raise ValueError(
                "Provide mask_video or mask_video_images. "
                "Use mask_video when the upstream node outputs MASK, and mask_video_images when it outputs IMAGE."
            )

        ready_mask = _quadmask_to_ready_mask(quadmask)
        preview = _preview_from_quadmask(quadmask)
        return (ready_mask, preview)


def _run_void_inpaint(
    *,
    void_model: dict,
    images: torch.Tensor,
    mask: torch.Tensor,
    prompt: str,
    seed: int,
    height: int,
    width: int,
    temporal_window_size: int,
    max_video_length: int,
    use_model_defaults: bool,
    guidance_scale: float,
    num_inference_steps: int,
    fps: int,
    negative_prompt: str = "",
    pass1_images: Optional[torch.Tensor] = None,
    warped_noise_path: str = "",
):
    pipeline = void_model["pipeline"]
    defaults = void_model["defaults"]
    device = void_model["device"]
    dtype = void_model["dtype"]

    _validate_temporal_window_size(temporal_window_size, pipeline)

    if use_model_defaults:
        guidance_scale = defaults["guidance_scale"]
        num_inference_steps = defaults["num_inference_steps"]
    if not negative_prompt.strip():
        negative_prompt = defaults["negative_prompt"]

    input_video, original_frames = _prepare_video_tensor(
        images=images,
        height=height,
        width=width,
        max_video_length=max_video_length,
        temporal_window_size=temporal_window_size,
    )
    frame_count = input_video.shape[2]
    input_mask = _prepare_mask_sequence(mask, frame_count, height, width).unsqueeze(0).unsqueeze(0)

    latents = None
    if void_model["variant"] == "pass2":
        noise_path = warped_noise_path.strip()
        if not noise_path:
            if pass1_images is None:
                raise ValueError("VOID pass2 needs either pass1_images or warped_noise_path.")
            noise_path = _generate_warped_noise_from_images(pass1_images, fps=fps)

        latent_t = (frame_count - 1) // 4 + 1
        latent_h = height // 8
        latent_w = width // 8
        latents = _load_and_resize_warped_noise(
            noise_path,
            (latent_t, latent_h, latent_w, 16),
            device=device,
            dtype=dtype,
        )

    generator = torch.Generator(device=device).manual_seed(seed)
    progress_kwargs = {"comfyui_progressbar": ProgressBar is not None}

    with torch.no_grad():
        output = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=frame_count,
            temporal_window_size=temporal_window_size,
            height=height,
            width=width,
            generator=generator,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            video=input_video,
            mask_video=input_mask,
            strength=1.0,
            use_trimask=defaults["use_trimask"],
            zero_out_mask_region=False,
            use_vae_mask=True,
            stack_mask=False,
            latents=latents,
            **progress_kwargs,
        ).videos

    result = output if isinstance(output, torch.Tensor) else torch.from_numpy(output)
    if result.ndim != 5:
        raise ValueError(f"Unexpected VOID output shape: {tuple(result.shape)}")
    result_frames = result[0]
    if result_frames.shape[0] in (1, 3):
        result_frames = result_frames.permute(1, 2, 3, 0)
    result_frames = result_frames.clamp(0.0, 1.0)
    result_frames = result_frames[:original_frames]

    _soft_empty_cache()
    return (result_frames,)


class VOIDPass1Inpaint:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "void_pass1_model": ("VOID_PASS1_MODEL",),
                "images": ("IMAGE",),
                "mask": ("MASK",),
                "prompt": ("STRING", {"multiline": True}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "height": ("INT", {"default": 384, "min": 64, "max": 4096, "step": 8}),
                "width": ("INT", {"default": 672, "min": 64, "max": 4096, "step": 8}),
                "temporal_window_size": ("INT", {"default": 85, "min": 5, "max": 4093, "step": 8}),
                "max_video_length": ("INT", {"default": 197, "min": 1, "max": 4096, "step": 1}),
                "use_model_defaults": ("BOOLEAN", {"default": True}),
                "guidance_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 50.0, "step": 0.1}),
                "num_inference_steps": ("INT", {"default": 30, "min": 1, "max": 500, "step": 1}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = "VOID"

    def run(
        self,
        void_pass1_model: dict,
        images: torch.Tensor,
        mask: torch.Tensor,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        temporal_window_size: int,
        max_video_length: int,
        use_model_defaults: bool,
        guidance_scale: float,
        num_inference_steps: int,
        negative_prompt: str = "",
    ):
        return _run_void_inpaint(
            void_model=void_pass1_model,
            images=images,
            mask=mask,
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            temporal_window_size=temporal_window_size,
            max_video_length=max_video_length,
            use_model_defaults=use_model_defaults,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            fps=24,
            negative_prompt=negative_prompt,
        )


class VOIDPass2Inpaint:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "void_pass2_model": ("VOID_PASS2_MODEL",),
                "images": ("IMAGE",),
                "mask": ("MASK",),
                "prompt": ("STRING", {"multiline": True}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "height": ("INT", {"default": 384, "min": 64, "max": 4096, "step": 8}),
                "width": ("INT", {"default": 672, "min": 64, "max": 4096, "step": 8}),
                "temporal_window_size": ("INT", {"default": 85, "min": 5, "max": 4093, "step": 8}),
                "max_video_length": ("INT", {"default": 197, "min": 1, "max": 4096, "step": 1}),
                "use_model_defaults": ("BOOLEAN", {"default": True}),
                "guidance_scale": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 50.0, "step": 0.1}),
                "num_inference_steps": ("INT", {"default": 50, "min": 1, "max": 500, "step": 1}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 240, "step": 1}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "pass1_images": ("IMAGE",),
                "warped_noise_path": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = "VOID"

    def run(
        self,
        void_pass2_model: dict,
        images: torch.Tensor,
        mask: torch.Tensor,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        temporal_window_size: int,
        max_video_length: int,
        use_model_defaults: bool,
        guidance_scale: float,
        num_inference_steps: int,
        fps: int,
        negative_prompt: str = "",
        pass1_images: Optional[torch.Tensor] = None,
        warped_noise_path: str = "",
    ):
        return _run_void_inpaint(
            void_model=void_pass2_model,
            images=images,
            mask=mask,
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            temporal_window_size=temporal_window_size,
            max_video_length=max_video_length,
            use_model_defaults=use_model_defaults,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            fps=fps,
            negative_prompt=negative_prompt,
            pass1_images=pass1_images,
            warped_noise_path=warped_noise_path,
        )


_register_void_model_dir()

NODE_CLASS_MAPPINGS = {
    "VOIDPass1ModelLoader": VOIDPass1ModelLoader,
    "VOIDPass2ModelLoader": VOIDPass2ModelLoader,
    "VOIDMaskProcessor": VOIDMaskProcessor,
    "VOIDPass1Inpaint": VOIDPass1Inpaint,
    "VOIDPass2Inpaint": VOIDPass2Inpaint,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VOIDPass1ModelLoader": "VOID Pass1 Model Loader",
    "VOIDPass2ModelLoader": "VOID Pass2 Model Loader",
    "VOIDMaskProcessor": "VOID Quadmask Processor",
    "VOIDPass1Inpaint": "VOID Pass1 Inpaint",
    "VOIDPass2Inpaint": "VOID Pass2 Inpaint",
}
