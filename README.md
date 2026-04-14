# ComfyUI-Void

ComfyUI implementation of [Netflix/void-model](https://github.com/Netflix/void-model/tree/main).

This custom node package wraps the VOID inference pipeline for ComfyUI and exposes a small set of nodes for:

- loading VOID Pass 1 / Pass 2 checkpoints
- preparing quadmask-style inpainting masks
- running VOID video inpainting inside a ComfyUI workflow

This is an inference-focused port of the original project. It vendors the runtime needed for VOID/CogVideoX-Fun inference and adapts it to ComfyUI node execution.

## Upstream Project

- Project: https://github.com/Netflix/void-model/tree/main
- Model page: https://huggingface.co/netflix/void-model
- Base model: `CogVideoX-Fun-V1.5-5b-InP`

## Nodes

This package registers three nodes under the `VOID` category:

- `VOID Model Loader`
- `VOID Quadmask Processor`
- `VOID Inpaint`

## Requirements

Install the Python dependencies listed in `requirements.txt`.

Typical environment requirements:

- ComfyUI
- PyTorch with CUDA support
- `diffusers`
- `transformers`
- `safetensors`
- `imageio` / `imageio-ffmpeg`
- enough VRAM for CogVideoX-Fun + VOID inference

Install with:

```bash
cd ComfyUI/custom_nodes/ComfyUI-Void
pip install -r requirements.txt
```

## Installation

Clone or place this folder under:

```text
ComfyUI/custom_nodes/ComfyUI-Void
```

Then restart ComfyUI.

## Model Setup

Place models in the ComfyUI model directory used by this node:

```text
ComfyUI/models/void/
```

Expected layout:

```text
ComfyUI/models/void/
├── CogVideoX-Fun-V1.5-5b-InP/
│   ├── transformer/
│   ├── vae/
│   ├── tokenizer/
│   ├── text_encoder/
│   └── scheduler/
├── void_pass1.safetensors
└── void_pass2.safetensors
```

Files you need:

1. Download the base model:

```bash
hf download alibaba-pai/CogVideoX-Fun-V1.5-5b-InP \
  --local-dir ComfyUI/models/void/CogVideoX-Fun-V1.5-5b-InP
```

2. Download VOID checkpoints from Hugging Face:

- `void_pass1.safetensors`
- `void_pass2.safetensors`

## Basic Workflow

Recommended workflow:

1. Load your source video frames into ComfyUI as `IMAGE`.
2. Create or load a mask sequence.
3. Use `VOID Quadmask Processor` to convert masks into the expected VOID-ready mask format.
4. Use `VOID Model Loader` to load either Pass 1 or Pass 2.
5. Run `VOID Inpaint`.
6. Save the output frames back to video with your preferred ComfyUI video save node.

Recommended mask node: https://github.com/9nate-drake/Comfyui-SecNodes.git

## Node Notes

### VOID Model Loader

Loads:

- base model directory
- VOID checkpoint
- scheduler
- precision
- Pass 1 / Pass 2 variant

Use:

- `pass1` for standard inference
- `pass2` when you want warped-noise refinement

### VOID Quadmask Processor

Builds VOID-style masks from an input mask or mask-image sequence.

The resulting mask follows the VOID semantic layout:

- `0`: remove region
- `63`: overlap region
- `127`: affected region
- `255`: keep/background

### VOID Inpaint

Runs the VOID inpainting pipeline on the video sequence.

Important parameters:

- `temporal_window_size`: multidiffusion window size, not total output duration
- `max_video_length`: max number of frames to process from the input
- `fps`: used when generating temporary warped-noise video for Pass 2

## Important Behavior

- Output frame count follows the actual processed input length, not `temporal_window_size`.
- `temporal_window_size` only controls the temporal inference window.
- Some frame counts may be clipped slightly at the tail to satisfy the model's temporal patching constraints.
- Pass 2 requires either `pass1_images` or a `warped_noise_path`.

## Fixes In This Port

- Fixed an issue where total output duration could be incorrectly truncated by `temporal_window_size`. The window size now only controls the temporal multidiffusion window, while output length follows the actual processed input frame count.
- Fixed a crash caused by invalid latent temporal dimensions for certain frame counts. Temporal lengths are now aligned before entering the model so they satisfy the patch-size requirements.

## Tips

- Start with Pass 1 first.
- Use the upstream defaults as a baseline: `384x672`, `temporal_window_size=85`.
- If you process longer videos, increase `max_video_length` accordingly.
- Keep your output video save node fps consistent with the source fps.

## Limitations

- This is not a full reproduction of the upstream training/data-generation stack.
- The ComfyUI port focuses on inference only.
- Performance and memory usage depend heavily on VRAM, resolution, frame count, and scheduler choice.

## Credits

Based on the original VOID project by Netflix Research:

- https://github.com/Netflix/void-model/tree/main
