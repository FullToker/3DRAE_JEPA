<h1 align="center">[NeurIPS 2026] 🌍 Any 3D Scene is Worth 1K Tokens: 3D-Grounded Representation for Scene Generation at Scale</h1>


<p align="center">
  <a href="mailto:weidongxu@westlake.edu.cn">Dongxu Wei*</a>
  &nbsp;·&nbsp;
  <a href="mailto:xuqiwhu@whu.edu.cn">Qi Xu*</a>
  &nbsp;·&nbsp;
  <a href="mailto:lizhiqi49@westlake.edu.cn">Zhiqi Li</a>
  &nbsp;·&nbsp;
  <a href="mailto:zhouhangning@qianli-drive.com">Hangning Zhou†</a>
  &nbsp;·&nbsp;
  <a href="mailto:qiucong@qianli-drive.com">Cong Qiu</a>
  &nbsp;·&nbsp;<br>
  <a href="https://scholar.google.com/citations?user=2RvMxVUAAAAJ&hl=en">Hailong Qin</a>
  &nbsp;·&nbsp;
  <a href="mailto:ym@qianli-drive.com">Mu Yang</a>
  &nbsp;·&nbsp;
  <a href="https://zhpcui.github.io/">Zhaopeng Cui</a>
  &nbsp;·&nbsp;
  <a href="https://ethliup.github.io/">Peidong Liu✉️</a>
</p>

<h3 align="center">
  <a href="https://arxiv.org/abs/2604.11331">Paper</a> |
  <a href="https://wswdx.github.io/3DRAE">Project Page</a> |
  <a href="https://huggingface.co/tracywei/3drae">Models</a> |
  <a href="https://huggingface.co/datasets/tracywei/scannet_3drae">Data</a>
</h3>

<p align="center">If you find this repository useful, please consider giving it a star 🌟.</p>

## News

- **2026-10-20** — Code, pretrained models, and processed data released.
- **2026-10-20** — The revised NeurIPS 2026 paper became available on [arXiv](https://arxiv.org/abs/2604.11331).
- **2026-09-26** — 🎉 Congratulations! Our paper is accepted to NeurIPS 2026.
- **2026-04-13** — The initial paper was released on [arXiv](https://arxiv.org/abs/2604.11331).

## Overview

**3DRAE** grounds multi-view observed 3D scenes into fixed-length (e.g., 256, 512, 1024) 3D latent tokens, and decodes them back to view-based images and point maps. **3DDiT** directly performs diffusion modeling within this 3D latent space to generate 3D scenes **in 2 seconds**.

https://github.com/user-attachments/assets/9c584612-5fa0-4cfb-b711-046b8c0a2446

## Quick Start

All commands below are run from the repository root.

### Installation

The recommended environment is Python 3.10 with CUDA 12.4:

```bash
uv sync --python 3.10
source .venv/bin/activate
```

All commands below assume that `.venv` is active.

```bash
python -c "import numpy, torch; print(numpy.__version__, torch.__version__)"
```

### Prepare Models

Download the complete checkpoint bundle from [Hugging Face](https://huggingface.co/tracywei/3drae) directly into `checkpoints/`:

```bash
hf download tracywei/3drae \
  --exclude "README.md" ".gitattributes" \
  --local-dir checkpoints
```

The resulting directory has the following layout:

```text
checkpoints/
├── backbones/
│   ├── dinov2-with-registers-base/
│   │   ├── config.json
│   │   └── model.safetensors
│   └── siglip2-base-patch16-256/
│       ├── config.json
│       └── model.safetensors
├── dino2/
│   ├── neck.safetensors
│   ├── decoder.safetensors
│   ├── point_decoder.safetensors
│   ├── stats.safetensors
│   └── 3ddit.safetensors
├── dino2_256/
│   ├── neck.safetensors
│   ├── decoder.safetensors
│   └── stats.safetensors
├── siglip2/
│   ├── neck.safetensors
│   ├── decoder.safetensors
│   └── stats.safetensors
├── discriminator/
│   └── dino_vit_small_patch8_224.pth
├── perceptual/
│   ├── vgg16-397923af.pth
│   └── vgg.pth
└── manifest.json
```

The bundle includes local copies of [`facebook/dinov2-with-registers-base`](https://huggingface.co/facebook/dinov2-with-registers-base) and [`google/siglip2-base-patch16-256`](https://huggingface.co/google/siglip2-base-patch16-256). The discriminator and perceptual weights are needed only for Stage 1 training. Verify all files against the release manifest with:

```bash
python tools/verify_checkpoints.py
```

Use `--skip-sha256` for a faster structural check.

### Prepare Data

Six calibrated ten-frame examples are included for quick tests. Each contains six candidate views for conditioning and four held-out views for targeting:

- `examples/scene_000`
- `examples/scene_001`
- `examples/scene_002`
- `examples/scene_003`
- `examples/scene_004`
- `examples/scene_005`

A 256-frame example is also included for large-scale reconstruction. It contains 128 interleaved condition views and 128 target views:

- `examples/scene_006`

See [`examples/README.md`](examples/README.md) for the corresponding reconstruction command.

For training, download the [processed ScanNet release](https://huggingface.co/datasets/tracywei/scannet_3drae). It is stored as split TAR parts. Download all parts into a temporary directory:

```bash
hf download tracywei/scannet_3drae \
  --repo-type dataset \
  --include "scannet.tar.part-*" \
  --local-dir data/scannet_3drae_download
```

Concatenate the parts in filename order and stream the complete archive directly to `tar`:

```bash
mkdir -p data
cat data/scannet_3drae_download/scannet.tar.part-* | \
  tar --numeric-owner -xpf - -C data/
```

The archive extracts to `data/scannet` with the following structure:

```text
data/scannet/
├── train/
│   └── <scene>/
│       ├── images/*.png
│       ├── intrinsics.npy
│       ├── extrinsics.npy
│       └── metadata.json
└── test/
    └── <scene>/...
```

The training configs already point to `data/scannet`. Instructions for preparing the same format from the original ScanNet export are provided under [ScanNet preparation](#scannet-preparation).

### 3D Feature Extraction

`FeatureExtractor3D` accepts RGB images in `[S, 3, H, W]` or `[B, S, 3, H, W]`, camera intrinsics, and OpenCV world-to-camera extrinsics. The default DINOv2 model returns normalized `[B, 1024, 768]` latent tokens in memory.

```python
from threedrae import FeatureExtractor3D, load_scene

scene = load_scene("examples/scene_000", frame_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
extractor = FeatureExtractor3D(
    model_name="dino2",  # or "siglip2"
    device="cuda",
    precision="bf16",
)
latent_feature_3d = extractor(
    scene.images,
    scene.intrinsics,
    scene.extrinsics,
)
print(latent_feature_3d.shape)  # [1, 1024, 768]
```

We also provide a compact DINOv2 model with only 256 latent tokens as the 3D representation:

```python
extractor = FeatureExtractor3D(
    model_name="dino2-256",
    device="cuda",
    precision="bf16",
)
latent_feature_3d = extractor(
    scene.images,
    scene.intrinsics,
    scene.extrinsics,
)
print(latent_feature_3d.shape)  # [1, 256, 768]
```

### Stage 1 Reconstruction and Novel-View Synthesis

Decode the 3D latent at the condition cameras for reconstruction, or at different cameras for novel-view synthesis. The DINOv2 decoder returns RGB, world-space point maps, and confidence maps; the SigLIP2 decoder returns RGB only.

```python
from threedrae import FeatureExtractor3D, Stage1Decoder, load_scene

condition = load_scene("examples/scene_000", frame_ids=[0, 2, 4, 6, 8])
target = load_scene("examples/scene_000", frame_ids=[1, 3, 5, 7, 9])

extractor = FeatureExtractor3D("dino2", device="cuda", precision="bf16")
decoder = Stage1Decoder("dino2", device="cuda", precision="fp32")

latent_feature_3d = extractor(
    condition.images,
    condition.intrinsics,
    condition.extrinsics,
)
images, points, confidence = decoder(
    latent_feature_3d,
    target.intrinsics,
    target.extrinsics,
    image_size=tuple(target.images.shape[-2:]),
)
```

For 256-token RGB reconstruction or NVS, initialize both modules with `"dino2-256"`. This checkpoint does not include a point decoder, so `points` and `confidence` are `None`.

```python
extractor = FeatureExtractor3D("dino2-256", device="cuda", precision="bf16")
decoder = Stage1Decoder("dino2-256", device="cuda", precision="fp32")
```

### Large Scene with Extremely Dense Views

The bundled DL3DV example contains a continuous 256-frame camera sequence at 350×196 resolution. Even-numbered frames provide 128 condition views, while odd-numbered frames provide 128 interleaved target views. Both sets span the full camera trajectory of the large outdoor scene.

Extract a single 3D representation from all 128 condition views:

```bash
python scripts/extract_features.py \
  --model dino2 \
  --scene examples/scene_006 \
  --frame-ids {0..254..2}
```

Reconstruct the scene at all 128 interleaved target cameras:

```bash
python scripts/reconstruct.py \
  --model dino2 \
  --scene examples/scene_006 \
  --condition-frame-ids {0..254..2} \
  --target-frame-ids {1..255..2} \
  --output-dir outputs/dl3dv_reconstruct
```

The reconstruction exports all condition images, 128 decoded target RGB images and point maps, and a fused `point_cloud.ply` built from the target-view point maps.

### Stage 2 Scene Generation

`SceneGenerator3D` separates appearance conditioning from trajectory conditioning. The images selected by `visible_ids` provide appearance, while all input camera poses define the spatial extent of the generated scene. The target cameras are used only by the Stage 1 decoder and do not affect the generated 3D latent.

```python
import torch

from threedrae import SceneGenerator3D, Stage1Decoder, load_scene

trajectory = load_scene(
    "examples/scene_000",
    frame_ids=[0, 1, 2, 3, 4, 5],
)
target = load_scene("examples/scene_000", frame_ids=[1, 3, 5, 7, 9])

generator = SceneGenerator3D(device="cuda", precision="bf16")
decoder = Stage1Decoder("dino2", device="cuda", precision="fp32")

generated_latent = generator(
    trajectory.images,
    trajectory.intrinsics,
    trajectory.extrinsics,
    visible_ids=torch.tensor([0, 2, 4, 6, 8]),  # positions of two appearance views
    seed=0,
    guidance_scale=2.0,
    num_steps=50,
)
images, points, confidence = decoder(
    generated_latent,
    target.intrinsics,
    target.extrinsics,
    image_size=tuple(target.images.shape[-2:]),
)
```

Note that `visible_ids` can vary to any view id and number of views to meet flexible conditioning requirements.

## Detailed Usage

### Scene Input Format

Inference scenes contain an `images/` directory and a `metadata.json` file:

```text
<scene>/
├── images/
│   ├── 000000.png
│   └── ...
└── metadata.json
```

All images in a scene must have the same resolution. Arbitrary positive image dimensions are supported: inference temporarily resizes the inputs and camera intrinsics to the model's patch grid, adapts them to the foundation encoder's native patch size, and resizes decoded RGB, point maps, and confidence maps back to the original resolution. `metadata.json` maps every image filename to a `3 x 3` intrinsic matrix and a `4 x 4` OpenCV world-to-camera extrinsic matrix. See the bundled examples for the exact schema.

### Feature Extraction CLI

```bash
python scripts/extract_features.py \
  --model dino2 \
  --scene examples/scene_000 \
  --frame-ids 0 1 2 3 4 5 6 7 8 9
```

Use the compact 256-token representation with:

```bash
python scripts/extract_features.py \
  --model dino2-256 \
  --scene examples/scene_000 \
  --frame-ids 0 1 2 3 4 5 6 7 8 9
```

Note that this command only prints the latent shape and device. You can add serialization code to this interface for your customized downstream 3D latent uses.

| Argument | Meaning |
|---|---|
| `--model {dino2,dino2-256,siglip2}` | Feature model; `dino2-256` selects the compact 256-token DINOv2 checkpoint. Default: `dino2`. |
| `--config PATH` | Optional model config override. |
| `--scene PATH` | Calibrated input scene. |
| `--frame-ids ID ...` | Frames used for feature extraction; defaults to all frames. |
| `--device DEVICE` | Torch device; default: `cuda`. |
| `--precision {bf16,fp32}` | Encoder inference precision; default: `bf16`. |

### Stage 1 Reconstruction and NVS CLI

Reconstruct the condition views by omitting `--target-frame-ids`:

```bash
python scripts/reconstruct.py \
  --model dino2 \
  --scene examples/scene_000 \
  --condition-frame-ids 0 2 4 6 8 \
  --output-dir outputs/reconstruction
```

Render novel target views by specifying separate target cameras:

```bash
python scripts/reconstruct.py \
  --model dino2 \
  --scene examples/scene_000 \
  --condition-frame-ids 0 2 4 6 8 \
  --target-frame-ids 1 3 5 7 9 \
  --output-dir outputs/nvs
```

The corresponding 256-token RGB-only reconstruction uses the same arguments:

```bash
python scripts/reconstruct.py \
  --model dino2-256 \
  --scene examples/scene_000 \
  --condition-frame-ids 0 2 4 6 8 \
  --target-frame-ids 1 3 5 7 9 \
  --output-dir outputs/nvs_256
```

The output contains condition RGB images, decoded RGB images, and metadata. The 1024-token DINOv2 model additionally exports point-map visualizations and a fused RGB-colored `point_cloud.ply` with per-point confidence values. The 256-token DINOv2 and SigLIP2 checkpoints are RGB-only. The script refuses to write into a non-empty output directory.

| Argument | Meaning |
|---|---|
| `--model {dino2,dino2-256,siglip2}` | Stage 1 model; `dino2-256` selects the compact RGB-only checkpoint. Default: `dino2`. |
| `--scene PATH` | Calibrated input scene. |
| `--condition-frame-ids ID ...` | Views encoded into the 3D latent. |
| `--target-frame-ids ID ...` | Cameras to decode; defaults to the condition views. |
| `--output-dir PATH` | Empty or new output directory. |
| `--device DEVICE` | Torch device; default: `cuda`. |
| `--encoder-precision {bf16,fp32}` | Feature encoder precision; default: `bf16`. |
| `--decoder-precision {bf16,fp32}` | Decoder precision; default: `fp32`. |
| `--confidence-percentile FLOAT` | Fraction of lowest-confidence points removed from the exported PLY; default: `0.2`. |
| `--viser` | Launch the interactive 3D viewer after export. |
| `--viser-port PORT` | Viser server port; default: `8080`. |
| `--viser-max-points N` | Maximum number of points sent to the browser; default: `500000`. |

### Stage 2 Generation CLI

In `frames` mode, appearance, trajectory, and decoder target views are controlled independently:

```bash
# 1-view appearance conditioning
python scripts/generate.py \
  --scene examples/scene_000 \
  --appearance-condition-frame-ids 0 \
  --trajectory-condition-frame-ids 0 1 2 3 4 5 \
  --target-frame-ids 6 7 8 9 \
  --output-dir outputs/generation_1view

# 2-view appearance conditioning
python scripts/generate.py \
  --scene examples/scene_000 \
  --appearance-condition-frame-ids 0 5 \
  --trajectory-condition-frame-ids 0 1 2 3 4 5 \
  --target-frame-ids 6 7 8 9 \
  --output-dir outputs/generation_2view

# 6-view appearance conditioning
python scripts/generate.py \
  --scene examples/scene_000 \
  --appearance-condition-frame-ids 0 1 2 3 4 5 \
  --trajectory-condition-frame-ids 0 1 2 3 4 5 \
  --target-frame-ids 6 7 8 9 \
  --output-dir outputs/generation_6view
```

Appearance views provide DINOv2 features. Trajectory views provide camera poses that define scene coverage. Target views are only decoded for visualization. If an appearance frame is absent from the trajectory list, the script emits a warning and appends it to the trajectory automatically.

For 1-view generation without ground-truth trajectory poses, create a smooth orbit from the first camera:

```bash
python scripts/generate.py \
  --scene examples/scene_000 \
  --appearance-condition-frame-ids 0 \
  --trajectory-mode synthetic-orbit \
  --synthetic-trajectory-views 6 \
  --synthetic-trajectory-baseline 1.5 \
  --synthetic-trajectory-arc-degrees 60 \
  --output-dir outputs/generation_1view_synthetic
```

Synthetic-orbit mode decodes every synthetic camera and exports all views to the fused PLY, including the condition view. It accepts exactly one appearance view and does not accept `--trajectory-condition-frame-ids` or `--target-frame-ids`.

| Argument | Meaning |
|---|---|
| `--scene PATH` | Calibrated input scene. |
| `--appearance-condition-frame-ids ID ...` | Exactly 1, 2, or 6 views providing appearance features. |
| `--trajectory-mode {frames,synthetic-orbit}` | Use input camera poses or synthesize an orbit; default: `frames`. |
| `--trajectory-condition-frame-ids ID ...` | Ordered trajectory views; required in `frames` mode. |
| `--synthetic-trajectory-views N` | Number of synthesized cameras; default: `6`. |
| `--synthetic-trajectory-baseline FLOAT` | Maximum baseline in normalized camera coordinates; default: `1.5`. |
| `--synthetic-trajectory-arc-degrees FLOAT` | Horizontal orbit arc; default: `60`. |
| `--target-frame-ids ID ...` | Decoder views; required in `frames` mode. |
| `--output-dir PATH` | Empty or new output directory. |
| `--device DEVICE` | Torch device; default: `cuda`. |
| `--precision {bf16,fp32}` | Feature encoder and 3DDiT precision; default: `bf16`. |
| `--decoder-precision {bf16,fp32}` | Stage 1 decoder precision; default: `fp32`. |
| `--guidance-scale FLOAT` | Classifier-free guidance scale, at least 1; default: `2.0`. |
| `--num-steps N` | Euler ODE sampling steps; default: `50`. |
| `--seed N` | Sampling seed; default: `0`. |
| `--confidence-percentile FLOAT` | Fraction of lowest-confidence points removed from the exported PLY; default: `0.2`. |
| `--viser` | Launch the interactive 3D viewer after export. |
| `--viser-port PORT` | Viser server port; default: `8080`. |
| `--viser-max-points N` | Maximum number of points sent to the browser; default: `500000`. |

### Interactive Viser Viewer

Append the Viser arguments to either reconstruction or generation:

```bash
python scripts/reconstruct.py \
  --model dino2 \
  --scene examples/scene_000 \
  --condition-frame-ids 0 2 4 6 8 \
  --target-frame-ids 1 3 5 7 9 \
  --output-dir outputs/nvs_viser \
  --viser \
  --viser-port 8080 \
  --viser-max-points 500000
```

Open `http://localhost:8080`. For a remote machine, forward the port first:

```bash
ssh -L 8080:localhost:8080 <user>@<host>
```

The viewer starts after file export and runs until `Ctrl+C`. Camera colors have fixed semantics:

- **Red:** appearance condition or Stage 1 condition.
- **Blue:** trajectory-only condition.
- **Green:** decoder-only target.

The GUI controls confidence filtering, per-view point visibility, point size, camera size, and camera/trajectory/point-cloud visibility. Click a camera frustum to move to that view. SigLIP2 has no point decoder, so its viewer shows cameras, trajectories, and image frustums only.

### ScanNet Preparation

ScanNet is included only as an example training dataset for demonstrating the complete training pipeline. To train with additional or custom datasets, follow the ScanNet preprocessing and dataloader implementation in `src/training/data.py`, then extend the current training code to combine the required datasets for multi-dataset training.

The ready-to-use processed data and split-TAR extraction commands are provided under [Prepare Data](#prepare-data). To reproduce the dataset from the original ScanNet export instead, first obtain ScanNet access and export each scene with `color/`, `pose/`, and `intrinsic/intrinsic_color.txt`. Then convert the official training and validation lists directly to the frame-based directory format:

```bash
python tools/prepare_scannet.py \
  --scans-root <scannet_root>/scans \
  --train-list <scannet_root>/Tasks/Benchmark/scannetv2_train.txt \
  --test-list <scannet_root>/Tasks/Benchmark/scannetv2_val.txt \
  --output data/scannet \
  --workers 8
```

`prepare_scannet.py` provides the following controls:

| Argument | Meaning |
|---|---|
| `--scans-root PATH` | Root containing exported ScanNet scene directories. |
| `--train-list PATH` | Text file containing training scene names. |
| `--test-list PATH` | Optional text file containing validation/test scene names. |
| `--output PATH` | Frame-based dataset root; default: `data/scannet`. |
| `--max-image-side N` | Resize the longest RGB side to at most this value; default: `560`. |
| `--frame-stride N` | Keep every N-th valid frame; default: `1`. |
| `--min-frames N` | Minimum valid frames required per scene; default: `18`. |
| `--png-compression N` | PNG compression level in `[0, 9]`; default: `1`. |
| `--workers N` | Parallel RGB decoding, resizing, and PNG writer threads; default: `8`. |
| `--max-scenes N` | Optional per-split scene limit for testing. |

### Stage 1 Training

Stage 1 trains the DINOv2 3D representation and RGB decoder for 120k steps. The discriminator starts warming up at step 50k, and the generator GAN loss starts at step 60k. Training defaults to eight GPUs:

```bash
bash scripts/train_stage1.sh \
  configs/training/dino2_stage1_scannet.yaml \
  outputs/stage1
```

Override the GPU count through `NPROC_PER_NODE`:

```bash
NPROC_PER_NODE=4 bash scripts/train_stage1.sh \
  configs/training/dino2_stage1_scannet.yaml \
  outputs/stage1
```

Resume from a full training checkpoint with optimizer, scheduler, scaler, and EMA state:

```bash
bash scripts/train_stage1.sh \
  configs/training/dino2_stage1_scannet.yaml \
  outputs/stage1 \
  --resume outputs/stage1/checkpoints/step-00060000.pt
```

Key Stage 1 overrides passed after the output directory are:

| Argument | Meaning |
|---|---|
| `--config PATH` | Training config; supplied as the shell wrapper's first positional argument. |
| `--output-dir PATH` | Run directory; supplied as the shell wrapper's second positional argument. |
| `--resume PATH` | Full checkpoint used to resume training. |
| `--data-path PATH` | Override `dataset.path`. |
| `--max-steps N` | Override total training steps. |
| `--num-workers N` | Dataloader workers per rank. |
| `--max-tokens-per-gpu N` | Token budget per GPU. |
| `--max-batch-size N` | Maximum local batch size. |
| `--discriminator-start-step N` | First discriminator update step. |
| `--gan-start-step N` | First step using generator GAN loss. |
| `--vis-interval N` | Visualization interval; `0` disables it. |
| `--precision {fp32,fp16,bf16}` | Training precision; default: `bf16`. |

Export the Stage 1 EMA neck and decoder as standalone `safetensors` files for Stage 2 uses:

```bash
python tools/export_weights.py \
  outputs/stage1/checkpoints/step-00120000.pt \
  outputs/stage1/neck.safetensors \
  --state-key ema \
  --prefix neck.

python tools/export_weights.py \
  outputs/stage1/checkpoints/step-00120000.pt \
  outputs/stage1/decoder.safetensors \
  --state-key ema \
  --prefix decoder.
```

### Calculate Latent Statistics

Stage 2 requires population mean and variance computed from the trained Stage 1 latent space:

```bash
bash scripts/calculate_stat.sh \
  configs/training/dino2_stat_scannet.yaml \
  outputs/stage1/stats.safetensors
```

By default, the script uses all batches in the ScanNet training split and aggregates exact `sum`, `sum_sq`, and `count` values across all eight ranks. Useful overrides are:

| Argument | Meaning |
|---|---|
| `--config PATH` | Statistics config; supplied as the shell wrapper's first positional argument. |
| `--output PATH` | Output `safetensors`; supplied as the shell wrapper's second positional argument. |
| `--neck-weights PATH` | Override the exported Stage 1 neck weights. |
| `--data-path PATH` | Override `dataset.path`. |
| `--num-workers N` | Dataloader workers per rank. |
| `--max-batches N` | Optional batch limit for a smoke test; omitted means all batches. |
| `--max-tokens-per-gpu N` | Token budget per GPU. |
| `--max-batch-size N` | Maximum local batch size. |
| `--seed N` | Dataset and sampling seed. |
| `--precision {fp32,bf16}` | Statistics precision; default: `bf16`. |

### Stage 2 Training

The default Stage 2 config reads the exported Stage 1 neck, decoder, and statistics from `outputs/stage1/` and trains DINOv2 3DDiT for 160k steps on eight GPUs:

```bash
bash scripts/train_stage2.sh \
  configs/training/dino2_stage2_scannet.yaml \
  outputs/stage2
```

Resume training with:

```bash
bash scripts/train_stage2.sh \
  configs/training/dino2_stage2_scannet.yaml \
  outputs/stage2 \
  --resume outputs/stage2/checkpoints/step-00080000.pt
```

Key Stage 2 overrides passed after the output directory are:

| Argument | Meaning |
|---|---|
| `--config PATH` | Training config; supplied as the shell wrapper's first positional argument. |
| `--output-dir PATH` | Run directory; supplied as the shell wrapper's second positional argument. |
| `--resume PATH` | Full checkpoint used to resume training; mutually exclusive with `--init-weights`. |
| `--init-weights PATH` | Initialize only model weights without optimizer state. |
| `--data-path PATH` | Override `dataset.path`. |
| `--neck-weights PATH` | Override the Stage 1 neck weights. |
| `--decoder-weights PATH` | Override the Stage 1 decoder weights used for visualization. |
| `--normalization-stats PATH` | Override latent mean/variance. |
| `--max-steps N` | Override total training steps. |
| `--num-workers N` | Dataloader workers per rank. |
| `--max-tokens-per-gpu N` | Token budget per GPU. |
| `--max-batch-size N` | Maximum local batch size. |
| `--gradient-accumulation-steps N` | Gradient accumulation steps. |
| `--vis-interval N` | Visualization interval; `0` disables it. |
| `--vis-num-steps N` | Euler ODE steps used for training-time visualization. |
| `--precision {fp32,fp16,bf16}` | Training precision; default: `bf16`. |

Export the Stage 2 EMA model as pure inference weights after training:

```bash
python tools/export_weights.py \
  outputs/stage2/checkpoints/step-00160000.pt \
  outputs/stage2/3ddit.safetensors \
  --state-key ema
```

Stage 1 and Stage 2 visualizations are written to `outputs/<stage>/visualizations/step-XXXXXXXX.png`.

## Acknowledgements

This project builds on ideas and code from [RAE](https://github.com/bytetriper/RAE), [LVSM](https://github.com/haian-jin/LVSM), [Stable Virtual Camera](https://github.com/Stability-AI/stable-virtual-camera), [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3), [VGGT](https://github.com/facebookresearch/vggt), [DINOv2](https://github.com/facebookresearch/dinov2), [SigLIP 2](https://huggingface.co/google/siglip2-base-patch16-256), and [Viser](https://github.com/nerfstudio-project/viser). We thank the authors for making their work publicly available.

The bundled examples are derived from [ScanNet](https://github.com/ScanNet/ScanNet) and [RealEstate10K](https://google.github.io/realestate10k/). Users are responsible for complying with the corresponding dataset terms.

## Citation

If you use 3DRAE in your research, please cite:

```bibtex
@inproceedings{wei20263drae,
  title     = {Any 3D Scene is Worth 1K Tokens: 3D-Grounded Representation for Scene Generation at Scale},
  author    = {Wei, Dongxu and Xu, Qi and Li, Zhiqi and Zhou, Hangning and Qiu, Cong and Qin, Hailong and Yang, Mu and Cui, Zhaopeng and Liu, Peidong},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

This project is released under the [MIT License](LICENSE).
