# Examples

This directory contains six compact calibrated multi-view scenes and one 256-frame large-scale scene. The compact scenes provide six candidate condition views and four held-out target views. The DL3DV scene provides 128 interleaved condition views and 128 target views.

| Path | Source | Resolution | Condition IDs | Target IDs |
|---|---|---:|---|---|
| `scene_000/` | ScanNet | 308×224 | `0 1 2 3 4 5` | `6 7 8 9` |
| `scene_001/` | ScanNet | 308×224 | `0 1 2 3 4 5` | `6 7 8 9` |
| `scene_002/` | ScanNet | 308×224 | `0 1 2 3 4 5` | `6 7 8 9` |
| `scene_003/` | RealEstate10K | 364×196 | `0 1 2 3 4 5` | `6 7 8 9` |
| `scene_004/` | RealEstate10K | 364×196 | `0 1 2 3 4 5` | `6 7 8 9` |
| `scene_005/` | RealEstate10K | 364×196 | `0 1 2 3 4 5` | `6 7 8 9` |
| `scene_006/` | DL3DV | 350×196 | even IDs `0–254` | odd IDs `1–255` |

`metadata.json` stores OpenCV world-to-camera extrinsics. Original sequence/frame mappings, preprocessing scales, and per-file SHA256 hashes are recorded in `manifest.json`.

## Feature Extraction

```bash
python scripts/extract_features.py \
  --model dino2 \
  --scene examples/scene_000 \
  --frame-ids 0 5

python scripts/extract_features.py \
  --model siglip2 \
  --scene examples/scene_003 \
  --frame-ids 0 5
```

## Reconstruction and NVS

```bash
python scripts/reconstruct.py \
  --model dino2 \
  --scene examples/scene_000 \
  --condition-frame-ids 0 5 \
  --target-frame-ids 6 7 8 9 \
  --output-dir outputs/example_nvs_dino2

python scripts/reconstruct.py \
  --model siglip2 \
  --scene examples/scene_003 \
  --condition-frame-ids 0 5 \
  --target-frame-ids 6 7 8 9 \
  --output-dir outputs/example_nvs_siglip2
```

Omit `--target-frame-ids` to reconstruct the condition cameras.

For the 256-frame large-scale scene, shell brace expansion provides the 128 interleaved condition and target IDs:

```bash
python scripts/reconstruct.py \
  --model dino2 \
  --scene examples/scene_006 \
  --condition-frame-ids {0..254..2} \
  --target-frame-ids {1..255..2} \
  --output-dir outputs/dl3dv_large_scale
```

## 1/2/6-View Conditioned Generation

```bash
# 1-view
python scripts/generate.py \
  --scene examples/scene_000 \
  --appearance-condition-frame-ids 0 \
  --trajectory-condition-frame-ids 0 1 2 3 4 5 \
  --target-frame-ids 6 7 8 9 \
  --output-dir outputs/example_generation_1view

# 2-view
python scripts/generate.py \
  --scene examples/scene_000 \
  --appearance-condition-frame-ids 0 5 \
  --trajectory-condition-frame-ids 0 1 2 3 4 5 \
  --target-frame-ids 6 7 8 9 \
  --output-dir outputs/example_generation_2view

# 6-view
python scripts/generate.py \
  --scene examples/scene_000 \
  --appearance-condition-frame-ids 0 1 2 3 4 5 \
  --trajectory-condition-frame-ids 0 1 2 3 4 5 \
  --target-frame-ids 6 7 8 9 \
  --output-dir outputs/example_generation_6view
```

For 1-view generation without ground-truth trajectory poses:

```bash
python scripts/generate.py \
  --scene examples/scene_000 \
  --appearance-condition-frame-ids 0 \
  --trajectory-mode synthetic-orbit \
  --output-dir outputs/example_generation_1view_synthetic
```

Append `--viser --viser-port 8080` to a reconstruction, NVS, or generation command to inspect its cameras, trajectories, and point cloud interactively.

## Provenance

The example images are derived from [ScanNet](https://github.com/ScanNet/ScanNet), [RealEstate10K](https://google.github.io/realestate10k/), and [DL3DV](https://dl3dv-10k.github.io/DL3DV-10K/). Users are responsible for following the corresponding dataset terms. The RealEstate10K camera trajectory data is provided by Google LLC under CC BY 4.0.
