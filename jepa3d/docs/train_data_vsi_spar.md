# Training data v1: VSI-590K + SPAR-7M (posed multi-view)

Status: design, numbers from read-only scans on ssh_bgd (2026-10-28). Nothing sampled yet.
Goal: real multi-view rows (8 / 16 / 24 distinct posed frames), quality over row count.

## 1. Sources

All images, poses and depth already sit under `vjepa201/source_data/`; only
`vsi590k/vsi_590k.jsonl` (277 MB, QA text) was downloaded. No videos.

| Part | QA rows | Scenes | Frame pool (image + c2w + K + depth) |
|---|---|---|---|
| VSI-590K / ScanNet | 92,145 | 1,201 (ScanNet train split) | `scannet/posed_images/<scene>/NNNNN.{jpg,txt,png}` + `intrinsic.txt`; every 10th raw frame, ~210 / scene |
| VSI-590K / ScanNet++ v2 | 138,311 (of 138,701; 6 scenes have no poses) | 850 | `spar/scannetpp/images/<scene>/{image,video}_{color,depth,pose}`; ~71 + 32 frames / scene, file names = raw frame ids |
| SPAR-7M / ScanNet + ScanNet++, 5 types (§3) | ~481k train / ~296k val (estimate) | 2,007 train | same scene pools as the two rows above |

VSI-590K other parts (ARKitScenes, ADT, S3DIS, Hypersim, ProcTHOR, web, robotics) are not used:
no poses on the server, or single images.

## 2. VSI-590K processing

- Question types (ScanNet + ScanNet++): relative_direction 85k, relative_distance 47k,
  absolute_distance 23k, absolute_direction 20k, relative_size 20k, appearance_order 14k,
  absolute_size 9.6k, absolute_count 6.4k, relative_count 3.8k, room_size 2k.
  All are scene-level (no frame references), so any posed subset of the scene is valid input.
- Query: drop `<image>`, "These are frames of a video.", and answer-format hints
  ("Answer with the option's letter...", "single word or phrase"). Keep the option list text.
- Target:
  - multiple choice: letter -> option text (`D` -> `back-left`), no letter prefix
    (matches the SPAR train format, see `vjepa201/notebooks/eval_format_check.ipynb`).
  - numeric: add the unit parsed from the question (meters / feet / square feet / degrees);
    counts stay bare numbers.
  - keep the original answer as `answer_raw`.
- appearance_order needs temporal order: selected frames are re-sorted by raw frame id.

## 3. SPAR-7M filtering (strict)

Only types whose rows carry 32 distinct frames, only the `sentence` kind, and only types
whose question is answerable from plain frames (no drawn points / boxes).

| Type | train rows (sentence) | Kept | Rule |
|---|---|---|---|
| obj_count | 149,560 | all | answers only 33.5% unique -> answer-key masking (§6) |
| room_size | 37,780 | all | answers 16.6% unique -> answer-key masking |
| appearance_order | 73,537 | all | target cut to the order clause; the "X at 14" frame numbers refer to the original 32 frames |
| spatial_imagination_oc_video_hard | 148,779 | ~146k | drop rows whose referenced objects share a noun (1.8%); fix " ." spacing (20.3%) |
| spatial_imagination_oo_video_hard | 147,976 | ~74k | drop broken templates with an empty slot ("to the left and  sink", 47.8%) and shared nouns (3.5%) |

Excluded:

| Type | Why |
|---|---|
| distance_infer_center_oo_video, distance_prediction_oo_video | question points at "(in Frame-27, point0)" / "red and blue points"; markers are not drawn on the images (the old `convert_spar_sftqa.py` never drew them either), so the target is not grounded. Candidate for v2 with markers drawn on the sampled frames. |
| spatial_imagination_oc_video, spatial_imagination_oo_video | same marker dependence (red / green / blue bbox); also 23% / 36% shared-noun ambiguity and 51% broken templates (oo) |
| obj_frame_locate | answer is a list of original frame indices, little spatial semantics |
| all 1-3 image types | not multi-view |

Splits: SPAR `train` for training; SPAR `val` filtered with the same rules and kept as-is.
Caveat: SPAR's ScanNet++ train and val use the same 803 scenes (ScanNet's are disjoint),
so SPAR val is not scene-held-out.

Leakage vs VSI-Bench: VSI-590K shares 0 scenes with VSI-Bench; SPAR train shares 1 ScanNet
scene -> dropped. VSI-Bench ScanNet (88 scenes) and ScanNet++ (40 of 50 with SPAR poses)
stay usable as a clean eval.

## 4. Frame sampling (N = 8, later 16 / 24)

Same sampler for both sources, per row:

1. Frame pool: every posed frame of the scene, same pool for VSI and SPAR rows (the five
   SPAR types are scene-level, so they are not tied to SPAR's fixed 32 frames):
   ScanNet `posed_images` (~210 / scene) and ScanNet++ SPAR `image_color` + `video_color`
   (~100 / scene; `video_depth` is named `frame_<id>.png`). ScanNet `posed_images` also
   holds `-annotated.jpg`, `-label.npy`, `-overlap.npz`; only `NNNNN.jpg` is taken.
2. Quality filter (before FPS): drop frames with a non-finite pose, too little valid depth
   (fraction of pixels in 0.3-8 m), or a close-up view (low median depth, e.g. facing a wall).
   Thresholds set from the per-frame distribution; `vggt_scannet_pose_fps_ablation.ipynb`
   showed plain pose-FPS picks exactly these fringe frames (valid_frac ~0).
3. Pose FPS on the remaining frames: distance = camera-center distance (m)
   + w * rotation angle / 180 deg, w = 1 (same metric as that notebook). The seed frame is
   chosen from a hash of the row id: reproducible, but different rows of one scene get
   different subsets.
4. Sort selected frames by raw frame id.
5. Fewer than N frames after filtering -> drop the row (counted in stats).

SPAR's 5 kept types have no frame references, so no forced frames and no `Frame-k`
re-indexing are needed.

## 5. Output

```
3DRAE_JEPA/data/vsi590k_sn_f{N}.jsonl
3DRAE_JEPA/data/spar_video_{train,val}_f{N}.jsonl
3DRAE_JEPA/data/mix_vsi_spar_train_f{N}.jsonl  (+ .stats.json)
```

Row fields (same as `PosedSparDataset` plus metadata): `images`, `intrinsics`, `c2w`,
`query`, `target`, `answer_raw`, `answer_key`, `group_id` (scene), `source`, `subset`,
`scene`, `question_type`, `src_line`.
Mix = concat of VSI + SPAR train, shuffled with a fixed seed; `--max-per-type` available
if the stats show one type dominating. Expected f8 mix: ~230k VSI + ~481k SPAR ≈ 0.7M rows.

## 6. Same answer as negative (InfoNCE false negatives)

Handled in both places, each doing the part it can:

- Data side (here): every row gets `answer_key`, a canonical form of the answer that is
  equal whenever two answers mean the same thing even if the wording differs:
  - VSI: normalized target text (option text, or number + unit).
  - SPAR obj_count: `count:<n>:<category>`; room_size: `area:<rounded m²>`;
    appearance_order: `order:<obj1>|<obj2>|...`; spatial_imagination_*_hard:
    normalized sentence after template stripping (the parser is checked in the sample notebook).
- Training side: `bidirectional_infonce` already masks off-diagonal pairs that share a
  `group_id`. jepa3d will mask pairs whose `answer_key` matches (neither positive nor
  negative), via its own loss wrapper; vjepa201 stays untouched. Whether same-scene pairs
  stay masked is a separate switch: they are hard negatives with different answers,
  so default is to keep them as negatives and mask only same-answer pairs.

The data side alone cannot fix it, since which rows meet in a batch is only known at
training time; the training side alone sees only text, so paraphrased duplicates in SPAR
templates ("2 monitor are in the room" / "Observations ... indicate 2 monitor") would slip
through without `answer_key`.

## 7. Next

1. Write `jepa3d/tools/sample_vsi590k.py`, `sample_spar_video.py`, `mix_manifests.py`;
   commit locally, pull on the server, run f8 via sbatch (CPU only).
2. Sample notebook: SPAR train / val and VSI rows with their 8 frames, query, target,
   answer_key, camera positions; review before 16 / 24 frames and before training.
