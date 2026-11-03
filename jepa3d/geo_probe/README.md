# Geometry probe: can object positions be read from frozen 3DRAE tokens?

Status: done (jobs 12360 real, 12361 shuffled). Verdict: coarse information only,
below the pass bar, so no Llama geometry supervision yet.

| Val, 229 scenes, visible objects | real tokens | shuffled-trained |
|---|---|---|
| quadrant acc (chance 25%) | 45.6% [43.3, 48.9] | 33.3% [32.0, 34.5] |
| left / right / back acc | 54.5% | 42.5% |
| azimuth error, median | 42.8 deg | 65.7 deg |
| position error, median | 0.95 m | 1.85 m |
| pair distance error, median | 0.64 m | 1.08 m |

Bar was quadrant >= 50% and >= 15 points over shuffled; got 45.6% and +12.3.
Real run: train loss 0.15 vs 1.60 for shuffled, val flat after epoch 60, so the
probe overfits 913 scenes.

- Labels: sampled projection sheets put centres on the named objects. "Visible" is
  loose: only 30% of a frame's visible objects have their centre inside that frame.
  Some category names are noisy (a "blackboard" on a desk panel in scene0264_01).
- Overfit, 8 scenes, 1000 steps: visible objects reach 0.00 m position error and 100%
  quadrant accuracy with real tokens, 3.6 m and 61% with tokens swapped at eval.

Blind check on the 5-epoch VL-JEPA run: VSI-Bench accuracy is the same with real,
shuffled and zero visual tokens. That cannot separate "the tokens hold no object
geometry" from "Llama / InfoNCE does not use it". This probe skips Llama and
InfoNCE and asks only the first question. It is a diagnostic, not part of the
final model.

## Data

- Train: EmbodiedScan train, 958 ScanNet scenes. Val: EmbodiedScan val, 243 scenes.
- Views: 8 frames, 224x308, same quality filter and pose FPS as the VL-JEPA data,
  restricted to frames listed in EmbodiedScan (about 160 per train scene, 143 per
  val scene), since visibility is only known for those. Train draws a new FPS seed per sample;
  val uses 3 fixed seeds per scene.
- Objects: category occurs once in the scene and is not in
  `labels.EXCLUDED` (structure and catch-all classes). Train: visible in at
  least one selected frame. Val: reported for visible and for all objects.

## Ground truth

Box centres are in the axis-aligned frame, so in the first camera's frame:

    p = inv(axis_align @ pose_0) @ center
    Delta_AB = p_B - p_A

Units are metres. Pairs closer than 0.1 m are not used. Gravity for metrics is the
raw world z axis, expressed in the first camera's frame. Distances are between
centres, so they are not VSI's closest-point distances.

## Model and loss

    category name -> embeddinggemma (frozen) -> Linear(512) -> query
    8 views -> 3DRAE (frozen) -> scene + view tokens -> Linear(512) -> memory
    query, memory -> 2 cross-attention layers (8 heads) -> MLP -> p_norm
    p_hat = s * p_norm     # s = input pose normalization scale, metres

Objects are queried independently and never see each other. The view tokens'
pose / feature projections train with the probe.

    L = SmoothL1(p_hat, p)
      + mean[1 - cos(Delta_hat_AB, Delta_AB)]
      + mean[SmoothL1(log(1 + |Delta_hat_AB|), log(1 + |Delta_AB|))]

over all ordered pairs of a scene.

## Runs

| Stage | What | Gate |
|---|---|---|
| `labels` | build labels, project box centres onto frames | look at the sheets before training |
| `overfit` | 8 train scenes, fixed views | position error must approach 0 |
| `train` | real tokens | main result |
| `train` with `TOKENS=shuffled` | tokens from another scene | category-prior baseline |

Final evaluation of each run also repeats with shuffled tokens at eval time.
Reported at the last epoch, never selected on val.

    STAGE=labels  sbatch jepa3d/geo_probe/geo_probe.sbatch
    STAGE=overfit sbatch jepa3d/geo_probe/geo_probe.sbatch
    STAGE=train TOKENS=real     sbatch jepa3d/geo_probe/geo_probe.sbatch
    STAGE=train TOKENS=shuffled sbatch jepa3d/geo_probe/geo_probe.sbatch

## Metrics and decision

On the val scenes, with scene-bootstrap 95% intervals: position error, centre
distance error, horizontal azimuth error, 4-quadrant accuracy (chance 25%) and
left / right / back accuracy, for "stand at A, face C, where is B".

Fixed before running:

| Result | Reading | Next |
|---|---|---|
| quadrant >= 50% and >= 15 points above shuffled | tokens carry usable position | add geometry supervision to Llama |
| clearly above shuffled, below that bar | coarse information only | improve token use or the encoder first |
| no clear gap to shuffled | position not readable | go back to 3DRAE, skip Llama supervision |

The third row only counts if the overfit stage passed.

## Control: DINOv2 patches without the neck

`MEMORY=dino` (`dino_memory.py`): the probe reads the frozen DINOv2 patch features
that go into the 3DRAE neck (8 views x 16 x 22 = 2816 tokens) plus each patch's
Plucker ray (trainable linear layer) and a view index embedding. Probe, labels,
loss and schedule are unchanged. Clearly better than 45.6% quadrant: the neck
loses position; about the same: the limit is earlier (DINOv2) or in the probe.

    STAGE=train MEMORY=dino TOKENS=real sbatch jepa3d/geo_probe/geo_probe.sbatch

Result (job 12364, val visible objects): DINOv2 patches 51.3% quadrant [48.6, 54.3]
vs 45.6% for 3DRAE tokens and 33.3% shuffled; azimuth error 34.5 deg vs 42.8 deg;
position error 0.94 m vs 0.95 m. Better direction, same position; one seed, CIs not paired.
