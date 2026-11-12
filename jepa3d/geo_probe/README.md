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

## Error breakdown (job 12365, `analyze.py`)

Val objects are 1.97 m (median) from the first camera; an oracle that puts every
object at the sample's true centroid errs by 1.53 m. Probe error (3DRAE / DINOv2):

| | 3DRAE | DINOv2 |
|---|---|---|
| centre in no view / in 1 / in 2-3 views | 1.21 / 0.95 / 0.92 m | 1.13 / 0.96 / 0.94 m |
| object 0-2 / 2-3 / 3-4 / 4-6 m from first camera | 0.74 / 1.11 / 1.61 / 2.30 m | 0.73 / 1.13 / 1.57 / 2.41 m |
| category seen 0 / 1-9 / 10-49 / 50-199 / 200+ times in train (centre in a view) | 2.78 / 1.57 / 1.32 / 0.90 / 0.65 m | 2.65 / 1.51 / 1.26 / 0.94 / 0.72 m |
| azimuth error, shorter arm 0.1-0.5 / 1-2 / 3+ m | 63 / 40 / 26 deg | 60 / 32 / 19 deg |

Error follows category frequency far more than visibility (toilet 0.25 m, sink
0.32 m; socket 1.75 m). With 8,838 train objects over 195 categories, the probe
mostly has to learn which tokens are "the printer"; this run measures category
grounding and geometry together, not geometry alone.

## Experiment 2: is the error grounding or geometry?

The error breakdown fits "the probe cannot find the object from its name" but does
not prove it (frequent categories are also large, distinctive furniture). Two tests:

- Point prompt (`QUERY=point`): the query is the DINOv2 feature of the patch holding
  the object's projected centre in one view (random view in training, the view where
  it is nearest the image centre in val) plus that pixel's Plucker ray. No category
  name, so no recognition; only objects whose centre is in a view. Same query for
  every memory: 3DRAE tokens, DINOv2 patches, and shuffled 3DRAE tokens (floor: one
  patch and its ray alone). If errors drop well below 1 m and stop following category
  frequency, recognition was the limit.
- Attention (`analyze.py --attention` on `dino_real`): share of each cross-attention
  layer on the 3x3 / 7x7 patches around the object's projected centre, against
  uniform, by category frequency and against the position error.

    STAGE=train QUERY=point MEMORY=3drae TOKENS=real     sbatch jepa3d/geo_probe/geo_probe.sbatch
    STAGE=train QUERY=point MEMORY=dino  TOKENS=real     sbatch jepa3d/geo_probe/geo_probe.sbatch
    STAGE=train QUERY=point MEMORY=3drae TOKENS=shuffled sbatch jepa3d/geo_probe/geo_probe.sbatch
    STAGE=analyze RUNS=dino_real ATTENTION_RUNS=dino_real sbatch jepa3d/geo_probe/geo_probe.sbatch

### Results (jobs 12368-12372, val, objects whose centre is in a view)

Attention, category query on DINOv2 patches (12371): layer-1 share on the 7x7 patches
around the centre is 1.5x uniform for unseen categories, 2.1x for 1-9 train objects,
6.7-8.2x for 50+; position error by layer-1 attention tertile 1.41 / 0.91 / 0.76 m.

Point prompt:

| | 3DRAE tokens | DINOv2 patches | shuffled 3DRAE (floor) | category query, 3DRAE |
|---|---|---|---|---|
| position error, median | 0.27 m [0.25, 0.30] | 0.52 m [0.48, 0.56] | 0.68 m | 0.95 m |
| pair distance error, median | 0.21 m | 0.42 m | 0.55 m | 0.64 m |
| azimuth error, median | 8.0 deg | 10.8 deg | 15.5 deg | 42.8 deg |
| quadrant acc | 83.3% [81.9, 84.8] | 78.9% | 71.5% | 45.6% |
| same model, tokens shuffled at eval | 1.17 m, 66.8% | 1.15 m, 65.1% | - | - |
| error by train frequency 0 / 1-9 / 10-49 / 50-199 / 200+ | 0.42 / 0.35 / 0.29 / 0.27 / 0.27 m | 0.71 / 0.67 / 0.52 / 0.51 / 0.51 m | | 2.78 / 1.57 / 1.32 / 0.90 / 0.65 m |
| error, object 0-2 / 2-3 / 3-4 / 4-6 m from first camera | 0.22 / 0.28 / 0.40 / 0.60 m | 0.41 / 0.58 / 0.71 / 1.11 m | | |

The category-query column is from experiment 1 (visible objects) and covers a
slightly different object set. With the object located for it, the probe reads
positions from 3DRAE tokens to 0.27 m, about 13% of the distance from the first
camera, and the frequency dependence is mostly gone. The 3DRAE neck does not lose
geometry: its tokens beat raw DINOv2 patches by about 2x. Experiment 1 was limited
by finding the object from its name, not by geometry.

## Experiment 3: the same probe on VGGT tokens

`MEMORY=vggt` (`vggt_memory.py`): frozen VGGT-1B aggregator, last-round patch tokens
(2048-d), plus the same Plucker rays and view embedding as the DINOv2 control. Input
224x308 (2816 tokens) or `MEMORY_SIZE="378 518"` (7992 tokens). Report with tables,
plots and conclusions: `vggt_probe_report.ipynb` (`STAGE=report` re-executes it into
`outputs/geo_probe`).

Results (jobs 12545-12548, analysis 12554):

| | 3DRAE | VGGT 224x308 | VGGT 378x518 | DINOv2 patches |
|---|---|---|---|---|
| point prompt: position / azimuth / quadrant | 0.27 m / 8.0 deg / 83.3% | 0.29 m / 7.8 deg / 83.2% | 0.26 m / 7.4 deg / 83.3% | 0.52 m / 10.8 deg / 78.9% |
| category query: position / azimuth / quadrant | 0.95 m / 42.8 deg / 45.6% | 1.45 m / 56.9 deg / 37.2% | 1.46 m / 57.3 deg / 37.1% | 0.94 m / 34.5 deg / 51.3% |
| category query: final train loss | 0.15 | 0.49 | 0.45 | 0.09 |

With the object located, VGGT and 3DRAE tokens give the same accuracy. From a category
name the probe does much worse on VGGT tokens: its layer-1 attention stays near uniform
(1.0-1.4x on the 7x7 patches around the object, 2.7x only for 200+ train objects; DINOv2
5-8x) and it does not fit the train set either. One seed; last VGGT round only.

## Experiment 4: reverse probe, category from a point

`train_reverse.py` (`STAGE=reverse`): classify the category (195 train categories,
cross-entropy) of each object whose centre is in a view. `READOUT=attention`: the same
cross-attention probe, query = Fourier features of the centre pixel's Plucker ray only,
so the category must come from the tokens. `READOUT=index` (DINOv2 / VGGT): MLP on the
one patch token holding the centre. Report: `reverse_probe_report.ipynb`
(`STAGE=report NOTEBOOK=reverse_probe_report`).

Results (jobs 12620-12625, val, 4920 objects; always guessing "door" gives 4.5%):

| | top-1 | top-5 | macro top-1 | train top-1 |
|---|---|---|---|---|
| 3DRAE, attention | 31.8% [29.1, 34.6] | 58.8% | 16.1% | 86% |
| DINOv2 patches, attention | 34.5% [31.8, 37.2] | 61.5% | 18.6% | 88% |
| VGGT, attention | 19.3% | 44.2% | 7.8% | 71% |
| DINOv2 patches, index | 56.4% | 78.6% | 40.7% | 87% |
| VGGT, index | 40.2% | 64.8% | 19.2% | 63% |
| shuffled 3DRAE (floor) | 7.7% | 23.2% | 2.0% | 10% |

3DRAE tokens hold as much location-indexed category information as DINOv2 patches under
the same readout, far above the floor, but only for frequent categories (top-1 50% for
200+ train objects, 10% for 10-49, about 0 below 10). All probes overfit heavily (train
86-88%, val 32-35%): 913 scenes are too few to learn the name-to-feature mapping. Attention
by ray loses information against indexing the patch (34.5% vs 56.4% on DINOv2). VGGT
tokens hold clearly less category information.
