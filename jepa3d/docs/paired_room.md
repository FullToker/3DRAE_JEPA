# Does seeing the correct room improve answers about spatial relationships?

Replay the slot-free `hardneg_v1` final checkpoint from job 12610 on each original question with its own eight views and with eight views of a different physical room. The third prediction comes from one model trained on the same examples with access only to the question. Each model scores the original long-form options with its own answer encoder.

## Run on the research server

Use this branch in the 3DRAE checkout and the original job's `fusion_gv` checkout in `VJEPA_ROOT`. Set `HARDNEG_RAE_REV` and `HARDNEG_VJEPA_REV` to the 3DRAE_JEPA commit and full sibling commit from the job archive. The hard-negative report was published at `8c6b50d`. The public sibling revision exercised in the CPU integration test was `10c8aa48b08d76673a3be84e24be8e7744cc2d50`. The historical run used three GPUs for about 4 h 17 min. Allow additional time for hashing and inference and space for the new final checkpoint (the original was about 11 GB).

Recover the original `experiments/hardneg_v1/ckpt/{config.yaml,final.pt,run_meta.json}`, `results/eval_final.json` and the two manifests named in that config. The frame paths, cached Llama and EmbeddingGemma, Open-VL-JEPA warm start, DINO backbone, neck and statistics must resolve on the worker. Preserve the original logs alongside the checkpoint. A separate checkout can point to these assets using the original paths or symlinks.

Supply verified ScanNet++ scan-to-room correspondences as a JSON object if available. For example, two scans of the same physical space must have the same value. Use keys of the form `scannetpp/<scan-id>` and cover training as well as holdout scans. ScanNet rescan suffixes are resolved automatically. Without complete training-room coverage for a subset, affected rows are labeled unresolved. The original real-scene replay still includes them.

Run all five stages from the repository root in the original environment on a three-GPU allocation.

```bash
python -m jepa3d.tools.paired_room run \
  --out experiments/paired_room/run_01 \
  --baseline-revision "$HARDNEG_RAE_REV" \
  --sibling-revision "$HARDNEG_VJEPA_REV" \
  --aliases "$ROOM_ALIASES"
```

Omit `--aliases` if the mapping is unavailable. For SLURM, export the revision variables and optionally `ROOM_ALIASES`. The launcher uses the existing container and storage defaults.

```bash
sbatch --export=ALL jepa3d/trainbatch/paired_room.sbatch
```

For a staged run, replace `run` with `prepare`. Inspect `plan.json`, including `rejected`, each row's split and donor, and `review_ids`. For those IDs check the actual views for object visibility, identity and the direction reference axis. Record observations alongside the run. Run the remaining stages with the same output directory.

```bash
python -m jepa3d.tools.paired_room baseline --out experiments/paired_room/run_01
python -m jepa3d.tools.paired_room train    --out experiments/paired_room/run_01
python -m jepa3d.tools.paired_room control  --out experiments/paired_room/run_01
python -m jepa3d.tools.paired_room report   --out experiments/paired_room/run_01
```

`prepare` also accepts `--config`, `--checkpoint` and `--reference` for relocated job archives. Enrollment hashes the original manifests, assets and ordered image content. The data split and answer order come from the original run. Room pairs and triples span all question types, so a physical room belongs to one uncertainty block. Donors match source, subset, view count, pose provenance and selection protocol. Repeated RGB content across a candidate pair disqualifies that assignment.

## Verification and outputs

- `baseline.json` contains per-question cosine scores in native evaluation order, winning answer, gold index, margin and visual source ID. Donor IDs point to the complete locked RGB and camera bundle in `plan.json`. The original 6,786 real-scene predictions must reproduce every historical question-type correct count before the control starts. A mismatch produces `replay_failed.json`. Check the recorded counts, revisions, input hashes, batch size and precision before attempting a fresh enrollment.
- `control.yaml` preserves the historical warm start, random seed, data order, hard negatives, distributed loss, batch size, optimizer and update schedule. The unchanged encoder construction preserves initialization order. A fixed zero prefix of 1,032 × 768 replaces the complete visual output before projection. The training loader bypasses image decoding. The final state includes a control-specific marker checked by strict loading. `trained.json` binds the control checkpoint to the successful baseline. The original trainer starts from step zero for a fresh training run.
- `question.json` uses the trained control's own answer encoder. Both evaluations check the prefix, candidate scores and option permutation and record singleton versus batch differences. Each artifact identifies its enrollment. Generated files stay in the run directory.
- `metrics.json` contains counts for attempted, eligible, scored and common IDs, including the full paired holdout, the four relation families, and the verified unseen-room primary panel. Each family includes label confusion and recall where labels can be recovered, answer flips in both directions and margin changes. The two primary effects are correct-room minus donor-room and correct-room minus question-only accuracy, averaged equally across families. The intervals are paired two-sided 97.5% intervals from 10,000 joint resamples of donor blocks. Inspect the number of physical rooms and supported families alongside the intervals.

The prespecified success criterion is at least five percentage points over **each** comparison, both lower bounds above zero and a positive room effect in at least three of four families. An upper bound below five points excludes the stipulated advantage for that comparison. Missing families, insufficient independent room blocks or intervals spanning the decision boundary leave the result inconclusive.

If a stage is interrupted, keep completed artifacts and rerun the first unfinished stage. The training stage recognizes a completed final checkpoint. If training stopped before writing it, archive the partial `control_ckpt/` directory before restarting training from step zero. A failed baseline replay needs reconciliation before the control can start.

## CPU tests

Use the QA environment with the sibling source available. The integration fixture uses the real sibling predictor, target scorer, hard-negative trainer and checkpoint format, tiny offline text encoders, and generated frame bundles.

```bash
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
```
