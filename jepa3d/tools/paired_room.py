"""Run the paired physical-room experiment for hardneg_v1 job 12610."""

import argparse
import copy
import fcntl
import json
import os
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import yaml

from jepa3d.paired_metrics import compare_reference, summarize, validate_predictions
from jepa3d.paired_rooms import RELATIONS, digest, file_hash, make_plan, publish, read_json, write_json


ROOT = Path(__file__).resolve().parents[2]
JOB = ROOT / "experiments/hardneg_v1"


def revision(repo, ref="HEAD"):
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}"], text=True
    ).strip()


def verify_file(path, expected):
    if file_hash(path) != expected:
        raise ValueError(f"file changed since enrollment: {path}")


def verify_code(plan):
    from jepa3d.paths import VJEPA_ROOT

    if revision(ROOT) != plan["implementation_revision"]:
        raise ValueError("implementation revision differs from enrollment")
    sibling = VJEPA_ROOT
    if revision(sibling) != plan["sibling_revision"]:
        raise ValueError("VJEPA_ROOT differs from the recorded training revision")
    for root, source in ((ROOT, "jepa3d"), (sibling, "fusion_gv")):
        subprocess.run(["git", "-C", str(root), "diff", "--quiet", "HEAD", "--", source], check=True)


def prepare(args):
    from jepa3d.paths import expand_paths

    if not args.baseline_revision or not args.sibling_revision:
        raise ValueError("supply --baseline-revision and --sibling-revision from job 12610")
    provenance = {"historical_revision": revision(ROOT, args.baseline_revision),
                  "implementation_revision": revision(ROOT), "sibling_revision": args.sibling_revision}
    verify_code(provenance)
    cfg = expand_paths(yaml.safe_load(args.config.read_text()))
    data, train = cfg["data"], cfg["train"]
    expected = (not any(cfg.get(k, {}).get("enabled") for k in ("slots", "ground", "numeric_head"))
                and not cfg["rae"].get("question_only") and cfg["rae"].get("encoder", "3drae") == "3drae"
                and not any(data.get(k) for k in ("delexicalize", "delexicalize_eval", "answer_weights", "ground_objects"))
                and cfg["fusion"]["x_encoder_output_dim"] == 768 and cfg["model"]["shared_embed_dim"] == 1536
                and train["max_steps"] == 6870 and data["mask_key"] == "answer_key" and data["num_negatives"] == 3
                and len(data["train_manifests"]) == len(data["val_manifests"]) == 1)
    if not expected:
        raise ValueError("expected the effective slot-free hardneg_v1 job 12610 config")
    artifacts = {"config": args.config, "baseline": args.checkpoint, "reference": args.reference,
                 "warm_start": Path(cfg["init_vljepa_ckpt"])}
    run_meta = args.checkpoint.parent / "run_meta.json"
    world = read_json(run_meta)["num_processes"]
    if world != 3:
        raise ValueError(f"expected the original three-process job, got {world}")
    artifacts["run_meta"] = run_meta
    for name in ("train_log.jsonl", "hardneg_log.jsonl"):
        path = args.checkpoint.parent / name
        if path.is_file():
            artifacts[name] = path
    feature = ROOT / cfg["rae"]["feature_config"]
    params = yaml.safe_load(feature.read_text())["params"]
    artifacts.update(feature_config=feature, neck=ROOT / params["pretrained_neck_path"],
                     stats=ROOT / params["normalization_stat_path"])
    backbone = ROOT / params["encoder_params"]["dinov2_path"]
    weights = sorted(p for p in backbone.iterdir() if p.is_file() and p.suffix in {".bin", ".safetensors", ".json"})
    if not any(p.suffix in {".bin", ".safetensors"} for p in weights):
        raise ValueError(f"no local DINO weights in {backbone}")
    artifacts.update({f"dino/{p.name}": p for p in weights})
    plan = make_plan(data["train_manifests"][0], data["val_manifests"][0],
                     read_json(args.aliases) if args.aliases else {}, args.seed)
    plan.update(config=cfg, processes=world, **provenance,
                artifacts={k: {"path": str(p.resolve()), "sha256": file_hash(p)} for k, p in artifacts.items()})
    plan["review_ids"] = {f: [r["id"] for r in plan["rows"] if r["family"] == f and r["donor"]][:2]
                          for f in RELATIONS}
    write_json(args.out / "plan.json", plan)
    print(f"Enrolled {len(plan['rows'])}/{plan['attempted']}, "
          f"paired {sum(bool(r['donor']) for r in plan['rows'])}. See plan.json for IDs and exclusions.")


def control_config(plan, out):
    cfg = copy.deepcopy(plan["config"])
    cfg["rae"]["question_only"] = True
    cfg["train"]["output_dir"] = str((out / "control_ckpt").resolve())
    cfg.setdefault("mlflow", {})["run_name"] = "hardneg_v1_question_only"
    return cfg


def checked_baseline(plan, out):
    baseline = read_json(out / "baseline.json")
    if baseline["plan_sha256"] != digest(plan) or not baseline["replay_passed"]:
        raise ValueError("historical real-scene replay has not passed for this enrollment")
    if (baseline["checkpoint_sha256"] != plan["artifacts"]["baseline"]["sha256"] or
            set(baseline["predictions"]) != {"real", "donor"}):
        raise ValueError("baseline checkpoint or scored conditions differ from enrollment")
    return baseline


def check_training_inputs(plan):
    verify_code(plan)
    verify_file(plan["train"], plan["train_sha256"])
    verify_file(plan["holdout"], plan["holdout_sha256"])
    for item in plan["artifacts"].values():
        verify_file(item["path"], item["sha256"])


def train_control(args, plan):
    checked_baseline(plan, args.out)
    check_training_inputs(plan)
    cfg = control_config(plan, args.out)
    ckpt_dir = Path(cfg["train"]["output_dir"])
    cfg_path = args.out / "control.yaml"
    checkpoint = ckpt_dir / "final.pt"
    binding = {"plan_sha256": digest(plan), "config_sha256": digest(cfg),
               "baseline_sha256": file_hash(args.out / "baseline.json")}
    with (args.out / "train.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt = args.out / "train_inputs.json"
        if receipt.exists():
            if read_json(receipt) != binding or yaml.safe_load(cfg_path.read_text()) != cfg:
                raise ValueError("training inputs differ from the reserved run")
        else:
            if ckpt_dir.exists():
                raise FileExistsError(f"unreserved checkpoint directory {ckpt_dir}")
            if not cfg_path.exists():
                publish(cfg_path, yaml.safe_dump(cfg, sort_keys=False))
            elif yaml.safe_load(cfg_path.read_text()) != cfg:
                raise ValueError("existing control config differs from this run")
            write_json(receipt, binding)
        if not checkpoint.exists():
            # A partial optimizer run cannot be resumed by the historical trainer.
            if ckpt_dir.exists():
                raise FileExistsError(f"incomplete control in {ckpt_dir}; preserve it before restarting from step zero")
            ckpt_dir.mkdir()
            command = [sys.executable, "-m", "accelerate.commands.launch", "--multi_gpu", "--num_processes",
                       str(plan["processes"]), "jepa3d/train.py", "--config", str(cfg_path.resolve())]
            subprocess.run(command, cwd=ROOT, check=True)
        if read_json(ckpt_dir / "run_meta.json")["num_processes"] != plan["processes"]:
            raise ValueError("trained control used a different process count than job 12610")
        import torch
        state = torch.load(checkpoint, weights_only=True, map_location="cpu", mmap=True)
        marker = state["model"].get("question_only_prefix_length")
        if state.get("step") != cfg["train"]["max_steps"] or marker is None or marker.item() != 1032:
            raise ValueError("control checkpoint is missing its final update or fixed-prefix marker")
        final = {**binding, "checkpoint_sha256": file_hash(checkpoint)}
        record = args.out / "trained.json"
        if record.exists():
            if read_json(record) != final:
                raise ValueError("trained control checkpoint changed")
        else:
            write_json(record, final)


def inference(args, plan, question_only):
    import torch
    from jepa3d.data import spar3d_collate
    from jepa3d.model import build_model_from_config
    from jepa3d.paired_scoring import check_prefix, score_condition
    from fusion_gv.infer_gvjepa import _score_candidate_batch

    verify_code(plan)
    device = torch.device(args.device)
    if question_only:
        baseline = checked_baseline(plan, args.out)
        trained = read_json(args.out / "trained.json")
        cfg = control_config(plan, args.out)
        if (trained["plan_sha256"] != digest(plan) or trained["config_sha256"] != digest(cfg)
                or trained["baseline_sha256"] != file_hash(args.out / "baseline.json")):
            raise ValueError("trained control does not match the replay and enrollment")
        checkpoint = Path(cfg["train"]["output_dir"]) / "final.pt"
        checkpoint_sha = trained["checkpoint_sha256"]
        modes, output = ("question",), "question.json"
    else:
        cfg = plan["config"]
        checkpoint = plan["artifacts"]["baseline"]["path"]
        checkpoint_sha = plan["artifacts"]["baseline"]["sha256"]
        modes, output = ("real", "donor"), "baseline.json"
    verify_file(checkpoint, checkpoint_sha)
    images = {}
    for row in plan["rows"]:
        images.update(zip(row["bundle"]["images"], row["image_sha256"]))
    for path, expected in images.items():
        verify_file(path, expected)
    model = build_model_from_config(cfg)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state.get("step") != cfg["train"]["max_steps"]:
        raise ValueError("expected the historical final microstep")
    model.load_state_dict(state["model"], strict=True)
    del state
    model.to(device).eval()
    size = cfg["data"]["image_size"]
    tokenizers = {name: digest([json.loads(tok.backend_tokenizer.to_str()), tok.special_tokens_map,
                                tok.padding_side, tok.truncation_side])
                  for name, tok in (("query", model.query_tokenizer), ("target", model.y_tokenizer))}
    if question_only and tokenizers != baseline["tokenizers"]:
        raise ValueError("the control and visual baseline loaded different tokenizers")
    result = {"plan_sha256": digest(plan), "checkpoint_sha256": checkpoint_sha, "step": cfg["train"]["max_steps"],
              "tokenizers": tokenizers,
              "versions": {k: version(k) for k in ("torch", "transformers", "accelerate", "tokenizers")},
              "diagnostics": {}, "predictions": {}}
    result["diagnostics"]["prefix"] = check_prefix(model, plan["rows"], question_only, spar3d_collate, device, size)
    for mode in modes:
        scores, checks = score_condition(model, plan["rows"], mode, device, args.batch_size, args.workers, size,
                                         _score_candidate_batch, spar3d_collate, calibration=True)
        validate_predictions(plan["rows"], scores, mode)
        result["predictions"][mode], result["diagnostics"][mode] = scores, checks
        if mode == "real":
            ref = plan["artifacts"]["reference"]
            verify_file(ref["path"], ref["sha256"])
            ref = read_json(ref["path"])
            mismatches = compare_reference(plan["rows"], scores, ref)
            hist = ref["real"]["holdout"]
            correct_job = (ref.get("step") == result["step"] and plan["attempted"] == 6786
                           and plan["train_rows"] == 658518 and not plan["rejected"]
                           and hist["long/all"]["rows"] == 6786 and round(100 * hist["long/all"]["acc"], 1) == 46.7
                           and all(hist.get("long/" + f, {}).get("rows") == 1000 for f in RELATIONS))
            result.update(replay_passed=bool(correct_job and not mismatches), replay_discrepancies=mismatches)
            if not result["replay_passed"]:
                write_json(args.out / "replay_failed.json", result)
                raise ValueError("real-scene replay differs from job 12610; see replay_failed.json")
    write_json(args.out / output, result)
    print(f"Wrote {args.out / output}", flush=True)


def report(args, plan):
    baseline = checked_baseline(plan, args.out)
    question = read_json(args.out / "question.json")
    trained = read_json(args.out / "trained.json")
    if (question["plan_sha256"] != digest(plan) or trained["plan_sha256"] != digest(plan)
            or set(question["predictions"]) != {"question"}
            or question["checkpoint_sha256"] != trained["checkpoint_sha256"]
            or trained["baseline_sha256"] != file_hash(args.out / "baseline.json")):
        raise ValueError("mixed plans, baseline or control checkpoints in report")
    summary = summarize(plan, {**baseline["predictions"], **question["predictions"]})
    summary.update(plan_sha256=question["plan_sha256"], baseline_sha256=baseline["checkpoint_sha256"],
                   control_sha256=question["checkpoint_sha256"])
    write_json(args.out / "metrics.json", summary)
    print(json.dumps({"coverage": summary["coverage"], **summary["inference"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["run", "prepare", "baseline", "train", "control", "report"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=JOB / "ckpt/config.yaml")
    parser.add_argument("--checkpoint", type=Path, default=JOB / "ckpt/final.pt")
    parser.add_argument("--reference", type=Path, default=JOB / "results/eval_final.json")
    parser.add_argument("--aliases", type=Path, help="JSON object mapping verified subset/scan to physical room")
    parser.add_argument("--baseline-revision", help="3DRAE_JEPA commit recorded for the original training job")
    parser.add_argument("--sibling-revision", help="full original fusion_gv git SHA, checked against VJEPA_ROOT")
    parser.add_argument("--seed", type=int, default=12610, help="pre-inference donor assignment")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    args.out = args.out.resolve()
    os.environ["HF_HUB_OFFLINE"] = "1"
    if args.stage in ("prepare", "run"):
        if (args.out / "plan.json").exists():
            raise FileExistsError(f"existing enrollment at {args.out}; use the next individual stage")
        prepare(args)
        if args.stage == "run":
            # Separate processes release the baseline CUDA allocator before
            # Accelerate starts three ranks in the same allocation.
            for stage in ("baseline", "train", "control", "report"):
                subprocess.run([sys.executable, "-m", "jepa3d.tools.paired_room", stage, "--out", str(args.out),
                                "--device", args.device, "--batch-size", str(args.batch_size),
                                "--workers", str(args.workers)], cwd=ROOT, check=True)
    else:
        plan = read_json(args.out / "plan.json")
        if args.stage == "train":
            train_control(args, plan)
        elif args.stage == "report":
            report(args, plan)
        else:
            inference(args, plan, question_only=args.stage == "control")


if __name__ == "__main__":
    main()
