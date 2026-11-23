"""Offline CPU integration with the sibling's actual predictor, scorer, loss and checkpoint writer."""

import copy
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from test_paired_room import fixture, write_rows
from jepa3d.paired_metrics import summarize
from jepa3d.paired_rooms import make_plan


@unittest.skipUnless(os.environ.get("VJEPA_ROOT"), "set VJEPA_ROOT to run the sibling integration")
class IntegrationTests(unittest.TestCase):
    def test_prepare_records_revisions_and_hashes_all_recovered_assets(self):
        import yaml
        from types import SimpleNamespace
        from jepa3d.paired_rooms import read_json, write_json
        from jepa3d.tools.paired_room import ROOT, check_training_inputs, prepare, revision
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture(root, 3)
            cfg = yaml.safe_load((ROOT / "jepa3d/configs/hardneg_v1.yaml").read_text())
            cfg["data"].update(train_manifests=[str(root / "train.jsonl")], val_manifests=[str(root / "holdout.jsonl")])
            cfg["init_vljepa_ckpt"] = str(root / "warm.pt")
            cfg["rae"]["feature_config"] = str(root / "features.yaml")
            backbone = root / "dino"
            backbone.mkdir()
            assets = [root / name for name in ("warm.pt", "neck", "stats", "final.pt")] + [backbone / "tiny.bin"]
            for path in assets:
                path.write_bytes(b"fixture")
            feature = {"pretrained_neck_path": str(root / "neck"), "normalization_stat_path": str(root / "stats"),
                       "encoder_params": {"dinov2_path": str(backbone)}}
            (root / "features.yaml").write_text(yaml.safe_dump({"params": feature}))
            config = root / "config.yaml"
            config.write_text(yaml.safe_dump(cfg))
            write_json(root / "eval_final.json", {})
            write_json(root / "run_meta.json", {"num_processes": 3})
            args = SimpleNamespace(out=root / "run", config=config, checkpoint=root / "final.pt",
                                   reference=root / "eval_final.json", aliases=None, seed=12610,
                                   baseline_revision=revision(ROOT), sibling_revision="fixture")
            with patch("jepa3d.tools.paired_room.verify_code") as verify:
                prepare(args)
                plan = read_json(args.out / "plan.json")
                self.assertEqual(plan["historical_revision"], args.baseline_revision)
                self.assertEqual(plan["attempted"], 12)
                verify.assert_called_once()
                check_training_inputs(plan)
                (root / "neck").write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "changed since enrollment"):
                    check_training_inputs(plan)

    def test_training_loader_preserves_ids_masks_and_negative_draws(self):
        from jepa3d.data import build_loader
        from jepa3d.train import _set_seed
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = fixture(root, 3)
            for i, row in enumerate(rows):
                row.update(answer_key=f"key-{i % 3}", negatives=["right", "back", "above", "below", "further"])
            path = root / "mix.jsonl"
            write_rows(path, rows)
            cfg = {"data": {"image_size": [8, 8], "num_workers": 0, "mask_key": "answer_key",
                            "num_negatives": 3, "pin_memory": False}}
            batches = []
            for blind in (False, True):
                _set_seed(42)
                loader = build_loader([path], {**cfg, "rae": {"question_only": blind}}, 4)
                batch = next(iter(loader))
                batches.append((batch["query"], batch["target"], batch["negatives"], batch["group_id"]))
                loader.dataset._fh.close()
            self.assertEqual(batches[0], batches[1])
            self.assertTrue(all(len(ns) == 3 for ns in batches[1][2]))

    def test_recovered_completed_control_cannot_launch_a_second_run(self):
        import torch
        import yaml
        from types import SimpleNamespace
        from jepa3d.paired_rooms import digest, file_hash, read_json, write_json
        from jepa3d.tools.paired_room import control_config, train_control
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            plan = {"config": {"rae": {}, "train": {"max_steps": 6870}}, "processes": 3,
                    "artifacts": {"baseline": {"sha256": "original"}}}
            cfg = control_config(plan, out)
            write_json(out / "baseline.json", {"plan_sha256": digest(plan), "replay_passed": True,
                       "checkpoint_sha256": "original", "predictions": {"real": {}, "donor": {}}})
            write_json(out / "train_inputs.json", {"plan_sha256": digest(plan), "config_sha256": digest(cfg),
                       "baseline_sha256": file_hash(out / "baseline.json")})
            (out / "control.yaml").write_text(yaml.safe_dump(cfg))
            ckpt = out / "control_ckpt/final.pt"
            ckpt.parent.mkdir()
            torch.save({"model": {"question_only_prefix_length": torch.tensor(1032)}, "step": 6870}, ckpt)
            with patch("jepa3d.tools.paired_room.check_training_inputs"), \
                    patch("jepa3d.tools.paired_room.subprocess.run", side_effect=AssertionError("launched again")):
                meta = ckpt.parent / "run_meta.json"
                meta.write_text('{"num_processes": 1}')
                with self.assertRaisesRegex(ValueError, "process count"):
                    train_control(SimpleNamespace(out=out), plan)
                meta.write_text('{"num_processes": 3}')
                train_control(SimpleNamespace(out=out), plan)
                train_control(SimpleNamespace(out=out), plan)
            self.assertEqual(read_json(out / "trained.json")["checkpoint_sha256"], file_hash(ckpt))

    def test_train_reload_and_score_all_conditions(self):
        import torch
        from accelerate import Accelerator
        from jepa3d.data import build_loader, spar3d_collate
        from jepa3d.model import build_model_from_config
        from jepa3d.paired_scoring import check_prefix, sample, score_condition
        from jepa3d.train import _set_seed
        from jepa3d.trainer import HardNegTrainer
        from fusion_gv.infer_gvjepa import _score_candidate_batch

        class FixtureVisualEncoder(torch.nn.Module):
            dim, num_latents = 768, 1024

            def __init__(self, **_):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.randn(1))

            def forward(self, images, cams):
                signal = images.mean((1, 2, 3, 4)) + cams[:, :, 12].mean(1)
                return (signal * self.scale)[:, None, None].expand(-1, 1032, 768)

        cfg = {"fusion": {"x_encoder_type": "3drae", "x_encoder_output_dim": 768},
               "model": {"query_model_name": "toy", "y_encoder_name": "toy", "predictor_hidden_size": 16,
                         "predictor_heads": 2, "predictor_layers": 1, "shared_embed_dim": 1536},
               "rae": {}, "data": {"image_size": [8, 8], "mask_key": "answer_key", "num_negatives": 3,
                                    "num_workers": 0, "val_num_workers": 0, "pin_memory": False}}
        control_cfg = copy.deepcopy(cfg)
        control_cfg["rae"]["question_only"] = True
        device = torch.device("cpu")
        with tempfile.TemporaryDirectory() as temp, patch("jepa3d.model.ThreeDRAEXEncoder", FixtureVisualEncoder):
            root = Path(temp)
            fixture(root, 4)
            path = root / "holdout.jsonl"
            plan = make_plan(root / "train.jsonl", path, {})
            rows = plan["rows"]
            _set_seed(42)
            real = build_model_from_config(cfg)
            _set_seed(42)
            control = build_model_from_config(control_cfg)
            for name, value in real.state_dict().items():
                self.assertTrue(torch.equal(value, control.state_dict()[name]), name)
            with self.assertRaises(RuntimeError):
                control.load_state_dict(real.state_dict(), strict=True)

            by_id = {r["id"]: r for r in rows}
            donor = sample(rows[0], by_id, "donor")
            self.assertEqual(donor["query"], rows[0]["query"])
            self.assertEqual(donor["c2w"], by_id[rows[0]["donor"]]["bundle"]["c2w"])
            self.assertEqual(donor["image_paths"], by_id[rows[0]["donor"]]["bundle"]["images"])
            for model, blind in ((real, False), (control, True)):
                model.eval()
                self.assertEqual(check_prefix(model, rows, blind, spar3d_collate, device, [8, 8])["identical"], blind)

            loader = build_loader([path], control_cfg, 2, train=True)
            before = control.pred_proj.weight.detach().clone()
            with patch("jepa3d.data.load_views", side_effect=AssertionError("control opened images")), \
                    patch.object(control.x_encoder, "forward", side_effect=AssertionError("control encoded images")):
                trainer = HardNegTrainer(model=control, optimizer=torch.optim.AdamW(control.parameters(), lr=1e-3),
                                         train_loader=loader, output_dir=root / "ckpt", max_steps=2,
                                         grad_accum_steps=2, save_every=0, accelerator=Accelerator(cpu=True),
                                         historical_hardneg=True)
                trainer.fit()
            loader.dataset._fh.close()
            state = torch.load(root / "ckpt/final.pt", map_location="cpu", weights_only=True)
            self.assertEqual(state["step"], 2)
            self.assertFalse(torch.equal(control.pred_proj.weight, before))
            control = build_model_from_config(control_cfg)
            control.load_state_dict(state["model"], strict=True)
            predictions = {}
            for mode in ("real", "donor", "question"):
                model = control if mode == "question" else real
                scored, checks = score_condition(model.eval(), rows, mode, device, 3, 0, [8, 8],
                                                 _score_candidate_batch, spar3d_collate, calibration=True)
                self.assertLess(checks["cached_score_error"], 1e-5)
                predictions[mode] = scored
            self.assertEqual(summarize(plan, predictions)["coverage"]["common"], len(rows))
            solo, _ = score_condition(real, rows, "donor", device, 1, 0, [8, 8],
                                      _score_candidate_batch, spar3d_collate)
            for qid, rec in solo.items():
                torch.testing.assert_close(torch.tensor(rec["scores"]), torch.tensor(predictions["donor"][qid]["scores"]),
                                           atol=1e-5, rtol=1e-5)
