import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from jepa3d.paired_metrics import compare_reference, summarize
from jepa3d.paired_rooms import RELATIONS, digest, file_hash, make_plan, physical_room, publish, read_json, write_json
from jepa3d.tools.paired_room import checked_baseline, control_config, report


def write_rows(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def fixture(path, rooms=6):
    rows = []
    for idx in range(rooms):
        frames, cameras = [], []
        for view in range(8):
            image = path / f"{idx}-{view}.ppm"
            image.write_bytes(b"P6\n8 8\n255\n" + bytes([idx * 20 + 2, view, 10]) * 64)
            frames.append(str(image))
            cameras.append([[1, 0, 0, view + idx], [0, 1, 0, view * idx], [0, 0, 1, 0], [0, 0, 0, 1]])
        for family in RELATIONS:
            source, kind = family.split("/")
            line = len(rows)
            rows.append({"source": source, "scene": f"scene{idx:04d}_00", "subset": "scannet",
                         "question_type": kind, "src": f"original.jsonl:{line}", "src_line": line,
                         "query": f"Where is item {line}?", "target": "left", "target_short": "left",
                         "negatives": ["right", "back"], "options": ["right", "left", "back"],
                         "answer_key": "left", "images": frames, "intrinsics": [[[8, 0, 4], [0, 8, 4], [0, 0, 1]]] * 8,
                         "c2w": cameras, "frame_ids": list(range(8))})
    train = [{**rows[0], "src": "train:0", "scene": "scene0000_01"}]
    write_rows(path / "train.jsonl", train)
    write_rows(path / "holdout.jsonl", rows)
    return rows


def predictions(plan, wins):
    answer = {m: {} for m in wins}
    for row in plan["rows"]:
        for mode, correct in wins.items():
            if mode == "donor" and row["donor"] is None:
                continue
            gold = row["gold"]
            winner = gold if correct else (gold + 1) % len(row["options"])
            answer[mode][row["id"]] = {
                "scores": [float(i == winner) for i in range(len(row["options"]))], "winner": winner,
                "correct": correct, "gold": gold, "answer": row["options"][winner], "margin": 1 if correct else -1,
                "visual_id": row["donor"] if mode == "donor" else row["id"] if mode == "real" else None}
    return answer


class EnrollmentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.raw = fixture(self.root, rooms=7)

    def plan(self, aliases=None):
        return make_plan(self.root / "train.jsonl", self.root / "holdout.jsonl", aliases or {})

    def test_donors_are_complete_different_rooms_in_disjoint_blocks(self):
        plan = self.plan()
        bank = {r["id"]: r for r in plan["rows"]}
        room_blocks = {}
        for row in bank.values():
            donor = bank[row["donor"]]
            self.assertNotEqual(row["room"], donor["room"])
            self.assertEqual(row["stratum"], donor["stratum"])
            self.assertEqual(row["block"], donor["block"])
            self.assertEqual(room_blocks.setdefault(row["room"], row["block"]), row["block"])
            self.assertTrue(set(row["image_sha256"]).isdisjoint(donor["image_sha256"]))
        self.assertEqual(len(set(room_blocks.values())), 3)
        self.assertEqual(plan["attempted"], 28)
        self.assertEqual(sum(r["split"] == "overlap" for r in bank.values()), 4)
        self.assertEqual(digest(plan), digest(self.plan()))

    def test_donors_do_not_use_question_text_correctness_or_file_order(self):
        first = {r["id"]: r["donor"] for r in self.plan()["rows"]}
        write_rows(self.root / "holdout.jsonl", [{**r, "query": "different question", "target": "new target"}
                                                for r in reversed(self.raw)])
        self.assertEqual(first, {r["id"]: r["donor"] for r in self.plan()["rows"]})

    def test_duplicate_ids_and_ambiguous_gold_are_accounted_for(self):
        self.raw[2]["negatives"].append("left")
        write_rows(self.root / "holdout.jsonl", self.raw + [self.raw[0]])
        plan = self.plan()
        self.assertEqual(plan["attempted"], 29)
        self.assertEqual(len(plan["rejected"]), 3)
        self.assertEqual(sorted(e["line"] for e in plan["rejected"]), [1, 3, 29])

    def test_scannetpp_aliases_unresolved_training_and_singletons(self):
        hold = [{**self.raw[0], "subset": "scannetpp", "scene": "scanA"}]
        train = [{**hold[0], "src": "train:10", "scene": "scanB"}]
        write_rows(self.root / "holdout.jsonl", hold)
        write_rows(self.root / "train.jsonl", train)
        aliases = {"scannetpp/scanA": "room-A"}
        self.assertEqual(self.plan(aliases)["rows"][0]["split"], "unresolved")
        row = self.plan({**aliases, "scannetpp/scanB": "room-A"})["rows"][0]
        self.assertEqual(row["split"], "overlap")
        self.assertIsNone(row["donor"])
        self.assertEqual(physical_room("scannetpp", "scanA", {}), None)

    def test_shared_pixels_cannot_be_used_as_an_intervention(self):
        r, other = self.raw[0], self.raw[4]
        write_rows(self.root / "holdout.jsonl", [r, {**other, "images": r["images"]}])
        self.assertEqual([r["unpaired"] for r in self.plan()["rows"]], ["shared frame content"] * 2)

    def test_alias_on_one_rescan_covers_every_scan_of_the_same_room(self):
        plan = self.plan({"scannet/scene0000_01": "room-A"})
        rows = [r for r in plan["rows"] if r["scene"] == "scene0000_00"]
        self.assertTrue(all(r["room"] == "room-A" and r["split"] == "overlap" for r in rows))
        with self.assertRaisesRegex(ValueError, "conflicting"):
            self.plan({"scannet/scene0000_01": "room-A", "scannet/scene0000_00": "room-B"})

    def test_paired_decision_and_missing_results(self):
        plan = self.plan()
        modes = {"real": True, "donor": False, "question": False}
        scores = predictions(plan, modes)
        summary = summarize(plan, scores)
        self.assertEqual(summary["coverage"]["primary"], 24)
        self.assertEqual(summary["inference"]["decision"], "useful scene dependence")
        self.assertEqual(summary["inference"]["contrasts"]["donor"]["ci_97_5_pp"], [100., 100.])
        self.assertEqual(summary["primary"]["all"]["donor"]["paired"]["correct_to_wrong"], 24)
        self.assertEqual(summary["primary"][RELATIONS[2]]["real"]["recall"], {"left": 1.})
        failed = summarize(plan, predictions(plan, {**modes, "question": True}))
        self.assertEqual(failed["inference"]["decision"], "excludes stipulated advantage")
        del scores["real"][next(iter(scores["real"]))]
        with self.assertRaisesRegex(ValueError, "missing"):
            summarize(plan, scores)

    def test_replay_requires_correct_counts_for_every_type(self):
        plan = self.plan()
        scores = predictions(plan, {"real": True})["real"]
        ref = {"real": {"holdout": {"long/all": {"rows": 28, "acc": 1.},
                                    **{"long/" + k: {"rows": 7, "acc": 1.} for k in RELATIONS}}}}
        self.assertEqual(compare_reference(plan["rows"], scores, ref), {})
        ref["real"]["holdout"]["long/" + RELATIONS[2]]["acc"] = 0.5
        self.assertIn("long/" + RELATIONS[2], compare_reference(plan["rows"], scores, ref))

    def test_missing_family_is_inconclusive_and_checkpoint_gate_is_bound_to_plan(self):
        plan = self.plan()
        for row in plan["rows"]:
            if row["family"] == RELATIONS[0]:
                row["split"] = "overlap"
        report = summarize(plan, predictions(plan, {"real": True, "donor": False, "question": False}))
        self.assertEqual(report["inference"]["decision"], "inconclusive")
        publish(self.root / "baseline.json", json.dumps({"plan_sha256": digest(plan), "replay_passed": False}))
        with self.assertRaisesRegex(ValueError, "replay"):
            checked_baseline(plan, self.root)
        with self.assertRaises(FileExistsError):
            publish(self.root / "baseline.json", "{}")

    def test_control_config_keeps_training_recipe_and_baseline_paths(self):
        original = {"rae": {}, "train": {"output_dir": "original", "max_steps": 6870},
                    "mlflow": {"enabled": False}, "data": {"mask_key": "answer_key", "num_negatives": 3}}
        before = copy.deepcopy(original)
        control = control_config({"config": original}, self.root)
        self.assertEqual(before, original)
        self.assertTrue(control["rae"]["question_only"])
        self.assertEqual(control["data"], original["data"])
        self.assertEqual(control["train"]["max_steps"], 6870)
        self.assertNotEqual(control["train"]["output_dir"], original["train"]["output_dir"])

    def test_report_joins_locked_outputs_and_refuses_foreign_checkpoints(self):
        plan = self.plan()
        plan["artifacts"] = {"baseline": {"sha256": "visual"}}
        data = predictions(plan, {"real": True, "donor": False, "question": False})
        stamp = {"plan_sha256": digest(plan)}
        write_json(self.root / "baseline.json", {**stamp, "replay_passed": True, "checkpoint_sha256": "visual",
                   "predictions": {k: data[k] for k in ("real", "donor")}})
        write_json(self.root / "trained.json", {**stamp, "checkpoint_sha256": "control",
                   "baseline_sha256": file_hash(self.root / "baseline.json")})
        question = {**stamp, "checkpoint_sha256": "control", "predictions": {"question": data["question"]}}
        write_json(self.root / "question.json", question)
        from contextlib import redirect_stdout
        from io import StringIO
        with redirect_stdout(StringIO()):
            report(SimpleNamespace(out=self.root), plan)
        self.assertEqual(read_json(self.root / "metrics.json")["inference"]["decision"], "useful scene dependence")
        (self.root / "question.json").write_text(json.dumps({**question, "checkpoint_sha256": "visual"}))
        with self.assertRaisesRegex(ValueError, "mixed"):
            report(SimpleNamespace(out=self.root), plan)
