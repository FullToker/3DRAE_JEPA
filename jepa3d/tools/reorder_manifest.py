#!/usr/bin/env python3
"""Copy a manifest in an order in which the "shuffled" control really gives every row another scene.

The control (jepa3d/tools/blind_check.py) gives each row the scene tokens of the previous row of its
batch (row 0: the last row's). The VSI-Bench manifests are ordered by scene, so in batches of 32 that
neighbour is the same scene for 68% of the rows of the test split: other frames, same room. This writes
the same rows in a seeded random order, repaired until no row of any batch has a neighbour of its scene.

Usage:
    python jepa3d/tools/reorder_manifest.py --manifests <jsonl> [<jsonl> ...] --output-dir <dir>
"""

import argparse
import json
import random
from pathlib import Path


def collisions(scenes: list, batch: int) -> list:
    out = []
    for start in range(0, len(scenes), batch):
        size = min(batch, len(scenes) - start)
        out += [start + i for i in range(size) if scenes[start + i] == scenes[start + (i - 1) % size]]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in args.manifests:
        lines = Path(path).read_text().splitlines()
        scene = lambda line: "{subset}/{scene}".format(**json.loads(line))
        before = len(collisions([scene(l) for l in lines], args.batch_size))
        rng = random.Random(args.seed)
        rng.shuffle(lines)
        scenes = [scene(l) for l in lines]
        for _ in range(1000):
            bad = collisions(scenes, args.batch_size)
            if not bad:
                break
            for i in bad:
                j = rng.randrange(len(lines))
                lines[i], lines[j] = lines[j], lines[i]
                scenes[i], scenes[j] = scenes[j], scenes[i]
        left = len(collisions(scenes, args.batch_size))
        (out_dir / Path(path).name).write_text("\n".join(lines) + "\n")
        print(f"{Path(path).name}: {len(lines)} rows, neighbour of the same scene {before} -> {left}")


if __name__ == "__main__":
    main()
