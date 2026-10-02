"""Evaluate a trusted .keras logits model on an NPZ containing x and y."""
import argparse
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from faro import AttackConfig, FaroAttack


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path, help="NPZ: x float NHWC [0,1], y integer labels")
    parser.add_argument("--output", required=True, type=Path, help="New output directory")
    parser.add_argument("--epsilon", required=True, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--steps", default=100, type=int)
    parser.add_argument("--blocks", default=10, type=int)
    parser.add_argument("--indices", type=Path, help="Optional JSON list of indices, or object with indices")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output directory already exists; choose a new one")
    with np.load(args.data, allow_pickle=False) as data:
        x, y = data["x"].copy(), data["y"].copy()
    if args.indices:
        selection = json.loads(args.indices.read_text(encoding="utf-8"))
        indices = selection["indices"] if isinstance(selection, dict) else selection
        x, y = x[indices], y[indices]
    model = tf.keras.models.load_model(args.model, compile=False)
    attacker = FaroAttack(model, AttackConfig(args.epsilon, args.steps, args.blocks, args.seed))
    attacker.warmup(x[:1], y[:1])
    result = attacker.generate(x, y)
    result.save(args.output)
    print(json.dumps(result.to_dict()["summary"], indent=2))


if __name__ == "__main__":
    main()
