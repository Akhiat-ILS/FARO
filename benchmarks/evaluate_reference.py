"""Re-evaluate a historical reference set with a supplied exact Keras model."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from faro import AttackConfig, FaroAttack


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("mnist", "cifar"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads((Path(__file__).parent / "results" / f"{args.benchmark}_reference.json").read_text())
    if hashlib.sha256(args.model.read_bytes()).hexdigest() != reference["model_sha256"]:
        parser.error("Checkpoint SHA-256 differs from the recorded model")
    if args.output.exists():
        parser.error("Output directory exists")
    (_, _), (x, y) = (tf.keras.datasets.mnist.load_data() if args.benchmark == "mnist"
                       else tf.keras.datasets.cifar10.load_data())
    if args.benchmark == "mnist":
        x = x[..., None]
    x, y = x.astype(np.float32) / 255, y.ravel()
    records = reference["records"]
    indices = [r["index"] for r in records]
    attacker = FaroAttack(tf.keras.models.load_model(args.model, compile=False),
                          AttackConfig(**reference["config"]))
    attacker.warmup(x[indices[:1]], y[indices[:1]])
    result = attacker.generate(x[indices], y[indices], seeds=[r["seed"] for r in records])
    result.save(args.output)
    differences = [r["index"] for r, s in zip(records, result.samples)
                   if s.success != r["success"] or s.search_forwards != r["search_forwards"]
                   or s.backwards != r["backwards"]]
    print(json.dumps(result.to_dict()["summary"], indent=2))
    if differences:
        raise SystemExit(f"Historical parity differences at indices: {differences}; new results preserved")
    print("Historical per-image successes and search counts match.")


if __name__ == "__main__":
    main()
