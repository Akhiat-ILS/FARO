"""Check a portable NPZ + JSON regression fixture against a supplied model.

Fixture NPZ: x, y, expected_adversarial. JSON: config, seeds, success,
search_forwards, backwards. Checkpoints/images remain external to the Git repo.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from faro import AttackConfig, FaroAttack


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--fixture", required=True, help="NPZ fixture")
    parser.add_argument("--reference", required=True, help="JSON fixture metadata")
    args = parser.parse_args()
    reference = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    with np.load(args.fixture, allow_pickle=False) as data:
        x, y, expected = (data[k].copy() for k in ("x", "y", "expected_adversarial"))
    model = tf.keras.models.load_model(args.model, compile=False)
    attacker = FaroAttack(model, AttackConfig(**reference["config"]))
    attacker.warmup(x[:1], y[:1])
    result = attacker.generate(x, y, seeds=reference["seeds"])
    np.testing.assert_array_equal(result.success, reference["success"])
    np.testing.assert_array_equal([s.search_forwards for s in result.samples], reference["search_forwards"])
    np.testing.assert_array_equal([s.backwards for s in result.samples], reference["backwards"])
    np.testing.assert_allclose(result.adversarial, expected, rtol=0, atol=5e-7)
    print(f"PASS: {len(y)} saved cases, candidates, success flags and search counts")


if __name__ == "__main__":
    main()
