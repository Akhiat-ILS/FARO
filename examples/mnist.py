"""Train a small MNIST CNN and attack test images; not a robust-model benchmark."""
import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from faro import AttackConfig, FaroAttack


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Must be positive")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("checkpoints/mnist_cnn.keras"))
    parser.add_argument("--output", type=Path, default=Path("runs/mnist-001"))
    parser.add_argument("--samples", type=positive_int, default=20)
    parser.add_argument("--epochs", type=positive_int, default=3)
    parser.add_argument("--epsilon", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--steps", type=positive_int, default=100)
    parser.add_argument("--blocks", type=positive_int, default=10)
    parser.add_argument("--train-limit", type=positive_int, default=60000,
                        help="Training subset size; reduce only for smoke tests")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a different --output")
    if args.model.suffix != ".keras":
        parser.error("--model must end in .keras")
    if args.samples > 10000 or args.train_limit > 60000:
        parser.error("MNIST has 10000 test images and 60000 training images")
    config = AttackConfig(epsilon=args.epsilon, steps_per_block=args.steps,
                          blocks=args.blocks, seed=args.seed)
    tf.keras.utils.set_random_seed(args.seed)
    print("Loading MNIST (first run may download the dataset)...", flush=True)
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
    test_x = test_x.astype(np.float32)[..., None] / 255.0
    trained = not args.model.exists()
    if trained:
        print("Training a standard CNN, without adversarial training...", flush=True)
        train_x = train_x[:args.train_limit].astype(np.float32)[..., None] / 255.0
        model = tf.keras.Sequential([
            tf.keras.Input((28, 28, 1)),
            tf.keras.layers.Conv2D(32, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(10),  # Raw logits, no softmax.
        ])
        model.compile(optimizer="adam",
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=["accuracy"])
        model.fit(train_x, train_y[:args.train_limit], epochs=args.epochs,
                  batch_size=128, verbose=2)
        args.model.parent.mkdir(parents=True, exist_ok=True)
        model.save(args.model)
    else:
        print(f"Reusing checkpoint: {args.model}", flush=True)
        model = tf.keras.models.load_model(args.model, compile=False)
    attacker = FaroAttack(model, config)
    predictions = model.predict(test_x, batch_size=256, verbose=0).argmax(axis=1)
    clean_accuracy = float(np.mean(predictions == test_y))
    print(f"Clean accuracy on all 10000 test images: {clean_accuracy:.2%}", flush=True)
    # Select independently of model correctness and attack outcomes.
    indices = np.random.default_rng(args.seed).permutation(len(test_y))[:args.samples]
    images, labels = test_x[indices], test_y[indices]
    seeds = [(args.seed + int(i)) % (2**63 - 1) for i in indices]
    attacker.warmup(images[:1], labels[:1])
    records, adversarial = [], []
    for i in range(args.samples):
        result = attacker.generate(images[i:i + 1], labels[i:i + 1], seeds=[seeds[i]])
        records.extend(result.samples)
        adversarial.append(result.adversarial)
        print(f"Image {i + 1}/{args.samples}: {result.samples[0].status}", flush=True)
    from faro import AttackResult
    result = AttackResult(np.concatenate(adversarial), tuple(records), result.config)
    result.save(args.output)
    np.savez_compressed(args.output / "originals.npz", x=images, y=labels, indices=indices)
    metadata = {"dataset": "MNIST test", "model": str(args.model),
                "trained_this_run": trained, "training_epochs_this_run": args.epochs if trained else 0,
                "training_samples_this_run": args.train_limit if trained else 0,
                "clean_test_accuracy": clean_accuracy, "indices": indices.tolist(),
                "seeds": seeds, "selection": "Seeded random subset without correctness filtering",
                "purpose": "Standard CNN tutorial, not the historical Madry secret benchmark"}
    (args.output / "experiment.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    preview = args.output / "images"
    preview.mkdir()
    for i, record in enumerate(records):
        for kind, array in (("original", images[i]), ("adversarial", result.adversarial[i])):
            pixels = np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)
            tf.io.write_file(str(preview / f"{int(indices[i]):05d}_{kind}.png"), tf.io.encode_png(pixels))
    print(json.dumps(result.to_dict()["summary"], indent=2))
    print(f"Saved to {args.output.resolve()}")
    print("PNG previews are rounded to uint8; arrays.npz preserves the evaluated float32 candidates.")
    print("A not_found result is not a robustness certificate.")


if __name__ == "__main__":
    main()
