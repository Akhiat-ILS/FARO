# Benchmark evidence and reproduction

The `results` directory contains compact, exported historical observations and
per-image reference records. It contains no model weights or image arrays.
The results were obtained by the research runner from which the fixed CE/DLR
schedule was extracted. They are not a claim that the entire benchmark was
rerun with this release, or that FARO is universally stronger than AutoAttack.

Both reference sets contain 128 fresh clean-correct test images, selected before
attacks after conservatively excluding earlier recorded image IDs. Surrogate
experiments ran alongside the fixed baseline; only the original fixed FARO and
full standard AutoAttack results are exported here. The fixed schedule contains
ten 100-iteration CE/DLR blocks, no global forward cutoff, and no learned policy.

Full AutoAttack used APGD-CE, targeted APGD, targeted FAB and Square with default
standard budgets at revision `a39220048b3c9f2cca9a4d3a54604793c68eca7e`. AutoAttack
ran in batches of 16; FARO ran one image at a time. Thus wall times are not an
equal-batching comparison. The MNIST AutoAttack wall time is explicitly unusable
because of an anomalous elapsed-time record. All saved historical candidates
passed the original TensorFlow 1 (MNIST) or ONNX (CIFAR) runtime audit.

## Evaluate a reference set

Supply the exact raw-input `.keras` logits checkpoint whose SHA-256 is recorded
in the relevant reference JSON. Dataset loading is provided by Keras; it may
download MNIST/CIFAR-10 on the first run. Checkpoints are not bundled, and this
script does not download or convert classifier weights. A newly serialized port
can have a different file hash; use your own dataset evaluation instead of
claiming exact reproduction if the reference hash is unavailable.

```bash
python benchmarks/evaluate_reference.py --benchmark cifar --model checkpoints/sri_resnet_b.keras --output runs/cifar-reference
python benchmarks/evaluate_reference.py --benchmark mnist --model checkpoints/secret.keras --output runs/mnist-reference
```

The runner verifies checkpoint identity, uses the recorded indices and seeds,
and writes new results. It compares per-image successes and search counts with
the historical baseline. A mismatch is reported rather than hidden or tuned
away. Package verification forwards are additional to the historical search
counts. Historical runtime comparisons should not be reused for this extraction.

## Portable regression fixture

`check_regression.py` accepts an external small fixture: NPZ fields `x`, `y`,
`expected_adversarial`, plus JSON fields `config`, `seeds`, `success`,
`search_forwards`, and `backwards`. It checks candidate arrays within 5e-7 and
requires exact success and search-count agreement:

```bash
python benchmarks/check_regression.py --model classifier.keras --fixture cases.npz --reference cases.json
```

This lets maintainers test extraction against saved cases without importing the
research project. See `results/package_validation.json` for the actual local
validation performed during packaging. GPU and cross-platform parity are not
established by these checks. No unsuccessful attack is a robustness certificate.
