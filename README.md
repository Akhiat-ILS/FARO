# FARO — fixed CE/DLR adversarial testing

FARO tests a trained TensorFlow/Keras image classifier for untargeted **Linf
adversarial examples**. Supply a logits model, raw images, true labels, and a
perturbation limit. FARO returns candidate images, independently checked success
flags, and per-image computation counts.

**This release packages the best-supported fixed schedule from our experiments.**
It uses the official AutoPGD optimizer, alternating cross-entropy (CE) and
difference-of-logits-ratio (DLR) blocks. It does **not** use fingerprint-guided
selection, an archive, RL, community detection, or pruning. The FARO name is
retained from the broader research project; those experimental variants are not
part of this release. This is not a new implementation of AutoPGD or the full
AutoAttack ensemble.

An unsuccessful search is **not a robustness certificate**.

## Install

Use a fresh environment: this package imports as `faro` and must not share an
environment with the earlier research distribution `faro-tf`.

Python 3.10–3.13 is accepted; local validation used Python 3.12. The GitHub Actions
workflow is configured for 3.11 and 3.12 on Windows and Linux. Git is needed to
install the pinned AutoAttack dependency. From the repository root:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead:
# .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install .
```

For development: `python -m pip install -e ".[test]"`.

Model computation and differentiation use **TensorFlow**. **PyTorch** is also
required because upstream AutoPGD uses it for optimizer tensors; this is not a
TensorFlow-only dependency stack. AutoAttack is pinned to commit
`a39220048b3c9f2cca9a4d3a54604793c68eca7e`. No neighboring research folders, dataset
files, or checkpoints are required to import and use the package.

CPU inference is the validated execution mode. For a smaller CPU-only PyTorch
installation, install it from its CPU index before installing FARO:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install .
```

## Quick start

```python
import numpy as np
import tensorflow as tf
from faro import AttackConfig, FaroAttack

model = tf.keras.models.load_model("classifier.keras", compile=False)
with np.load("samples.npz", allow_pickle=False) as data:
    images = data["x"]   # float NHWC in [0, 1]
    labels = data["y"]   # shape (N,), integer true labels

attack = FaroAttack(model, AttackConfig(epsilon=4 / 255, seed=2026))
attack.warmup(images[:1], labels[:1])  # optional: compile outside measured search
result = attack.generate(images, labels)

print(result.success)                 # one boolean per input
print(result.success_rate)            # denominator: initially correct inputs
print(result.to_dict()["summary"])
result.save("runs/experiment-001")    # refuses to overwrite an existing path
# result.adversarial has the same NHWC shape as images
```

Run `python examples/attack_keras_model.py` for a complete, deterministic smoke
example needing no downloads. It uses a tiny synthetic model; it is not a
benchmark of attack strength.

To evaluate your own dataset:

```bash
python examples/evaluate_dataset.py --model classifier.keras --data samples.npz --epsilon 0.01568627450980392 --output runs/evaluation --seed 2026
```

An optional `--indices manifest.json` selects a subset. The JSON may be a list or
an object containing `indices`. The CLI writes `arrays.npz` and `results.json`.

## First MNIST experiment

Run this from the repository root after installing FARO:

```bash
python examples/mnist.py --samples 20 --output runs/mnist-001
```

On Windows, you can explicitly use your Python 3.12 environment:

```powershell
.\.venv312\Scripts\python.exe examples\mnist.py --samples 20 --output runs\mnist-001
```

The first run downloads MNIST if needed, trains a small CNN for three epochs,
saves `checkpoints/mnist_cnn.keras`, and reports clean accuracy on all 10,000
test images. Later runs reuse the checkpoint. It attacks a seeded random subset
of test images with epsilon 0.3 and the normal ten 100-step blocks. Initially
misclassified images are excluded from the success-rate denominator.

Results include `results.json`, float32 candidates in `arrays.npz`, original
images and indices in `originals.npz`, and settings in `experiment.json`.
The `images` folder contains original/adversarial PNG pairs for viewing; these
are rounded previews, whereas the float32 candidates are the evaluated data.
Choose a new output directory for each run. This trains a standard classifier,
not the robust Madry secret model used in the historical benchmark, and is not
an AutoAttack comparison. Training time depends on your hardware.

## Model and input contract

- A built, single-input Functional/Sequential Keras classifier, with fixed image
  dimensions, NHWC float32 input, and one float32 `[N, classes]` logits output.
- At least **three classes** (DLR uses the three largest logits). Ten-class
  MNIST/CIFAR models are the empirical benchmark coverage.
- Deterministic, differentiable inference with `training=False`. Random defenses,
  stateful inference, mixed precision, subclass-only models, binary classifiers,
  nonimage inputs, and norms other than Linf are outside this release's contract.
- Images must already be floating-point values in `[0,1]`. Convert uint8 with
  `images.astype(np.float32) / 255`. Labels must be integer class indices, not
  one-hot vectors. Inputs and weights are not modified.
- Put differentiable normalization **inside the model** so epsilon remains in
  raw pixel units. For example, use a Keras `Rescaling` layer before your trained
  classifier. Do not pass standardized negative-valued tensors as raw images.
- Return **pre-softmax logits**, not probabilities. Common final softmax/sigmoid
  layers are rejected; the caller is responsible for hidden/custom normalization.

Initially incorrect images are returned unchanged with status
`already_misclassified` and `success=False`. `success_rate` excludes these inputs
and is NaN if none are initially correct (JSON uses null). Status `not_found`
means only that this configured search did not find a witness.

## Exact default schedule

`AttackConfig(epsilon=..., steps_per_block=100, blocks=10, seed=0)`:

1. Check the original prediction and initialize the best candidate to the input.
2. Run CE, DLR, CE, DLR, … for ten blocks. Each block uses upstream AutoPGD with
   100 iterations, one restart, Linf projection, and its standard step adaptation.
3. Start each block from the best actual classification-margin candidate so far.
   Reset AutoPGD optimizer state at each block. The **original input** always
   defines the perturbation ball. These are best-point continuations, not ten
   independent random starts.
4. Stop immediately when an evaluated point changes the original classifier's
   prediction, or when the schedule finishes.
5. Check the returned candidate's bounds and prediction independently.

There is **no 61-forward global cutoff**. Each block allows its normal
`steps_per_block + 2` forwards (clean setup and initial gradient plus iterations).
This bound preserves the evaluated research runner's accounting. For an
unsuccessful default run, search counts are normally 1,021 forwards and 1,010
backwards, plus one separately reported final verification forward.

Each batch item is attacked **sequentially**, not with a vectorized batch attack.
Image i defaults to `seed+i`; block j uses that seed plus `1009*j` (wrapped below
`2**63-1` at the integer limit). Use `generate(..., seeds=[...])` to preserve
per-image seeds when splitting or reordering data. CPU Torch RNG state is restored
after each optimizer block. Numerical reproducibility across hardware/library
versions is not guaranteed. Instances are not thread-safe.

The library does not change application thread counts or global TensorFlow
settings. `epsilon=0` returns the unchanged input without running optimizer blocks.

## Results and cost accounting

`AttackResult.samples` contains one `SampleResult` per input:

| Field | Meaning |
|---|---|
| `success`, `status` | Verified misclassification of a clean-correct input; or why no attack succeeded |
| `clean_prediction`, `adversarial_prediction`, `label` | Class indices |
| `linf`, `margin` | Measured maximum pixel change and final true-minus-best-other logit margin |
| `search_forwards`, `backwards` | Per-image search evaluations, including gradient forwards and setup |
| `verification_forwards` | Additional final prediction checks |
| `total_forwards` | Search plus final verification forwards |
| `blocks_executed`, `seed`, `seconds` | Executed schedule, image seed, and elapsed time |

These are **white-box computation counts**, not black-box query-complexity claims.
Warmup is outside counts. `seconds` includes final verification and, if no warmup
was called, initial tracing. Model loading and external dataset processing are
not included. Saved NPZ files contain no pickled objects.

## Evidence and limits

Historical fixed-schedule results on the original research runner:

| Dataset / checkpoint | Raw Linf epsilon | Images | FARO successes | Full AutoAttack successes |
|---|---:|---:|---:|---:|
| MNIST / Madry secret | 0.3 | 128 | 12 | 13 |
| CIFAR-10 / SRI ResNet B | 4/255 | 128 | 79 | 79 |

On CIFAR-10 the successful image sets matched. These results do not establish
general superiority, equivalence across other models, or a robustness guarantee.
Full standard AutoAttack also includes targeted APGD, targeted FAB and Square;
this package includes only the fixed untargeted CE/DLR schedule. Historical
runtime numbers include instrumentation absent from this clean extraction and
must not be presented as newly measured package performance. One historical
MNIST AutoAttack elapsed-time record was anomalous and is not usable for speed
claims. See [benchmark provenance and reproduction](benchmarks/README.md).

## Tests and development

```bash
python -m pytest -q
python examples/attack_keras_model.py
python -m pip install build
python -m build
```

Tests cover a real attack on a deterministic classifier, a failed search beyond
61 forwards, CE/DLR derivatives, constraint checks, already-misclassified inputs,
seed handling, weight/input preservation, and portable result serialization.
Real-checkpoint regression is separate because model weights and datasets are
not bundled in Git. CI is supplied; it has not run on GitHub until you push it.

Contributions: see [CONTRIBUTING.md](CONTRIBUTING.md). The code is licensed under
[MIT](LICENSE); dependencies retain their own licenses. AutoPGD/AutoAttack is
credited to Francesco Croce and Matthias Hein. See
[third-party notices](THIRD_PARTY_NOTICES.md) and their
[AutoAttack repository](https://github.com/fra31/auto-attack) and
[paper](https://arxiv.org/abs/2003.01690).
