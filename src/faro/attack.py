"""Public API for the experimentally evaluated fixed CE/DLR schedule."""
from dataclasses import asdict
import time
import numpy as np
import tensorflow as tf
from ._backend import TensorFlowBackend
from .config import AttackConfig
from .results import AttackResult, SampleResult


class FaroAttack:
    """Single-image search, with a sequential batch convenience API.

    Accepts a built single-input Keras image classifier with raw float32 NHWC
    inputs in [0,1] and at least three output logits. Inference must be
    deterministic and differentiable. Instances are not thread-safe.
    """

    def __init__(self, model: tf.keras.Model, config: AttackConfig):
        if not isinstance(config, AttackConfig):
            raise TypeError("config must be an AttackConfig")
        try:
            inputs, outputs = model.inputs, model.outputs
        except AttributeError as exc:
            raise ValueError("Use a built Functional or Sequential Keras model") from exc
        if len(inputs) != 1 or len(outputs) != 1 or len(inputs[0].shape) != 4 or len(outputs[0].shape) != 2:
            raise ValueError("Require one NHWC image input and one [batch, classes] logits output")
        if any(d is None for d in inputs[0].shape[1:]):
            raise ValueError("Input height, width and channels must be fixed")
        if outputs[0].shape[-1] is None or int(outputs[0].shape[-1]) < 3:
            raise ValueError("DLR requires at least three classes")
        if str(inputs[0].dtype) != "float32" or str(outputs[0].dtype) != "float32":
            raise ValueError("This release supports float32 model inputs and logits")
        final = model.layers[-1]
        activation = getattr(getattr(final, "activation", None), "__name__", "")
        if isinstance(final, tf.keras.layers.Softmax) or activation in ("softmax", "sigmoid"):
            raise ValueError("Supply pre-softmax logits, not probabilities")
        self.model, self.config = model, config
        self.input_shape = tuple(int(d) for d in inputs[0].shape[1:])
        self.classes = int(outputs[0].shape[-1])
        self._backend = TensorFlowBackend(model)

    def _inputs(self, images, labels):
        x, y = np.asarray(images), np.asarray(labels)
        if x.ndim != 4 or tuple(x.shape[1:]) != self.input_shape or len(x) == 0:
            raise ValueError(f"Expected a nonempty batch with shape (N, {self.input_shape})")
        if x.dtype.kind != "f" or not np.isfinite(x).all() or x.min() < 0 or x.max() > 1:
            raise ValueError("Images must be finite floating-point values in [0,1]; normalize uint8 first")
        if y.shape != (len(x),) or y.dtype.kind not in "iu" or np.any(y < 0) or np.any(y >= self.classes):
            raise ValueError("Labels must be a one-dimensional integer array of valid class indices")
        return x.astype(np.float32, copy=True), y.astype(np.int32, copy=True)

    def warmup(self, images, labels) -> None:
        """Compile on the first image; excluded from generate() evaluation counts."""
        x, y = self._inputs(images, labels)
        self._backend.warmup(x[:1], y[:1])

    def generate(self, images, labels, *, seeds=None) -> AttackResult:
        """Attack each image independently; preserve input arrays and model weights.

        By default image i uses config.seed+i. Supply explicit per-image seeds
        when splitting/reordering a dataset. Initially incorrect inputs are
        returned unchanged and never counted as attack successes.
        """
        x, y = self._inputs(images, labels)
        if seeds is None:
            seed_values = [(self.config.seed + i) % (2**63 - 1) for i in range(len(x))]
        else:
            values = np.asarray(seeds)
            if values.shape != (len(x),) or values.dtype.kind not in "iu" or np.any(values < 0) or np.any(values >= 2**63 - 1):
                raise ValueError("seeds must contain one nonnegative integer below 2**63-1 per image")
            seed_values = [int(v) for v in values]
        outputs, records = [], []
        for index, (label, seed) in enumerate(zip(y, seed_values)):
            candidate, record = self._one(x[index:index + 1], int(label), seed)
            outputs.append(candidate)
            records.append(record)
        return AttackResult(np.concatenate(outputs), tuple(records), asdict(self.config))

    def _one(self, original, label, seed):
        started = time.perf_counter()
        logits = self._backend.logits(original)[0]
        clean = int(logits.argmax())
        best = original.copy()
        best_margin = float(logits[label] - np.max(np.delete(logits, label)))
        forwards, backwards, blocks, verification = 1, 0, 0, 0
        success = False
        prediction = clean
        if clean != label:
            status = "already_misclassified"
        else:
            status = "not_found"
            if self.config.epsilon > 0:
                for block in range(self.config.blocks):
                    result = self._backend.block(
                        original, label, best, self.config.epsilon, self.config.steps_per_block,
                        (seed + 1009 * block) % (2**63 - 1), "ce" if block % 2 == 0 else "dlr")
                    forwards += result.forwards
                    backwards += result.backwards
                    blocks += 1
                    if result.margin < best_margin or result.success:
                        best, best_margin = result.candidate, result.margin
                    success = result.success
                    if success:
                        break
            if not np.isfinite(best).all() or best.min() < 0 or best.max() > 1:
                raise RuntimeError("Optimizer returned a candidate outside valid input bounds")
            if float(np.max(np.abs(best - original))) > self.config.epsilon + 5e-7:
                raise RuntimeError("Optimizer returned a candidate outside the original threat set")
            verified = self._backend.logits(best)[0]
            verification = 1
            prediction = int(verified.argmax())
            if (prediction != label) != success:
                raise RuntimeError("Final prediction disagrees with search; classifier may be nondeterministic")
            best_margin = float(verified[label] - np.max(np.delete(verified, label)))
            status = "success" if success else "not_found"
        record = SampleResult(
            label, clean, prediction, bool(success), status,
            float(np.max(np.abs(best - original))), best_margin, blocks,
            forwards, backwards, verification, time.perf_counter() - started, seed)
        return best, record
