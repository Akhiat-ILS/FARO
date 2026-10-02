"""TensorFlow logits/gradients for the pinned upstream AutoPGD optimizer.

No optimizer implementation is copied from AutoAttack. Its CPU PyTorch arrays
are an interface; differentiation of the classifier stays in TensorFlow.
"""
from dataclasses import dataclass
import numpy as np
import tensorflow as tf
import torch


def margin(logits: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
    mask = tf.one_hot(labels, tf.shape(logits)[-1], on_value=True, off_value=False)
    other = tf.reduce_max(tf.where(mask, tf.cast(-np.inf, logits.dtype), logits), axis=-1)
    return tf.gather(logits, labels, axis=1, batch_dims=1) - other


class _StopBlock(Exception):
    pass


@dataclass
class BlockResult:
    candidate: np.ndarray
    margin: float
    success: bool
    forwards: int
    backwards: int


class _Meter:
    def __init__(self, original: np.ndarray, label: int, limit: int):
        self.best = original.copy()
        self.label, self.limit = label, limit
        self.forwards = self.backwards = 0
        self.best_margin = float("inf")
        self.success = False

    def before(self, backward: bool) -> None:
        if self.forwards >= self.limit:
            raise _StopBlock
        self.forwards += 1
        self.backwards += int(backward)

    def observe(self, image: np.ndarray, value: float, prediction: int) -> None:
        if value < self.best_margin:
            self.best_margin, self.best = value, image.copy()
        if prediction != self.label:
            if self.forwards == 1:
                raise RuntimeError("Model changed its clean prediction; use deterministic inference")
            self.best, self.success = image.copy(), True
            raise _StopBlock


class TensorFlowBackend:
    def __init__(self, model: tf.keras.Model):
        self.model = model
        self.forward = tf.function(lambda x: model(x, training=False), reduce_retracing=True)
        self.ce = tf.function(lambda x, y: self._gradient(x, y, "ce"), reduce_retracing=True)
        self.dlr = tf.function(lambda x, y: self._gradient(x, y, "dlr"), reduce_retracing=True)
        self.meter: _Meter | None = None

    def _gradient(self, x: tf.Tensor, labels: tf.Tensor, loss: str):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(x)
            logits = self.model(x, training=False)
            if loss == "ce":
                values = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
            else:
                ordered = tf.sort(logits, axis=1)
                values = -margin(logits, labels) / (ordered[:, -1] - ordered[:, -3] + 1e-12)
            total = tf.reduce_sum(values)
        gradient = tape.gradient(total, x)
        if gradient is None:
            raise ValueError("Classifier output is not differentiable with respect to its input")
        return logits, values, gradient

    def logits(self, x: np.ndarray) -> np.ndarray:
        logits = self.forward(tf.constant(x)).numpy()
        if logits.ndim != 2 or not np.isfinite(logits).all():
            raise ValueError("Classifier must return finite [batch, classes] logits")
        return logits

    def warmup(self, x: np.ndarray, labels: np.ndarray) -> None:
        self.logits(x)
        for function in (self.ce, self.dlr):
            outputs = function(tf.constant(x), tf.constant(labels, tf.int32))
            if any(not np.isfinite(v.numpy()).all() for v in outputs):
                raise ValueError("Nonfinite model logits, loss or gradients")

    def _call(self, x, y=None, loss=None):
        if self.meter is None:
            raise RuntimeError("No active attack block")
        image = x.detach().cpu().numpy().transpose(0, 2, 3, 1).copy()
        self.meter.before(backward=loss is not None)
        if loss is None:
            logits = self.forward(tf.constant(image))
        else:
            logits, value, gradient = (self.ce if loss == "ce" else self.dlr)(
                tf.constant(image), tf.constant(y.cpu().numpy(), tf.int32))
            if not np.isfinite(gradient.numpy()).all() or not np.isfinite(value.numpy()).all():
                raise ValueError("Nonfinite classifier loss or input gradient")
        if not np.isfinite(logits.numpy()).all():
            raise ValueError("Nonfinite classifier logits")
        value_margin = margin(logits, tf.constant([self.meter.label], tf.int32))
        self.meter.observe(image, float(value_margin[0]), int(tf.argmax(logits[0])))
        result = torch.from_numpy(logits.numpy())
        if loss is None:
            return result
        return result, torch.from_numpy(value.numpy()), torch.from_numpy(
            gradient.numpy().transpose(0, 3, 1, 2).copy())

    def predict(self, x):
        return self._call(x)

    def get_logits_loss_grad_xent(self, x, y):
        return self._call(x, y, "ce")

    def get_logits_loss_grad_dlr(self, x, y):
        return self._call(x, y, "dlr")

    def block(self, original, label, initial, epsilon, steps, seed, loss) -> BlockResult:
        try:
            from autoattack.autopgd_base import APGDAttack
        except ImportError as exc:
            raise ImportError("Install FARO with its pinned AutoAttack dependency: pip install .") from exc
        self.meter = _Meter(original, label, steps + 2)
        optimizer = APGDAttack(self, n_iter=steps, n_restarts=1, norm="Linf", eps=epsilon,
                              seed=seed, loss=loss, device="cpu", is_tf_model=True)
        x = torch.from_numpy(original.transpose(0, 3, 1, 2).copy())
        y = torch.tensor([label])
        start = torch.from_numpy(initial.transpose(0, 3, 1, 2).copy())
        try:
            self.predict(x)
            # Preserve the caller's CPU Torch RNG rather than altering global state.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed)
                optimizer.init_hyperparam(x)
                optimizer.attack_single_run(x, y, x_init=start)
        except _StopBlock:
            pass
        meter = self.meter
        self.meter = None
        return BlockResult(meter.best, meter.best_margin, meter.success,
                           meter.forwards, meter.backwards)
