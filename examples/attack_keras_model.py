"""Self-contained API smoke example; requires no dataset downloads or training.

Run: python examples/attack_keras_model.py
The tiny deterministic model demonstrates the interface, not attack performance.
"""
import numpy as np
import tensorflow as tf
from faro import AttackConfig, FaroAttack


def main():
    inputs = tf.keras.Input((2, 2, 1))
    logits = tf.keras.layers.Dense(3)(tf.keras.layers.Flatten()(inputs))
    model = tf.keras.Model(inputs, logits)
    model.layers[-1].set_weights([
        np.tile(np.array([[.25, -.25, 0]], np.float32), (4, 1)),
        np.array([0., 1., -1.], np.float32),
    ])
    images = np.full((1, 2, 2, 1), .6, np.float32)
    labels = np.array([0])
    attacker = FaroAttack(model, AttackConfig(epsilon=.2))
    attacker.warmup(images, labels)
    result = attacker.generate(images, labels)
    print(result.to_dict()["summary"])
    print("Clean prediction:", result.samples[0].clean_prediction)
    print("Adversarial prediction:", result.samples[0].adversarial_prediction)


if __name__ == "__main__":
    main()
