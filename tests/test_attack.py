import json
import numpy as np
import pytest
import tensorflow as tf
import torch
from faro import AttackConfig, FaroAttack
from faro._backend import BlockResult, TensorFlowBackend, margin


def linear_classifier():
    inputs = tf.keras.Input((2, 2, 1))
    logits = tf.keras.layers.Dense(3)(tf.keras.layers.Flatten()(inputs))
    model = tf.keras.Model(inputs, logits)
    weights = np.tile(np.array([[.25, -.25, 0]], np.float32), (4, 1))
    model.layers[-1].set_weights([weights, np.array([0., 1., -1.], np.float32)])
    return model


def test_success_constraints_weights_inputs_rng_and_serialization(tmp_path):
    model = linear_classifier()
    x = np.full((1, 2, 2, 1), .6, np.float32)
    before = [w.copy() for w in model.get_weights()]
    attack = FaroAttack(model, AttackConfig(epsilon=.2, steps_per_block=10, blocks=2, seed=11))
    attack.warmup(x, [0])
    rng = torch.random.get_rng_state().clone()
    result = attack.generate(x, [0])
    assert torch.equal(rng, torch.random.get_rng_state())
    assert result.success.tolist() == [True]
    assert int(model(result.adversarial).numpy().argmax()) == 1
    assert result.samples[0].linf <= .2 + 5e-7
    assert result.samples[0].verification_forwards == 1
    assert result.samples[0].total_forwards == result.samples[0].search_forwards + 1
    np.testing.assert_array_equal(x, np.full_like(x, .6))
    for expected, actual in zip(before, model.get_weights()):
        np.testing.assert_array_equal(expected, actual)
    result.save(tmp_path / "run")
    report = json.loads((tmp_path / "run/results.json").read_text())
    assert report["summary"]["successes"] == 1
    with np.load(tmp_path / "run/arrays.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["adversarial"], result.adversarial)
    with pytest.raises(FileExistsError):
        result.save(tmp_path / "run")


def test_robust_case_finishes_schedule_beyond_old_61_limit():
    model = linear_classifier()
    attack = FaroAttack(model, AttackConfig(epsilon=.01, steps_per_block=100, blocks=2))
    result = attack.generate(np.full((1, 2, 2, 1), .9, np.float32), [0])
    sample = result.samples[0]
    assert sample.status == "not_found"
    assert sample.blocks_executed == 2
    assert sample.search_forwards == 205  # clean + two (100 iterations + 2 setup)
    assert sample.backwards == 202
    assert not sample.success


def test_mixed_batch_and_explicit_seed_reproducibility():
    model = linear_classifier()
    x = np.stack([np.full((2, 2, 1), .6), np.full((2, 2, 1), .1)]).astype("float32")
    attack = FaroAttack(model, AttackConfig(epsilon=.2, steps_per_block=10, blocks=2))
    together = attack.generate(x, [0, 0], seeds=[4, 9])
    single = attack.generate(x[:1], [0], seeds=[4])
    np.testing.assert_array_equal(together.adversarial[:1], single.adversarial)
    assert together.samples[1].status == "already_misclassified"
    assert not together.samples[1].success
    assert together.samples[1].search_forwards == 1
    assert together.samples[1].backwards == 0
    assert together.success_rate == 1.


def test_zero_epsilon_and_no_eligible_inputs():
    attack = FaroAttack(linear_classifier(), AttackConfig(epsilon=0))
    x = np.full((1, 2, 2, 1), .6, np.float32)
    result = attack.generate(x, [0])
    np.testing.assert_array_equal(result.adversarial, x)
    assert not result.success[0] and result.samples[0].blocks_executed == 0
    ineligible = attack.generate(x, [1])
    assert np.isnan(ineligible.success_rate)
    assert ineligible.to_dict()["summary"]["success_rate_on_clean_correct"] is None


@pytest.mark.parametrize("kwargs", [{"epsilon": -1}, {"epsilon": float("nan")},
                                   {"epsilon": True}, {"epsilon": .1, "blocks": 0},
                                   {"epsilon": .1, "seed": -1}])
def test_bad_configuration(kwargs):
    with pytest.raises(ValueError):
        AttackConfig(**kwargs)


def test_bad_inputs_and_probability_outputs():
    model = linear_classifier()
    attack = FaroAttack(model, AttackConfig(epsilon=.1))
    for x, y in [(np.ones((1,2,2,1), np.uint8), [0]),
                 (np.full((1,2,2,1), np.nan), [0]),
                 (np.ones((1,2,2,1)), [3]), (np.ones((1,2,2,1)), [[0]])]:
        with pytest.raises(ValueError):
            attack.generate(x, y)
    probabilities = tf.keras.Model(model.input, tf.keras.layers.Softmax()(model.output))
    with pytest.raises(ValueError, match="logits"):
        FaroAttack(probabilities, AttackConfig(epsilon=.1))


def test_dlr_and_ce_gradients_against_direct_tensorflow():
    model = linear_classifier()
    backend = TensorFlowBackend(model)
    x = tf.constant(np.full((1,2,2,1), .6, np.float32)); y = tf.constant([0])
    for loss, function in [("ce", backend.ce), ("dlr", backend.dlr)]:
        with tf.GradientTape() as tape:
            tape.watch(x)
            logits = model(x)
            ordered = tf.sort(logits, axis=1)
            value = (tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
                     if loss == "ce" else -margin(logits, y)/(ordered[:,-1]-ordered[:,-3]+1e-12))
            total = tf.reduce_sum(value)
        expected = tape.gradient(total, x)
        _, actual_value, actual = function(x, y)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(actual_value, value, rtol=1e-6)


@pytest.mark.parametrize("outside_ball", [False, True])
def test_final_verification_rejects_invalid_optimizer_claim(monkeypatch, outside_ball):
    attack = FaroAttack(linear_classifier(), AttackConfig(epsilon=.1))
    x = np.full((1,2,2,1), .6, np.float32)
    candidate = np.ones_like(x) if outside_ball else x.copy()
    monkeypatch.setattr(attack._backend, "block", lambda *args, **kwargs:
                        BlockResult(candidate, -1., True, 2, 1))
    with pytest.raises(RuntimeError, match="threat set" if outside_ball else "disagrees"):
        attack.generate(x, [0])
