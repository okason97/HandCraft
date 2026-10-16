from types import SimpleNamespace

import pytest
import torch

import models.argen_mlp as argen_mlp
import models.argen_tf as argen_tf
import utils.ops as ops

DATA = SimpleNamespace(num_keypoints=29, coords=2, num_classes=5, pred_frames=3, max_gen_len=40)
RUN = SimpleNamespace(mixed_precision=False)


def build(kind):
    torch.manual_seed(0)
    if kind == "tf":
        model = argen_tf.Model(DATA=DATA, RUN=RUN, MODULES=None, MODEL=SimpleNamespace(hidden_dim=32, nheads=4, mlp_dim=64, depth=2, dropout=0.0))
    else:
        MODULES = SimpleNamespace(glinear=ops.Temporal_FC, feature_norm=ops.SLayerNorm)
        MODEL = SimpleNamespace(hidden_dim=32, depth=2, context_frames=4, class_dropout_prob=0.0, noise_scale=0.0)
        model = argen_mlp.Model(DATA=DATA, RUN=RUN, MODULES=MODULES, MODEL=MODEL)
    # the drift head starts at zero; random weights make the outputs depend on the input for these tests
    torch.nn.init.normal_(model.drift_head.weight)
    return model.eval()


@pytest.mark.parametrize("kind", ["tf", "mlp"])
def test_shapes_and_last_only(kind):
    model = build(kind)
    frames, labels = torch.randn(2, 10, 58), torch.tensor([0, 4])
    drift, end = model(frames, labels)
    assert drift.shape == (2, 10, 3, 58) and end.shape == (2, 10)
    last_drift, last_end = model(frames, labels, last_only=True)
    torch.testing.assert_close(last_drift[:, 0], drift[:, -1], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(last_end[:, 0], end[:, -1], rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("kind", ["tf", "mlp"])
def test_drift_starts_at_zero(kind):
    model = build(kind)
    torch.nn.init.zeros_(model.drift_head.weight)
    drift, _ = model(torch.randn(1, 6, 58), torch.tensor([1]))
    assert not drift.any()


def test_transformer_is_causal():
    model = build("tf")
    frames, labels = torch.randn(1, 10, 58), torch.tensor([2])
    changed = frames.clone()
    changed[:, 6:] += 1.0
    drift, _ = model(frames, labels)
    drift2, _ = model(changed, labels)
    torch.testing.assert_close(drift[:, :6], drift2[:, :6])
    assert not torch.allclose(drift[:, 6:], drift2[:, 6:])


def test_mlp_sees_only_its_window():
    model = build("mlp")  # window of 4 frames
    frames, labels = torch.randn(1, 10, 58), torch.tensor([2])
    drift, _ = model(frames, labels)
    before = frames.clone()
    before[:, 3] += 1.0  # outside the window of position 7 (frames 4 to 7)
    torch.testing.assert_close(model(before, labels)[0][:, 7], drift[:, 7])
    inside = frames.clone()
    inside[:, 4] += 1.0
    assert not torch.allclose(model(inside, labels)[0][:, 7], drift[:, 7])


@pytest.mark.parametrize("kind", ["tf", "mlp"])
def test_generation_noise_changes_the_output_only_when_set(kind):
    model = build(kind)
    frames, labels = torch.randn(1, 6, 58), torch.tensor([3])
    torch.testing.assert_close(model(frames, labels)[0], model(frames, labels)[0])
    model.generation_noise = 0.5
    assert not torch.allclose(model(frames, labels)[0], model(frames, labels)[0])
