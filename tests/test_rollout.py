import pytest
import torch
from torch import nn

from generation.autoregressive import generate, random_cut, rollout, rollout_metrics, usable_openings


class ConstantDrift(nn.Module):
    """Predicts that every value grows by `step` per frame, with a constant end logit; records its inputs."""

    def __init__(self, pred_frames, step=1.0, end=-5.0, value=None):
        super().__init__()
        self.pred_frames, self.step, self.end, self.value, self.calls = pred_frames, step, end, value, []

    def forward(self, frames, labels, last_only=False):
        self.calls.append(frames.clone())
        B, T, features = frames.shape
        L = 1 if last_only else T
        drift = self.step * torch.arange(1, self.pred_frames + 1, dtype=frames.dtype).view(1, 1, -1, 1).expand(B, L, -1, features)
        if self.value is not None:
            drift = torch.full_like(drift, self.value)
        return drift, torch.full((B, L), self.end)


def ramp(n, features=4):
    return torch.arange(n, dtype=torch.float32)[None, :, None].expand(1, n, features).clone()


def test_rollout_feeds_its_own_frames_back():
    model = ConstantDrift(pred_frames=2)
    context = ramp(3)
    new, ends = rollout(model, context, torch.tensor([0]), steps=2)
    assert new[0, :, 0].tolist() == [3.0, 4.0, 5.0, 6.0]
    assert ends.shape == (1, 2)
    torch.testing.assert_close(model.calls[1][:, :3], context)
    torch.testing.assert_close(model.calls[1][:, 3:], new[:, :2])


def test_random_cut_stays_inside_the_shortest_clip():
    for _ in range(50):
        cut = random_cut(torch.tensor([20, 6]), opening_frames=8)
        assert 5 <= cut <= 5
    with pytest.raises(ValueError):
        random_cut(torch.tensor([1, 6]), opening_frames=8)


def test_rollout_metrics_masks_short_and_padded_clips():
    frames = torch.zeros(3, 12, 4)
    lengths = torch.tensor([12, 9, 3])  # the third is not longer than the 4 opening frames
    for b, n in enumerate(lengths.tolist()):
        frames[b, :n] = ramp(n)[0]
    error, valid, correct, decided = rollout_metrics(ConstantDrift(pred_frames=2), frames, lengths, torch.zeros(3, dtype=torch.long), 4, 2, coords=2)
    assert error.shape == (3, 8)
    assert valid[0].all() and valid[1].tolist() == [True] * 5 + [False] * 3 and not valid[2].any()
    assert error[valid].abs().max() < 1e-6
    # the end logit is always negative: right on every step but the last of each clip
    assert decided == 4 + 3 and correct == decided - 2


def test_generate_stops_at_the_source_length():
    clips = generate(ConstantDrift(pred_frames=3), ramp(4).expand(2, 4, 4), torch.tensor([0, 1]), torch.tensor([9, 15]), "length", max_len=12)
    assert [len(c) for c in clips] == [9, 12]
    assert clips[0][:, 0].tolist() == list(range(9))


def test_generate_stops_at_the_end_signal_or_the_cap():
    opening = ramp(4)
    assert len(generate(ConstantDrift(pred_frames=3, end=5.0), opening, torch.tensor([0]), torch.tensor([30]), "end_signal", max_len=20)[0]) == 7
    assert len(generate(ConstantDrift(pred_frames=3, end=-5.0), opening, torch.tensor([0]), torch.tensor([30]), "end_signal", max_len=20)[0]) == 20


def test_generate_refuses_non_finite_frames():
    with pytest.raises(FloatingPointError):
        generate(ConstantDrift(pred_frames=2, value=float("nan")), ramp(4), torch.tensor([0]), torch.tensor([10]), "length", max_len=20)


def test_usable_openings_drops_short_clips_and_empty_classes():
    dataset = [(torch.zeros(n, 4), torch.tensor(c)) for n, c in [(5, 0), (10, 1), (3, 2), (4, 1)]]
    subset, labels, kept, dropped = usable_openings(dataset, [0, 1, 2, 1], [0, 1, 2], opening_frames=4)
    assert list(subset.indices) == [0, 1] and labels == [0, 1]
    assert kept == [0, 1] and dropped == [2]
