import torch

from generation.autoregressive import AutoregressiveLoss, joint_error, next_frame_targets, rollout_targets


def ramp(lengths, features=4):
    """Frames whose every value is the frame index, padded with 0 after each length."""
    frames = torch.zeros(len(lengths), max(lengths), features)
    for b, n in enumerate(lengths):
        frames[b, :n] = torch.arange(n, dtype=torch.float32)[:, None]
    return frames


def test_joint_error_is_the_mean_distance_over_joints():
    pred, target = torch.zeros(58), torch.zeros(58)
    pred[0], pred[1] = 3.0, 4.0
    assert torch.isclose(joint_error(pred, target), torch.tensor(5.0 / 29))


def test_next_frame_targets_and_masks():
    frames = ramp([5, 3])
    targets, frame_mask, position_mask, end = next_frame_targets(frames, torch.tensor([5, 3]), pred_frames=2)
    assert targets.shape == (2, 5, 2, 4)
    assert targets[0, 1, :, 0].tolist() == [2.0, 3.0]
    assert frame_mask[0, 3].tolist() == [True, False] and frame_mask[1, 1].tolist() == [True, False]
    assert position_mask[0].tolist() == [True, True, True, True, False]
    assert position_mask[1].tolist() == [True, True, False, False, False]
    # position t ends the clip when the clip's last frame is among its next 2 frames
    assert end[0].tolist() == [0.0, 0.0, 1.0, 1.0, 1.0]


def test_rollout_targets_follow_the_cut():
    frames = ramp([9, 6])
    targets, frame_mask, position_mask, end = rollout_targets(frames, torch.tensor([9, 6]), cut=3, steps=2, pred_frames=2)
    assert targets[0, :, :, 0].tolist() == [[3.0, 4.0], [5.0, 6.0]]
    assert frame_mask[1].tolist() == [[True, True], [True, False]]
    assert position_mask[1].tolist() == [True, True] and end[1].tolist() == [0.0, 1.0]


def test_loss_is_zero_for_a_perfect_prediction_and_ignores_masked_frames():
    loss = AutoregressiveLoss(coords=2, vel_weight=1.0, end_weight=1.0, end_pos_weight=1.0)
    target = torch.randn(2, 3, 2, 58)
    last = torch.randn(2, 3, 58)
    frame_mask = torch.ones(2, 3, 2, dtype=torch.bool)
    frame_mask[1, 2, 1] = False
    end = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    position_mask = torch.tensor([[True, True, True], [True, True, False]])
    end_logit = (end * 2 - 1) * 50
    total, position = loss(target.clone(), target, last, frame_mask, end_logit, end, position_mask)
    assert total < 1e-6 and position < 1e-6
    pred = target.clone()
    pred[1, 2, 1] += 100.0  # a masked frame
    masked_total, _ = loss(pred, target, last, frame_mask, end_logit, end, position_mask)
    # the velocity of the masked frame is masked too; the last valid frame is unchanged
    assert masked_total < 1e-6
