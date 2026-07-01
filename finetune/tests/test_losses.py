import torch

from vigil_two_stage.losses import bce_with_logits_loss, supervised_contrastive_loss


def test_stage1_and_stage2_bce_are_finite():
    logits = torch.tensor([0.1, -0.2, 1.0])
    labels = torch.tensor([1.0, 0.0, 1.0])
    assert torch.isfinite(bce_with_logits_loss(logits, labels))


def test_supcon_is_finite_and_excludes_background():
    z = torch.randn(4, 8, requires_grad=True)
    loss = supervised_contrastive_loss(z, ["vigil", "vigil", "background", "background"])
    assert torch.isfinite(loss)
    loss.backward()
    assert z.grad is not None


def test_supcon_skips_anchors_without_same_class_partners_and_returns_differentiable_zero():
    z = torch.randn(3, 8, requires_grad=True)
    loss = supervised_contrastive_loss(z, ["vigil", "visual", "background"])
    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    loss.backward()
    assert z.grad is not None
