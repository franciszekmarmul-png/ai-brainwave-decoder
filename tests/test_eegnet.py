import pytest
import torch

from src.eegnet import EEGNet


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def test_parameter_count_matches_paper_table_3():
    assert n_params(EEGNet()) == 1716


def test_smaller_variant_parameter_count_matches_paper_table_3():
    assert n_params(EEGNet(f1=4, d=2, f2=8)) == 796


def test_output_shape():
    assert EEGNet()(torch.zeros(5, 22, 256)).shape == (5, 4)


def test_output_layer_follows_trial_length():
    assert EEGNet(n_samples=128)(torch.zeros(2, 22, 128)).shape == (2, 4)


def test_max_norm_limits_spatial_and_output_layers():
    m = EEGNet()
    with torch.no_grad():
        for p in m.parameters():
            p.mul_(100)
    m.apply_max_norm()
    assert m.spatial.weight.flatten(1).norm(dim=1).max() <= 1 + 1e-5
    assert m.classifier.weight.norm(dim=1).max() <= 0.25 + 1e-5


def test_max_norm_leaves_small_weights_alone():
    m = EEGNet()
    with torch.no_grad():
        m.spatial.weight.mul_(0).add_(0.01)
        m.classifier.weight.mul_(0).add_(0.001)
    spatial, classifier = m.spatial.weight.clone(), m.classifier.weight.clone()
    m.apply_max_norm()
    assert torch.equal(m.spatial.weight, spatial)
    assert torch.equal(m.classifier.weight, classifier)


def test_max_norm_limits_each_filter_separately():
    m = EEGNet()
    with torch.no_grad():
        m.spatial.weight.mul_(0).add_(0.01)
        m.spatial.weight[3] = 5.0
    small = m.spatial.weight[0].clone()
    m.apply_max_norm()
    assert m.spatial.weight[3].norm().item() == pytest.approx(1.0, abs=1e-5)
    assert torch.equal(m.spatial.weight[0], small)


def test_dropout_setting_is_used_in_training_and_off_in_eval():
    torch.manual_seed(0)
    x = torch.randn(8, 22, 256)
    m = EEGNet(dropout=0.5)
    assert not torch.equal(m(x), m(x))
    m.eval()
    assert torch.equal(m(x), m(x))
    no_dropout = EEGNet(dropout=0.0)
    assert torch.equal(no_dropout(x), no_dropout(x))


def test_padding_keeps_trial_length_on_both_convolutions():
    m = EEGNet()
    x = torch.zeros(2, 1, 22, 256)
    assert m.temporal_pad(x).shape[-1] == 256 + 31
    assert m.temporal(m.temporal_pad(x)).shape[-1] == 256
    pooled = torch.zeros(2, 16, 1, 64)
    assert m.separable_depthwise(m.separable_pad(pooled)).shape[-1] == 64


def test_no_warnings_from_a_forward_pass(recwarn):
    EEGNet()(torch.zeros(2, 22, 256))
    assert len(recwarn) == 0
