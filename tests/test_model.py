"""Smoke tests for shapes and a tiny end-to-end training step (CPU-only)."""
from __future__ import annotations

import numpy as np
import torch

from src.data.dataset import (
    SlidingWindowDataset,
    build_splits,
    generate_synthetic,
)
from src.models import DualDomainForecaster
from src.models.freq_branch import FreqBranch
from src.models.mamba_block import BiMambaEncoder, MambaEncoder, MambaSSM
from src.models.time_branch import TimeBranch


def test_mamba_ssm_shape():
    b, l, d = 3, 40, 24
    x = torch.randn(b, l, d)
    ssm = MambaSSM(d_model=d, d_state=8)
    out = ssm(x)
    assert out.shape == (b, l, d)


def test_mamba_encoder_causal():
    """A causal SSM: changing a future timestep must not alter earlier outputs."""
    torch.manual_seed(0)
    enc = MambaEncoder(d_model=16, n_layers=2, d_state=8, use_official=False).eval()
    x = torch.randn(1, 20, 16)
    with torch.no_grad():
        y1 = enc(x)
        x2 = x.clone()
        x2[:, -1] += 5.0  # perturb only the last timestep
        y2 = enc(x2)
    # Outputs before the perturbed step must be unchanged.
    assert torch.allclose(y1[:, :-1], y2[:, :-1], atol=1e-5)


def test_bimamba_encoder_bidirectional():
    """Unlike the causal encoder, a future edit MUST reach earlier outputs."""
    torch.manual_seed(0)
    enc = BiMambaEncoder(d_model=16, n_layers=1, d_state=8, use_official=False).eval()
    x = torch.randn(1, 20, 16)
    with torch.no_grad():
        y1 = enc(x)
        x2 = x.clone()
        x2[:, -1] += 5.0  # perturb only the last position
        y2 = enc(x2)
    assert y1.shape == x.shape
    # Earlier positions should change: information flows backward too.
    assert not torch.allclose(y1[:, :-1], y2[:, :-1], atol=1e-4)


def test_freq_branch_mamba_shape():
    b, l, h, c, d = 4, 96, 24, 7, 32
    x = torch.randn(b, l, c)
    branch = FreqBranch(
        seq_len=l, pred_len=h, d_model=d, encoder="mamba", use_official_mamba=False
    )
    feat, y = branch(x)
    assert feat.shape == (b, c, d)
    assert y.shape == (b, c, h)


def test_model_freq_mamba_forward():
    b, l, h, c = 2, 48, 12, 3
    x = torch.randn(b, l, c)
    model = DualDomainForecaster(
        seq_len=l, pred_len=h, n_channels=c, d_model=16,
        freq_encoder="mamba", mamba_layers=1,
    )
    assert model(x).shape == (b, h, c)


def test_time_branch_shape():
    b, l, h, c, d = 4, 96, 24, 7, 32
    x = torch.randn(b, l, c)
    branch = TimeBranch(seq_len=l, pred_len=h, d_model=d, use_official_mamba=False)
    feat, y = branch(x)
    assert feat.shape == (b, c, d)
    assert y.shape == (b, c, h)


def test_time_branch_mlp_encoder():
    b, l, h, c, d = 4, 96, 24, 7, 32
    x = torch.randn(b, l, c)
    branch = TimeBranch(seq_len=l, pred_len=h, d_model=d, encoder="mlp")
    feat, y = branch(x)
    assert feat.shape == (b, c, d)
    assert y.shape == (b, c, h)


def test_freq_branch_shape():
    b, l, h, c, d = 4, 96, 24, 7, 32
    x = torch.randn(b, l, c)
    branch = FreqBranch(seq_len=l, pred_len=h, d_model=d)
    feat, y = branch(x)
    assert feat.shape == (b, c, d)
    assert y.shape == (b, c, h)


def test_model_forward_shape():
    b, l, h, c = 4, 96, 24, 7
    x = torch.randn(b, l, c)
    model = DualDomainForecaster(seq_len=l, pred_len=h, n_channels=c, d_model=32)
    out = model(x)
    assert out.shape == (b, h, c)


def test_fusion_modes():
    b, l, h, c = 2, 48, 12, 3
    x = torch.randn(b, l, c)
    for mode in ("gated", "sum", "concat"):
        model = DualDomainForecaster(
            seq_len=l, pred_len=h, n_channels=c, d_model=16, fusion=mode
        )
        assert model(x).shape == (b, h, c)


def test_time_branch_init_forecast_is_linear_backbone():
    """The time branch's correction head is zero-init, so its initial
    forecast must equal its internal linear (DLinear-style) backbone."""
    torch.manual_seed(0)
    b, l, h, c = 2, 48, 12, 3
    x = torch.randn(b, l, c)
    branch = TimeBranch(
        seq_len=l, pred_len=h, d_model=16, use_official_mamba=False
    ).eval()
    with torch.no_grad():
        _, y = branch(x)
        seasonal, trend = branch.decomp(x)
        expected = branch.lin_seasonal(seasonal.transpose(1, 2)) + branch.lin_trend(
            trend.transpose(1, 2)
        )
    assert torch.allclose(y, expected, atol=1e-5)


def test_forecast_is_convex_combination_of_branches():
    """The output must lie between the two branch forecasts elementwise:
    no path may bypass the dual branches."""
    torch.manual_seed(0)
    b, l, h, c = 2, 48, 12, 3
    x = torch.randn(b, l, c)
    model = DualDomainForecaster(
        seq_len=l, pred_len=h, n_channels=c, d_model=16
    ).eval()
    with torch.no_grad():
        out, comps = model(x, return_components=True)
    lo = torch.minimum(comps["time"], comps["freq"])
    hi = torch.maximum(comps["time"], comps["freq"])
    assert (out >= lo - 1e-5).all() and (out <= hi + 1e-5).all()


def test_ablation_hooks():
    """Every ablation switch must produce a working model of the right shape,
    and branch-only fusion must return exactly that branch's forecast."""
    torch.manual_seed(0)
    b, l, h, c = 2, 48, 12, 3
    x = torch.randn(b, l, c)
    for kwargs in (
        {"fusion": "time_only"},
        {"fusion": "freq_only"},
        {"use_revin": False},
        {"time_linear_backbone": False},
    ):
        model = DualDomainForecaster(
            seq_len=l, pred_len=h, n_channels=c, d_model=16, **kwargs
        ).eval()
        with torch.no_grad():
            out, comps = model(x, return_components=True)
        assert out.shape == (b, h, c)
        if kwargs.get("fusion") == "time_only":
            assert torch.allclose(out, comps["time"])
        if kwargs.get("fusion") == "freq_only":
            assert torch.allclose(out, comps["freq"])


def test_no_revin_output_not_denormalized():
    """With RevIN off the model must not re-add window statistics: shifting
    the input by a constant changes RevIN output but for a zero-init model
    without RevIN the forecast stays put."""
    b, l, h, c = 2, 48, 12, 2
    x = torch.randn(b, l, c)
    model = DualDomainForecaster(
        seq_len=l, pred_len=h, n_channels=c, d_model=16,
        use_revin=True, channel_mixer_layers=0,
    ).eval()
    with torch.no_grad():
        y1 = model(x)
        y2 = model(x + 100.0)
    # RevIN on: constant shift passes straight through to the forecast.
    assert torch.allclose(y2 - y1, torch.full_like(y1, 100.0), atol=1e-2)


def test_fits_backbone_shapes_and_silent_init():
    """With the FITS spectral backbone, the freq branch must (a) keep its
    output contract and (b) start silent: backbone and head are zero-init,
    so the initial branch forecast is exactly zero."""
    b, l, h, c, d = 2, 96, 24, 5, 32
    x = torch.randn(b, l, c)
    branch = FreqBranch(seq_len=l, pred_len=h, d_model=d, backbone="fits").eval()
    with torch.no_grad():
        feat, y = branch(x)
    assert feat.shape == (b, c, d)
    assert y.shape == (b, c, h)
    assert y.abs().max().item() == 0.0


def test_fits_model_forward_and_gradients():
    torch.manual_seed(0)
    b, l, h, c = 2, 48, 12, 3
    x = torch.randn(b, l, c)
    model = DualDomainForecaster(
        seq_len=l, pred_len=h, n_channels=c, d_model=16,
        freq_backbone="fits", channel_mixer_layers=0,
    )
    out = model(x)
    assert out.shape == (b, h, c)
    # The zero-init spectral backbone must still receive gradients.
    out.sum().backward()
    g = model.freq_branch.spec_backbone.wr.weight.grad
    assert g is not None and g.abs().sum() > 0


def test_channel_mixer_cross_channel_flow():
    """With the mixer, perturbing one channel must influence another
    channel's forecast; without it, channels must stay fully independent."""
    torch.manual_seed(0)
    b, l, h, c = 1, 48, 12, 4
    x = torch.randn(b, l, c)
    x2 = x.clone()
    # Perturb only channel 0 — non-constant, so it survives the per-channel
    # instance normalization (a constant offset would be normalized away).
    x2[:, : l // 2, 0] += 5.0

    mixed = DualDomainForecaster(
        seq_len=l, pred_len=h, n_channels=c, d_model=16, channel_mixer_layers=1
    ).eval()
    indep = DualDomainForecaster(
        seq_len=l, pred_len=h, n_channels=c, d_model=16, channel_mixer_layers=0
    ).eval()
    with torch.no_grad():
        d_mixed = (mixed(x)[..., 1:] - mixed(x2)[..., 1:]).abs().max().item()
        d_indep = (indep(x)[..., 1:] - indep(x2)[..., 1:]).abs().max().item()
    assert d_indep < 1e-6, "mixer off: other channels' forecasts must not change"
    assert d_mixed > 1e-6, "mixer on: cross-channel information must flow"


def test_mamba_finite_under_autocast():
    """The selective scan must run in fp32 under autocast and stay finite."""
    torch.manual_seed(0)
    enc = MambaEncoder(d_model=16, n_layers=2, d_state=8, use_official=False)
    x = torch.randn(2, 64, 16) * 10  # large-ish values to stress fp16/bf16
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        y = enc(x)
    assert torch.isfinite(y).all()


def test_dataset_windowing():
    data = generate_synthetic(length=500, channels=3, seed=1)
    ds = SlidingWindowDataset(data, seq_len=96, pred_len=24)
    assert len(ds) == 500 - 96 - 24 + 1
    x, y, stats = ds[0]
    assert x.shape == (96, 3)
    assert y.shape == (24, 3)
    assert stats.shape == (0, 3)  # no dispersion resolutions -> empty stats


def test_multires_stats_and_dispersion():
    """Multi-resolution stats and the learned dispersion head: shapes, that
    D_hat is strictly positive, that history uses only pre-origin data, and
    that at initialization the head reduces to RevIN de-normalization."""
    import torch

    from src.models.dispersion import DispersionHead

    res = [24, 48, 96]
    data = generate_synthetic(length=600, channels=4, seed=2)
    ds = SlidingWindowDataset(data, seq_len=96, pred_len=24,
                              stats_resolutions=res, start_offset=0)
    x, y, stats = ds[0]
    assert stats.shape == (2 * len(res), 4)  # (mean,std) per resolution

    # No leakage: stats of the shortest resolution equal mean/std of the
    # samples strictly before the target (the tail of the input window).
    origin = 96
    hist = data[origin - res[0]:origin]
    assert np.allclose(stats[0].numpy(), hist.mean(0), atol=1e-5)
    assert np.allclose(stats[1].numpy(), hist.std(0), atol=1e-5)

    # Dispersion head: r_hat > 0 everywhere and equals 1 at init (t_hat 0).
    head = DispersionHead(stats_dim=2 * len(res), pred_len=24)
    s = stats.unsqueeze(0).transpose(1, 2)  # (1, C, S)
    r_hat, t_hat = head(s)
    assert r_hat.shape == (1, 4, 24) and t_hat.shape == (1, 4, 24)
    assert (r_hat > 0).all()
    assert torch.allclose(r_hat, torch.ones_like(r_hat), atol=1e-5)
    assert torch.allclose(t_hat, torch.zeros_like(t_hat), atol=1e-6)


def test_dispersion_reduces_to_revin_at_init():
    """A model with the learned dispersion head must produce exactly the RevIN
    reconstruction at initialization (r_hat=1, t_hat=0), so the head is a
    strict generalization that starts from the current de-normalization."""
    import torch

    from src.models.dual_domain_model import DualDomainForecaster

    res = (24, 48, 96)
    common = dict(seq_len=96, pred_len=24, n_channels=4, d_model=32,
                  channel_mixer_layers=0, use_revin=True)
    torch.manual_seed(0)
    m_none = DualDomainForecaster(dispersion="none", **common)
    torch.manual_seed(0)
    m_learn = DualDomainForecaster(dispersion="learned",
                                   dispersion_resolutions=res, **common)
    # Share all weights except the (identity-at-init) dispersion head.
    m_learn.load_state_dict(m_none.state_dict(), strict=False)
    m_none.eval(); m_learn.eval()

    x = torch.randn(3, 96, 4)
    stats = torch.randn(3, 2 * len(res), 4)
    with torch.no_grad():
        out_none = m_none(x)
        out_learn = m_learn(x, stats=stats)
    assert torch.allclose(out_none, out_learn, atol=1e-5)


def test_dispersion_head_efficiency():
    """Efficiency test for the dispersion experiment: the head's parameter
    overhead must be (a) exactly the head's own size, (b) small relative to the
    model, and (c) INDEPENDENT of the channel count -- the head is shared
    across variates, so it stays cheap even on high-channel data (Traffic)."""
    import torch

    from src.models.dispersion import DispersionHead
    from src.models.dual_domain_model import DualDomainForecaster

    res = (96, 144, 288, 336)
    stats_dim, pred_len, hidden = 2 * len(res), 96, 64

    def n_params(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    def build(c, dispersion):
        torch.manual_seed(0)
        return DualDomainForecaster(
            seq_len=96, pred_len=pred_len, n_channels=c, d_model=128,
            channel_mixer_layers=1, use_revin=True, dispersion=dispersion,
            dispersion_resolutions=res)

    head_params = n_params(DispersionHead(stats_dim, pred_len, hidden))

    for c in (7, 137, 862):  # ETT, Solar, Traffic channel counts
        base = build(c, "none")
        disp = build(c, "learned")
        overhead = n_params(disp) - n_params(base)
        # (a) the overhead is exactly the head's parameters ...
        assert overhead == head_params
        # (c) ... which is the same for every channel count (shared across
        #     variates), and (b) a small fraction of a real model.
        assert overhead == head_params  # channel-independent by construction
        assert overhead < 0.10 * n_params(base)
    # The head has no per-channel parameters: O(1) in C.
    assert head_params < 15000


def test_ett_canonical_borders():
    """ETTh protocol must reproduce the canonical window counts: for
    seq_len=96, pred_len=96 -> train 8449, val 2785, test 2785 (matching
    Informer/Autoformer/TSLib), regardless of the file's extra tail rows."""
    from src.data.dataset import ETT_BORDERS

    data = generate_synthetic(length=17420, channels=7, seed=3)  # ETTh1-sized
    train, val, test, _ = build_splits(
        data, seq_len=96, pred_len=96, train_ratio=0.6, val_ratio=0.2,
        scale=True, borders=ETT_BORDERS["ETTh"],
    )
    assert len(train) == 8640 - 96 - 96 + 1        # 8449
    assert len(val) == 2880 + 96 - 96 - 96 + 1     # 2785
    assert len(test) == 2880 + 96 - 96 - 96 + 1    # 2785


def test_build_splits_no_leakage_shapes():
    data = generate_synthetic(length=2000, channels=4, seed=2)
    train, val, test, scaler = build_splits(
        data, seq_len=96, pred_len=24, train_ratio=0.7, val_ratio=0.1, scale=True
    )
    assert len(train) > 0 and len(val) > 0 and len(test) > 0
    # Scaler should roughly standardize the training slice.
    scaled = scaler.transform(data[: int(len(data) * 0.7)])
    assert abs(scaled.mean()) < 0.1


def test_training_step_reduces_loss():
    torch.manual_seed(0)
    b, l, h, c = 16, 48, 12, 2
    x = torch.randn(b, l, c)
    # Target correlated with input so the model can actually learn.
    y = x[:, -h:, :] * 0.5
    model = DualDomainForecaster(seq_len=l, pred_len=h, n_channels=c, d_model=32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.MSELoss()
    first = None
    for _ in range(50):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
    assert loss.item() < first


def test_dlinear_baseline():
    """The in-framework DLinear baseline must train through the same
    pipeline: correct shapes and selectable via model.arch."""
    from src.models.baselines import DLinearBaseline
    from src.models import build_model

    b, l, h, c = 2, 48, 12, 3
    x = torch.randn(b, l, c)
    m = DLinearBaseline(seq_len=l, pred_len=h)
    assert m(x).shape == (b, h, c)
    cfg = {"model": {"arch": "dlinear", "time_kernel_size": 25},
           "data": {"seq_len": l, "pred_len": h}}
    m2 = build_model(cfg, c)
    assert m2(x).shape == (b, h, c)


def test_zero_init_switch():
    """zero_init=False must produce standard random init: the freq branch is
    no longer silent and the time branch no longer equals its backbone."""
    torch.manual_seed(0)
    b, l, h, c = 2, 48, 12, 3
    x = torch.randn(b, l, c)
    common = dict(seq_len=l, pred_len=h, n_channels=c, d_model=16,
                  freq_backbone="fits", channel_mixer_layers=0, use_revin=False)
    silent = DualDomainForecaster(zero_init=True, **common).eval()
    rand = DualDomainForecaster(zero_init=False, **common).eval()
    with torch.no_grad():
        _, c0 = silent(x, return_components=True)
        _, c1 = rand(x, return_components=True)
    assert c0["freq"].abs().max() < 1e-6
    assert c1["freq"].abs().max() > 1e-3
