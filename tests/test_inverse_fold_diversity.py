import torch

from boltzgen.data import const
from boltzgen.model.modules.inverse_fold import (
    BG_FREQ,
    InverseFoldingDecoder,
    InverseFoldingEncoder,
)


def _encoder(backbone_noise: float) -> InverseFoldingEncoder:
    enc = InverseFoldingEncoder(
        atom_s=8,
        atom_z=8,
        token_s=8,
        token_z=8,
        node_dim=8,
        pair_dim=8,
        hidden_dim=8,
        num_encoder_layers=1,
        backbone_noise=backbone_noise,
    )
    enc.eval()
    return enc


def _geo_inputs(n: int = 4):
    n_atoms = n * 4
    t2a = torch.eye(n_atoms)[None]  # (1, N*4, A): one atom per bb4 slot
    coords = torch.randn(1, n_atoms, 3) * 5
    valid_mask = torch.ones(1, n, dtype=torch.bool)
    src, dst = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    edge_idx = torch.stack([src.flatten(), dst.flatten()])
    return {"token_to_bb4_atoms": t2a, "coords": coords}, edge_idx, valid_mask


def test_bg_freq_matches_canonical_tokens():
    assert BG_FREQ.shape[0] == len(const.canonical_tokens)
    assert abs(BG_FREQ.sum().item() - 1.0) < 0.02


def test_background_offset_favors_rare_residues():
    log_prior = torch.log(BG_FREQ)
    offset = log_prior - log_prior.mean()
    idx = const.canonical_tokens.index
    # mask -= offset: common residues are penalised, rare ones boosted
    assert offset[idx("LEU")] > 0 > offset[idx("TRP")]


def test_inference_backbone_noise_changes_geometry_features():
    feats, edge_idx, valid_mask = _geo_inputs()
    torch.manual_seed(0)
    clean = _encoder(0.0).extract_geo_feat(feats, edge_idx, valid_mask)
    noisy_a = _encoder(0.3).extract_geo_feat(feats, edge_idx, valid_mask)
    noisy_b = _encoder(0.3).extract_geo_feat(feats, edge_idx, valid_mask)
    assert not torch.allclose(clean, noisy_a)
    assert not torch.allclose(noisy_a, noisy_b)


def test_zero_noise_is_deterministic_at_inference():
    feats, edge_idx, valid_mask = _geo_inputs()
    enc = _encoder(0.0)
    a = enc.extract_geo_feat(feats, edge_idx, valid_mask)
    b = enc.extract_geo_feat(feats, edge_idx, valid_mask)
    assert torch.allclose(a, b)


def test_decoder_accepts_new_options():
    dec = InverseFoldingDecoder(
        atom_s=8,
        atom_z=8,
        token_s=8,
        token_z=8,
        node_dim=8,
        pair_dim=8,
        hidden_dim=8,
        num_decoder_layers=1,
        double_tape_polar_bias=1.0,
        background_unbias_strength=0.5,
        top_p=0.9,
        repetition_penalty=0.5,
    )
    assert dec.double_tape_polar_bias == 1.0
    assert dec.background_unbias_strength == 0.5
    assert dec.top_p == 0.9
    assert dec.repetition_penalty == 0.5
