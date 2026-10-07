from typing import Dict, Tuple, List, Optional
import warnings

import torch
from torch import Tensor, nn
from torch.nn.functional import one_hot

from boltzgen.data import const

# Replace torch_scatter with native PyTorch implementation
from boltzgen.model.modules.scatter_utils import scatter_sum, scatter_softmax
import torch.nn.functional as F

# Natural proteome background frequencies, ordered as const.canonical_tokens:
# ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL
BG_FREQ = torch.tensor(
    [
        0.0825, 0.0553, 0.0406, 0.0545, 0.0137, 0.0393, 0.0675, 0.0707, 0.0227, 0.0596,
        0.0966, 0.0584, 0.0242, 0.0386, 0.0470, 0.0656, 0.0534, 0.0108, 0.0292, 0.0687,
    ],
    dtype=torch.float32,
)



class GaussianSmearing(torch.nn.Module):
    # used to embed the edge distances
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = (dist.unsqueeze(-1) - self.offset.view(1, 1, -1)).flatten(start_dim=1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


def softmax_dropout(
    attn_weight: Tensor, softmax_dropout: float, dst_idx: Tensor = None
):
    for _ in range(10):
        dropout_mask = torch.rand_like(attn_weight) < softmax_dropout
        not_all_drop_mask = scatter_sum(~dropout_mask, index=dst_idx, dim=0)
        if not_all_drop_mask.all():
            attn_weight = attn_weight.masked_fill(dropout_mask, -float("inf"))
            return attn_weight
    raise RuntimeError(
        "Softmax dropout failed to keep at least one edge for each node after 10 attempts."
    )


def build_constraint_logit_mask(
    num_nodes: int,
    aa_constraint_mask: Optional[Tensor],
    inverse_fold_restriction: list[str],
    canonical_tokens: list[str],
    inf: float,
    device: torch.device,
) -> Tensor:
    """Build per-position inverse-folding logit mask."""
    num_aa = len(canonical_tokens)
    bias = torch.zeros(num_nodes, num_aa, dtype=torch.float32, device=device)

    if aa_constraint_mask is not None:
        expected_shape = (num_nodes, num_aa)
        if aa_constraint_mask.shape == expected_shape:
            custom_mask = aa_constraint_mask.to(device=device, dtype=torch.float32)
            bias = custom_mask.clone()
            # 1.0 is the legacy/magic value for "disallowed" hard blocks.
            bias[custom_mask == 1.0] = -inf

    global_blocked = torch.zeros(num_aa, dtype=torch.bool, device=device)
    for res_type in inverse_fold_restriction:
        global_blocked[canonical_tokens.index(res_type)] = True
        
    bias[:, global_blocked] = -inf
    
    still_all_blocked = (bias < 0).all(dim=1)
    if still_all_blocked.any():
        blocked_positions = torch.where(still_all_blocked)[0].tolist()
        raise ValueError(
            f"Inverse folding has no valid amino acids at token positions "
            f"{blocked_positions} after applying global restrictions."
        )

    return bias


class MLPAttnGNN(nn.Module):
    def __init__(
        self,
        node_dim: int,
        pair_dim: int,
        hidden_dim: int,
        dropout: float,
        softmax_dropout: float,
        transformation_scale_factor: float,
        num_heads: int = 4,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.pair_dim = pair_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.softmax_dropout = softmax_dropout
        self.transformation_scale_factor = transformation_scale_factor
        self.num_heads = num_heads

        self.attn_weight_mlp = nn.Sequential(
            nn.Linear(self.node_dim * 2 + self.pair_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, num_heads),
        )

        self.attn_value_mlp = nn.Sequential(
            nn.Linear(self.node_dim + self.pair_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.node_dim),
        )

        self.attn_output_linear = nn.Sequential(
            nn.Linear(self.hidden_dim * self.num_heads, self.node_dim),
            nn.Dropout(self.dropout),
            nn.SyncBatchNorm(self.node_dim),
        )

        self.attn_FFN = nn.Sequential(
            nn.Linear(self.node_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.node_dim),
            nn.Dropout(self.dropout),
            nn.SyncBatchNorm(self.node_dim),
        )

        self.edge_FFN = nn.Sequential(
            nn.Linear(self.node_dim * 2 + self.pair_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.pair_dim),
            nn.Dropout(self.dropout),
            nn.SyncBatchNorm(self.pair_dim),
        )

    def forward(self, s, z, edge_idx):
        src_idx, dst_idx = edge_idx[0], edge_idx[1]

        z = z + self.edge_FFN(torch.cat([s[src_idx], s[dst_idx], z], dim=1))

        attn_weight = self.attn_weight_mlp(
            torch.cat([s[dst_idx], s[src_idx], z], dim=1)
        )
        attn_value = self.attn_value_mlp(torch.cat([s[src_idx], z], dim=1))

        if self.training & (self.softmax_dropout > 0):
            attn_weight = softmax_dropout(
                attn_weight, softmax_dropout=self.softmax_dropout, dst_idx=dst_idx
            )

        attn_weight = scatter_softmax(attn_weight, index=dst_idx, dim=0)
        attn_output = attn_weight.unsqueeze(-1) * attn_value.unsqueeze(1)
        attn_output = scatter_sum(attn_output, index=dst_idx, dim=0).flatten(
            start_dim=1
        )

        s = s + self.attn_output_linear(attn_output)
        s = s + self.attn_FFN(s)

        return s, z


class MLPAttnGNNDecoder(nn.Module):
    def __init__(
        self,
        node_dim: int,
        pair_dim: int,
        hidden_dim: int,
        dropout: float,
        softmax_dropout: float,
        transformation_scale_factor: float,
        num_heads: int = 4,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.pair_dim = pair_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.softmax_dropout = softmax_dropout
        self.transformation_scale_factor = transformation_scale_factor
        self.num_heads = num_heads

        self.attn_weight_mlp = nn.Sequential(
            nn.Linear(self.node_dim * 2 + self.pair_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, num_heads),
        )

        self.attn_value_mlp = nn.Sequential(
            nn.Linear(self.node_dim + self.pair_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.node_dim),
        )

        self.attn_output_linear = nn.Sequential(
            nn.Linear(self.hidden_dim * self.num_heads, self.node_dim),
            nn.Dropout(self.dropout),
            nn.SyncBatchNorm(self.node_dim),
        )

        self.attn_FFN = nn.Sequential(
            nn.Linear(self.node_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.node_dim),
            nn.Dropout(self.dropout),
            nn.SyncBatchNorm(self.node_dim),
        )

    def forward(self, s, z, edge_idx):
        src_idx, dst_idx = edge_idx[0], edge_idx[1]

        attn_weight = self.attn_weight_mlp(torch.cat([s[dst_idx], z], dim=1))
        attn_value = self.attn_value_mlp(z)

        if self.training & (self.softmax_dropout > 0):
            attn_weight = softmax_dropout(
                attn_weight, softmax_dropout=self.softmax_dropout, dst_idx=dst_idx
            )

        attn_weight = scatter_softmax(attn_weight, index=dst_idx, dim=0)
        attn_output = attn_weight.unsqueeze(-1) * attn_value.unsqueeze(1)
        attn_output = scatter_sum(attn_output, index=dst_idx, dim=0).flatten(
            start_dim=1
        )

        s = s + self.attn_output_linear(attn_output)
        s = s + self.attn_FFN(s)

        return s

    def sample(self, s, z):
        dst_idx = torch.zeros(z.shape[0], dtype=torch.long, device=z.device)
        attn_weight = self.attn_weight_mlp(torch.cat([s[dst_idx], z], dim=1))
        attn_value = self.attn_value_mlp(z)

        if self.training & (self.softmax_dropout > 0):
            attn_weight = softmax_dropout(
                attn_weight, softmax_dropout=self.softmax_dropout, dst_idx=dst_idx
            )

        attn_weight = scatter_softmax(attn_weight, index=dst_idx, dim=0)
        attn_output = attn_weight.unsqueeze(-1) * attn_value.unsqueeze(1)
        attn_output = scatter_sum(attn_output, index=dst_idx, dim=0).flatten(
            start_dim=1
        )

        s = s + self.attn_output_linear(attn_output)
        s = s + self.attn_FFN(s)

        return s


class InverseFoldingEncoder(nn.Module):
    def __init__(
        self,
        atom_s: int,
        atom_z: int,
        token_s: int,
        token_z: int,
        node_dim: int = 128,
        pair_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        softmax_dropout: float = 0.2,
        num_encoder_layers: int = 3,
        transformation_scale_factor: float = 1.0,
        inverse_fold_noise: float = 0.3,
        topk: int = 30,
        num_heads: int = 4,
        enable_input_embedder: bool = False,
        backbone_noise: float = 0.0,
        **kwargs, # old checkpoint compatibility
    ):
        """Initialize the Inverse Folding Encoder."""
        super().__init__()

        self.atom_s = atom_s
        self.atom_z = atom_z
        self.token_s = token_s
        self.token_z = token_z
        self.node_dim = node_dim
        self.pair_dim = pair_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.softmax_dropout = softmax_dropout
        self.num_encoder_layers = num_encoder_layers
        self.transformation_scale_factor = transformation_scale_factor
        self.inverse_fold_noise = inverse_fold_noise
        self.topk = topk
        self.backbone_noise = backbone_noise
        self.num_heads = num_heads
        self.r_max = 32
        self.enable_input_embedder = enable_input_embedder
        self.edge_input_dim = (
            256 + 2 * self.r_max + 2 + 1 + 1 + len(const.bond_types) + 1 + 1
        )

        self.linear_token_to_node = nn.Linear(
            135 if not self.enable_input_embedder else self.token_s, self.node_dim
        )
        self.linear_token_to_pair = nn.Linear(self.edge_input_dim, self.pair_dim)

        self.encoder_layers = nn.ModuleList()
        for i in range(self.num_encoder_layers):
            layer = MLPAttnGNN(
                node_dim=node_dim,
                pair_dim=pair_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                softmax_dropout=softmax_dropout,
                transformation_scale_factor=transformation_scale_factor,
                num_heads=num_heads,
            )
            self.encoder_layers.append(layer)

        self.distance_gaussian_smearing = GaussianSmearing(
            start=0.0,
            stop=20.0,
            num_gaussians=16,
        )

    @torch.no_grad()
    def init_knn_graph(self, feats: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        valid_mask = (
            feats["token_resolved_mask"].bool() & feats["token_pad_mask"].bool()
        )
        B, N = valid_mask.shape
        valid_mask_pair = valid_mask.unsqueeze(1) & valid_mask.unsqueeze(2)
        token_index = (valid_mask.flatten().cumsum(dim=0) - 1).view(B, N)
        topk = min(self.topk, N)

        coords = feats["center_coords"]
        if (not self.training) and self.backbone_noise > 0:
            coords = coords + torch.randn_like(coords) * self.backbone_noise
        dists = torch.cdist(coords, coords)
        dists = dists.masked_fill(~valid_mask_pair, float("inf"))
        src_idx = torch.topk(dists, topk, largest=False).indices
        dst_idx = torch.arange(N, device=src_idx.device)[None, :, None].expand(
            B, -1, topk
        )

        assert feats["token_bonds"].shape[-1] == 1
        token_bonds = feats["token_bonds"][..., 0]
        type_bonds = feats["type_bonds"]
        token_bonds = torch.gather(token_bonds, 2, src_idx)
        type_bonds = torch.gather(type_bonds, 2, src_idx)

        src_idx, dst_idx = src_idx.flatten(start_dim=1), dst_idx.flatten(start_dim=1)
        token_bonds, type_bonds = token_bonds.flatten(), type_bonds.flatten()

        src_valid_mask = torch.gather(valid_mask, 1, src_idx).flatten()
        dst_valid_mask = torch.gather(valid_mask, 1, dst_idx).flatten()
        edge_valid_mask = src_valid_mask & dst_valid_mask

        src_idx = torch.gather(token_index, 1, src_idx).flatten()[edge_valid_mask]
        dst_idx = torch.gather(token_index, 1, dst_idx).flatten()[edge_valid_mask]
        edge_idx = torch.stack([src_idx, dst_idx], dim=0)

        token_bonds, type_bonds = (
            token_bonds[edge_valid_mask],
            type_bonds[edge_valid_mask],
        )
        type_bonds = one_hot(type_bonds, num_classes=len(const.bond_types) + 1)
        bond = torch.cat([token_bonds[..., None], type_bonds], dim=-1)

        return edge_idx, valid_mask, bond

    @torch.no_grad()
    def extract_geo_feat(
        self, feats: Dict[str, Tensor], edge_idx: Tensor, valid_mask: Tensor
    ) -> Tensor:
        src_idx, dst_idx = edge_idx[0], edge_idx[1]
        token_to_bb4_atoms = feats["token_to_bb4_atoms"]
        r = feats["coords"]
        noise_std = self.inverse_fold_noise if self.training else self.backbone_noise
        if noise_std > 0:
            # Independent Gaussian noise on every atom (incl. N, CA, C, O)
            r = r + torch.randn_like(r) * noise_std
        B, N = valid_mask.shape
        r_repr = torch.bmm(
            token_to_bb4_atoms.float().view(B, N * 4, -1), r.view(B, -1, 3)
        )
        r_repr = r_repr.reshape(B, N, 4, 3)
        r_repr = r_repr[valid_mask]

        dist = torch.norm(
            r_repr[src_idx, None] - r_repr[dst_idx, :, None], dim=-1
        ).flatten(start_dim=1)
        dist = self.distance_gaussian_smearing(dist)
        return dist

    @torch.no_grad()
    def extract_attr_feat(
        self, feats: Dict[str, Tensor], edge_idx: Tensor, valid_mask: Tensor
    ) -> Tensor:
        src_idx, dst_idx = edge_idx[0], edge_idx[1]

        feature_asym_id = feats["feature_asym_id"][valid_mask]
        b_same_chain = feature_asym_id[src_idx] == feature_asym_id[dst_idx]

        feature_residue_index = feats["feature_residue_index"][valid_mask]
        b_same_residue = (
            feature_residue_index[src_idx] == feature_residue_index[dst_idx]
        )
        d_residue = feature_residue_index[dst_idx] - feature_residue_index[src_idx]
        d_residue = torch.clip(
            d_residue + self.r_max,
            0,
            2 * self.r_max,
        )
        d_residue = torch.where(
            b_same_chain,
            d_residue,
            torch.zeros_like(d_residue) + 2 * self.r_max + 1,
        )
        a_rel_pos = one_hot(d_residue, 2 * self.r_max + 2)

        edge_attr = torch.cat(
            [
                a_rel_pos.float(),
                b_same_chain[..., None].float(),
                b_same_residue[..., None].float(),
            ],
            dim=-1,
        )

        b_standard = feats["is_standard"].bool()[valid_mask][..., None]
        mol_type = feats["mol_type"][valid_mask]
        mol_type_one_hot = F.one_hot(mol_type, num_classes=len(const.chain_type_ids))
        nonpolymer_mask = mol_type == const.chain_type_ids["NONPOLYMER"]
        modified = feats["modified"].unsqueeze(-1)[valid_mask]

        atom_feats = torch.cat(
            [
                feats["ref_charge"].unsqueeze(-1),
                feats["ref_element"],
            ],
            dim=-1,
        )

        with torch.autocast("cuda", enabled=False):
            atom_to_token = feats["atom_to_token"].float()
            atom_to_token_mean = atom_to_token / (
                atom_to_token.sum(dim=1, keepdim=True) + 1e-6
            )
            a = torch.bmm(atom_to_token_mean.transpose(1, 2), atom_feats)

            atom_mask = feats["atom_pad_mask"]
            atom_padding_sum_to_token = torch.bmm(
                atom_to_token.transpose(1, 2), 1 - atom_mask.unsqueeze(-1)
            )
            assert atom_padding_sum_to_token[valid_mask].sum() == 0
        a = a[valid_mask]

        node_attr = torch.cat(
            [
                b_standard.float(),
                modified,
                mol_type_one_hot.float(),
                a,
            ],
            dim=-1,
        )

        return node_attr, edge_attr

    def forward(self, feats):
        edge_idx, valid_mask, bond = self.init_knn_graph(feats)
        geo_feat = self.extract_geo_feat(feats, edge_idx, valid_mask)
        node_attr, edge_attr = self.extract_attr_feat(feats, edge_idx, valid_mask)
        if self.enable_input_embedder:
            node_attr = feats["s_inputs"][valid_mask]

        N = valid_mask.sum()
        s = self.linear_token_to_node(node_attr)
        z = self.linear_token_to_pair(torch.cat([geo_feat, edge_attr, bond], dim=-1))

        for layer in self.encoder_layers:
            s, z = layer(s, z, edge_idx)

        return edge_idx, valid_mask, s, z



class InverseFoldingDecoder(nn.Module):
    def __init__(
        self,
        atom_s: int,
        atom_z: int,
        token_s: int,
        token_z: int,
        node_dim: int = 128,
        pair_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        softmax_dropout: float = 0.2,
        num_encoder_layers: int = 3,
        transformation_scale_factor: float = 1.0,
        inverse_fold_noise: float = 0.3,
        topk: int = 30,
        num_heads: int = 4,
        num_decoder_layers: int = 3,
        inverse_fold_restriction: List[str] = [],
        sampling_temperature: float = 0.1,
        tie_symmetric_sequences: bool = True,
        heteromer_anti_correlation: bool = False,
        anti_correlation_strength: float = 2.0,
        double_tape_core_bias: float = 0.0,
        double_tape_aromatic_bias: float = 0.0,
        double_tape_polar_bias: float = 0.0,
        background_unbias_strength: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 0.0,
        aa_bias: str = "",
        **kwargs, # old checkpoint compatibility
    ):
        super().__init__()
        self.atom_s = atom_s
        self.atom_z = atom_z
        self.token_s = token_s
        self.token_z = token_z
        self.node_dim = node_dim
        self.pair_dim = pair_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.softmax_dropout = softmax_dropout
        self.num_encoder_layers = num_encoder_layers
        self.transformation_scale_factor = transformation_scale_factor
        self.inverse_fold_noise = inverse_fold_noise
        self.topk = topk
        self.num_heads = num_heads
        self.num_res_type = const.num_tokens
        self.num_decoder_layers = num_decoder_layers
        self.inverse_fold_restriction = inverse_fold_restriction
        self.sampling_temperature = sampling_temperature
        self.tie_symmetric_sequences = tie_symmetric_sequences
        self.heteromer_anti_correlation = heteromer_anti_correlation
        self.anti_correlation_strength = anti_correlation_strength
        self.double_tape_core_bias = double_tape_core_bias
        self.double_tape_aromatic_bias = double_tape_aromatic_bias
        self.double_tape_polar_bias = double_tape_polar_bias
        self.background_unbias_strength = background_unbias_strength
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.aa_bias = aa_bias

        self.decoder_layers = nn.ModuleList()
        self.inf = 10**6
        for i in range(self.num_decoder_layers):
            layer = MLPAttnGNNDecoder(
                node_dim=node_dim,
                pair_dim=pair_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                softmax_dropout=softmax_dropout,
                transformation_scale_factor=transformation_scale_factor,
                num_heads=num_heads,
            )
            self.decoder_layers.append(layer)

        self.seq_to_s = nn.Linear(const.num_tokens, self.node_dim)
        self.predictor = nn.Linear(self.node_dim, const.num_tokens, bias=False)

        with torch.no_grad():
            # init the output of the predictor to be zero
            self.predictor.weight.zero_()

    def _build_symmetric_groups(
        self,
        feats: Dict[str, Tensor],
        valid_mask: Tensor,
        design_mask: Tensor,
    ) -> Tuple[Dict[int, List[int]], Dict[int, int]]:
        """Build mapping from positions to symmetric groups for homomer tying."""
        from collections import defaultdict

        symmetric_group = feats["symmetric_group"][valid_mask]
        res_idx = feats["feature_residue_index"][valid_mask]

        # Group by (symmetric_group, res_idx) - positions that should share sequence
        key_to_positions = defaultdict(list)

        num_nodes = symmetric_group.shape[0]
        for i in range(num_nodes):
            if design_mask[i]:
                group = symmetric_group[i].item()
                if group > 0:  # 0 = no group
                    key = (group, res_idx[i].item())
                    key_to_positions[key].append(i)

        # Build symmetric groups (only groups with >1 member)
        sym_groups = {}
        position_to_group = {}
        group_id = 0

        for positions in key_to_positions.values():
            if len(positions) > 1:
                sym_groups[group_id] = positions
                for pos in positions:
                    position_to_group[pos] = group_id
                group_id += 1

        return sym_groups, position_to_group

    def forward(self, s, z, edge_idx, valid_mask, feats):
        with torch.no_grad():
            src_idx, dst_idx = edge_idx[0], edge_idx[1]
            rand = torch.rand(valid_mask.sum(), device=valid_mask.device)
            res_type_edge_visibility = rand[src_idx] < rand[dst_idx]
            res_type_clone = feats["res_type_clone"].bool()
            res_type_clone = res_type_clone[valid_mask][src_idx]
            res_type_clone = res_type_clone * res_type_edge_visibility[:, None]
            res_type_clone = res_type_clone.to(z)
        res_rep = self.seq_to_s(res_type_clone)
        neighbors_rep = torch.concat([z, s[src_idx] + res_rep], dim=-1)
        for layer in self.decoder_layers:
            s = layer(s, neighbors_rep, edge_idx)

        logits = self.predictor(s)

        B, N = valid_mask.shape
        logist_dense = torch.zeros(B, N, self.num_res_type, device=logits.device)
        logist_dense[valid_mask] = logits

        out_dict = {
            "logits": logist_dense,
            "res_type": logist_dense,
            "valid_mask": valid_mask,
        }
        return out_dict

    @torch.no_grad()
    def sample(self, s, z, edge_idx, valid_mask, feats):
        """Sample the output from the decoder."""
        num_nodes = s.shape[0]

        if "inverse_fold_design_mask" in feats:
            design_mask = feats["inverse_fold_design_mask"].bool()[valid_mask]
        else:
            design_mask = feats["design_mask"].bool()[valid_mask]
        num_not_design = (~design_mask).sum().item()
        num_design = design_mask.sum().item()
        assert num_design == num_nodes - num_not_design, (
            f"num_design: {num_design}, num_not_design: {num_not_design}"
        )

        constraint_mask = None
        if "aa_constraint_mask" in feats:
            constraint_mask = feats["aa_constraint_mask"][valid_mask]
        per_residue_mask = build_constraint_logit_mask(
            num_nodes=num_nodes,
            aa_constraint_mask=constraint_mask,
            inverse_fold_restriction=self.inverse_fold_restriction,
            canonical_tokens=const.canonical_tokens,
            inf=self.inf,
            device=s.device,
        )

        if self.aa_bias:
            for term in self.aa_bias.split(","):
                aa, val = term.split(":")
                val = float(val)
                # Map 1-letter code to canonical token index
                token_name = const.prot_letter_to_token_name.get(aa)
                if token_name and token_name in const.canonical_tokens:
                    idx = const.canonical_tokens.index(token_name)
                    # For the plan: 'favor Threonine/Serine substitution on solvent positions'
                    # But ProteinMPNN global aa_bias applies to ALL positions!
                    per_residue_mask[:, idx] += val
        
        if self.background_unbias_strength > 0:
            # Remove the natural-proteome prior: penalize common residues, boost rare ones
            log_prior = torch.log(BG_FREQ.to(device=s.device))
            prior_offset = (log_prior - log_prior.mean()) * self.background_unbias_strength
            per_residue_mask -= prior_offset.unsqueeze(0)

        if (
            self.double_tape_core_bias > 0
            or self.double_tape_aromatic_bias > 0
            or self.double_tape_polar_bias > 0
        ):
            token_bb4 = feats["token_to_bb4_atoms"].float()
            coords = feats["coords"]
            
            # token_bb4 might be [B, N*4, A] or [B, N, 4, A]
            # coords might be [B, A, 3] or [B, 1, A, 3]
            # Let's flatten to [B, N*4, A] and [B, A, 3]
            B = coords.shape[0]
            A = coords.shape[-2]
            token_bb4_flat = token_bb4.view(B, -1, A)
            coords_flat = coords.view(B, A, 3)
            
            bb4_flat = torch.bmm(token_bb4_flat, coords_flat)
            bb4 = bb4_flat.view(B, -1, 4, 3)[valid_mask]
            
            CA = bb4[:, 1]
            asym_id = feats["feature_asym_id"][valid_mask]
            
            # Compute bilayer normal from side 0 to side 1
            com0 = CA[asym_id % 2 == 0].mean(dim=0)
            com1 = CA[asym_id % 2 == 1].mean(dim=0)
            normal = torch.nn.functional.normalize(com1 - com0, dim=0)
            
            # Sidechain direction: idealized C-beta reconstructed from N, CA, C
            # (bb4 atoms are N, CA, C, O; index 3 is the carbonyl O, not CB).
            b = CA - bb4[:, 0]
            c = bb4[:, 2] - CA
            vec = (
                -0.531018 * torch.nn.functional.normalize(b - c, dim=1)
                + 1.20673 * torch.nn.functional.normalize(torch.cross(b, c, dim=1), dim=1)
            )
            vec = torch.nn.functional.normalize(vec, dim=1)
            
            # Identify inward vs outward positions (half-space test on bilayer normal)
            curr_normal = torch.where((asym_id % 2 == 0).unsqueeze(1), normal.unsqueeze(0), -normal.unsqueeze(0))
            dot = (vec * curr_normal).sum(dim=1).clamp(-1, 1)
            
            is_inward = dot > 0.0
            is_outward = ~is_inward
            
            if self.double_tape_core_bias > 0:
                for token in ("ALA", "VAL", "ILE", "LEU", "MET", "PHE", "TRP"):
                    idx_co = const.canonical_tokens.index(token)
                    per_residue_mask[is_inward, idx_co] += self.double_tape_core_bias
            
            if self.double_tape_aromatic_bias > 0:
                for token in ("TYR", "TRP"):
                    idx_ar = const.canonical_tokens.index(token)
                    per_residue_mask[is_outward, idx_ar] += self.double_tape_aromatic_bias

            if self.double_tape_polar_bias > 0:
                # Residues able to form salt bridges / H-bonds on the exterior
                for token in ("ARG", "LYS", "ASP", "GLU", "ASN", "GLN", "HIS"):
                    idx_po = const.canonical_tokens.index(token)
                    per_residue_mask[is_outward, idx_po] += self.double_tape_polar_bias

        order = torch.randperm(num_nodes, device=s.device).cpu().numpy().tolist()
        # Non-design residues are not sampled and used as the condition. So the order should filter them out.
        if num_not_design > 0:
            id_not_design = torch.where(~design_mask)[0].cpu().numpy().tolist()
            for i in id_not_design:
                order.remove(i)
            decoded_seq = torch.zeros(num_nodes, const.num_tokens, device=s.device)
            logits = torch.zeros(num_nodes, const.num_tokens, device=s.device)
            decoded_seq[~design_mask] = logits[~design_mask] = feats["res_type_clone"][
                valid_mask
            ][~design_mask].float()
        else:
            decoded_seq = torch.zeros(num_nodes, const.num_tokens, device=s.device)
            logits = torch.zeros(num_nodes, const.num_tokens, device=s.device)

        # Build symmetric groups for homomer tying
        if self.tie_symmetric_sequences and "symmetric_group" in feats:
            sym_groups, position_to_group = self._build_symmetric_groups(
                feats, valid_mask, design_mask
            )
            sampled = set(torch.where(~design_mask)[0].cpu().numpy().tolist()) if num_not_design > 0 else set()
        else:
            sym_groups, position_to_group = {}, {}
            sampled = set()

        cross_partner_map = {}
        if self.heteromer_anti_correlation and sym_groups and "symmetric_group" in feats:
            from collections import defaultdict
            symmetric_group_tensor = feats["symmetric_group"][valid_mask]
            res_idx_tensor = feats["feature_residue_index"][valid_mask]
            
            # Find unique non-zero symmetric_group values among designable positions
            unique_sym_groups = set()
            for i in range(symmetric_group_tensor.shape[0]):
                if design_mask[i]:
                    g = symmetric_group_tensor[i].item()
                    if g > 0:
                        unique_sym_groups.add(g)
                        
            if len(unique_sym_groups) == 2:
                res_idx_to_groups = defaultdict(list)
                for gid, positions in sym_groups.items():
                    pos0 = positions[0]
                    sym_g = symmetric_group_tensor[pos0].item()
                    r_idx = res_idx_tensor[pos0].item()
                    res_idx_to_groups[r_idx].append((gid, sym_g))
                
                for r_idx, groups in res_idx_to_groups.items():
                    if len(groups) == 2:
                        gid_a, gid_b = groups[0][0], groups[1][0]
                        cross_partner_map[gid_a] = gid_b
                        cross_partner_map[gid_b] = gid_a
        sampled_groups = set()

        # Per-chain residue usage counts for the repetition penalty
        chain_aa_counts = {}
        node_chain = []
        if self.repetition_penalty > 0:
            node_chain = feats["feature_asym_id"][valid_mask].cpu().numpy().tolist()
            for c in set(node_chain):
                chain_aa_counts[c] = torch.zeros(
                    len(const.canonical_tokens), device=s.device
                )

        src_idx, dst_idx = edge_idx[0], edge_idx[1]

        # decoding in order
        for i in order:
            # Skip if already sampled (symmetric position was processed earlier)
            if self.tie_symmetric_sequences and i in sampled:
                continue

            # Get symmetric positions (or just [i] if no symmetry)
            if self.tie_symmetric_sequences and i in position_to_group:
                positions = sym_groups[position_to_group[i]]
            else:
                positions = [i]

            # Aggregate logits from all symmetric positions
            aggregated_logits = None
            for pos in positions:
                s_pos = s[pos : pos + 1]
                edge_mask_pos = dst_idx == pos
                z_pos = z[edge_mask_pos]
                src_idx_pos = src_idx[edge_mask_pos]
                res_type_pos = decoded_seq[src_idx_pos]
                res_rep = self.seq_to_s(res_type_pos)
                neighbors_rep_pos = torch.concat([z_pos, s[src_idx_pos] + res_rep], dim=-1)

                s_temp = s_pos
                for layer in self.decoder_layers:
                    s_temp = layer.sample(s_temp, neighbors_rep_pos)

                logits_pos = self.predictor(s_temp)
                if aggregated_logits is None:
                    aggregated_logits = logits_pos
                else:
                    aggregated_logits = aggregated_logits + logits_pos

            # Average logits across symmetric positions
            aggregated_logits = aggregated_logits / len(positions)

            # Sample from aggregated logits
            pred_canonical = (
                aggregated_logits[
                    :,
                    const.canonicals_offset : len(const.canonical_tokens)
                    + const.canonicals_offset,
                ]
                + per_residue_mask[i : i + 1]  # Position-specific mask
            )

            # Penalize amino acids already used within the same chain(s)
            if self.repetition_penalty > 0:
                rep_counts = torch.stack(
                    [chain_aa_counts[node_chain[pos]] for pos in positions]
                ).mean(dim=0)
                pred_canonical = pred_canonical - self.repetition_penalty * rep_counts[None]

            if self.sampling_temperature is None:
                ids_canonical = torch.argmax(pred_canonical, dim=-1)
            else:
                probs = F.softmax(pred_canonical / self.sampling_temperature, dim=-1)
                if self.top_p < 1.0:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                    cum_probs = torch.cumsum(sorted_probs, dim=-1)
                    # Always keep the most probable residue
                    remove_sorted = (cum_probs - sorted_probs) > self.top_p
                    remove = torch.zeros_like(remove_sorted).scatter(
                        1, sorted_idx, remove_sorted
                    )
                    probs = probs.masked_fill(remove, 0.0)
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                ids_canonical = torch.multinomial(probs, num_samples=1).squeeze(-1)

            ids = ids_canonical + const.canonicals_offset
            pred_one_hot = F.one_hot(ids, num_classes=const.num_tokens)

            # Apply same residue to all symmetric positions
            for pos in positions:
                decoded_seq[pos] = pred_one_hot
                logits[pos] = aggregated_logits
                if self.tie_symmetric_sequences:
                    sampled.add(pos)
                if self.repetition_penalty > 0:
                    chain_aa_counts[node_chain[pos]][ids_canonical.item()] += 1.0
                    
            if self.heteromer_anti_correlation and i in position_to_group:
                my_group_id = position_to_group[i]
                sampled_groups.add(my_group_id)
                partner_group_id = cross_partner_map.get(my_group_id)
                if partner_group_id is not None and partner_group_id not in sampled_groups:
                    # Penalize the same AA at the partner's corresponding position (multiplied to overcome sharp logits)
                    for partner_pos in sym_groups[partner_group_id]:
                        per_residue_mask[partner_pos, ids_canonical.item()] -= (self.anti_correlation_strength * 10.0)

        n_tokens = valid_mask.shape[1]
        res_type = torch.zeros(1, n_tokens, self.num_res_type, device=s.device)
        res_type[valid_mask] = decoded_seq
        unk_ids = torch.full(
            [(~valid_mask).sum().item()], const.tokens.index("UNK"), device=s.device
        )
        unk_value = F.one_hot(unk_ids, num_classes=const.num_tokens)
        res_type[~valid_mask] = unk_value.float()
        feats["res_type"] = res_type.long()

        logist_dense = torch.zeros(1, n_tokens, self.num_res_type, device=logits.device)
        logist_dense[valid_mask] = logits
        logist_dense[~valid_mask] = feats["res_type_clone"][~valid_mask].float()

        out_dict = {
            "logits": logist_dense,
            "res_type": res_type,
            "valid_mask": valid_mask,
        }
        out_dict["sample_atom_coords"] = feats["coords"][0]
        return out_dict
