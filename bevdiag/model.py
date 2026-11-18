"""Pure-PyTorch port of BEVFormer-tiny (Li et al., ECCV 2022) for inference.

Loads the official checkpoint `bevformer_tiny_epoch_24.pth` with zero missing / unexpected
keys. No mmcv / mmdet / mmdet3d and no custom CUDA ops: multi-scale deformable attention
uses the grid_sample reference implementation (numerically equivalent to the CUDA kernel).
Module and parameter names mirror the reference code so the state dict maps 1:1.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms.functional import rotate

PC_RANGE = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
POST_CENTER_RANGE = [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]


# --------------------------------------------------------------------------- utils
def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))


def ms_deform_attn(value, spatial_shapes, sampling_locations, attention_weights):
    """Reference multi-scale deformable attention (mmcv `multi_scale_deformable_attn_pytorch`).

    value: (bs, num_value, heads, c); sampling_locations: (bs, nq, heads, levels, points, 2) in [0,1];
    attention_weights: (bs, nq, heads, levels, points). Returns (bs, nq, heads*c).
    """
    bs, _, num_heads, c = value.shape
    _, nq, _, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([int(h) * int(w) for h, w in spatial_shapes], dim=1)
    grids = 2 * sampling_locations - 1
    sampled = []
    for lvl, (h, w) in enumerate(spatial_shapes):
        h, w = int(h), int(w)
        v = value_list[lvl].flatten(2).transpose(1, 2).reshape(bs * num_heads, c, h, w)
        g = grids[:, :, :, lvl].transpose(1, 2).flatten(0, 1)
        sampled.append(F.grid_sample(v, g, mode="bilinear", padding_mode="zeros", align_corners=False))
    attention_weights = attention_weights.transpose(1, 2).reshape(bs * num_heads, 1, nq, num_levels * num_points)
    out = (torch.stack(sampled, dim=-2).flatten(-2) * attention_weights).sum(-1)
    return out.view(bs, num_heads * c, nq).transpose(1, 2).contiguous()


# --------------------------------------------------------------------------- image branch
class ConvWrap(nn.Module):
    def __init__(self, cin, cout, k, p=0):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, padding=p)

    def forward(self, x):
        return self.conv(x)


class FPN1(nn.Module):
    """mmdet FPN with a single input level / single output (BEVFormer-tiny config)."""

    def __init__(self, cin=2048, cout=256):
        super().__init__()
        self.lateral_convs = nn.ModuleList([ConvWrap(cin, cout, 1)])
        self.fpn_convs = nn.ModuleList([ConvWrap(cout, cout, 3, 1)])

    def forward(self, x):
        return self.fpn_convs[0](self.lateral_convs[0](x))


class ResNet50C5(nn.Module):
    """torchvision ResNet-50 trunk returning stage-4 (C5) features; key names match mmdet."""

    def __init__(self):
        super().__init__()
        r = torchvision.models.resnet50(weights=None)
        self.conv1, self.bn1, self.relu, self.maxpool = r.conv1, r.bn1, r.relu, r.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = r.layer1, r.layer2, r.layer3, r.layer4

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        return self.layer4(self.layer3(self.layer2(self.layer1(x))))


# --------------------------------------------------------------------------- attention blocks
class FFN(nn.Module):
    def __init__(self, dim=256, hidden=512):
        super().__init__()
        self.layers = nn.Sequential(nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(inplace=True), nn.Dropout(0.1)),
                                    nn.Linear(hidden, dim), nn.Dropout(0.1))

    def forward(self, x, identity=None):
        return (x if identity is None else identity) + self.layers(x)


class TemporalSelfAttention(nn.Module):
    def __init__(self, dim=256, heads=8, levels=1, points=4, num_bev_queue=2):
        super().__init__()
        self.dim, self.heads, self.levels, self.points, self.nq = dim, heads, levels, points, num_bev_queue
        self.sampling_offsets = nn.Linear(dim * num_bev_queue, num_bev_queue * heads * levels * points * 2)
        self.attention_weights = nn.Linear(dim * num_bev_queue, num_bev_queue * heads * levels * points)
        self.value_proj = nn.Linear(dim, dim)
        self.output_proj = nn.Linear(dim, dim)

    def forward(self, query, value, query_pos, reference_points, spatial_shapes):
        bs, num_query, c = query.shape
        if value is None:
            value = torch.stack([query, query], 1).reshape(bs * 2, num_query, c)
        identity = query
        query = query + query_pos
        num_value = value.shape[1]
        query = torch.cat([value[:bs], query], -1)
        value = self.value_proj(value).reshape(bs * self.nq, num_value, self.heads, -1)
        off = self.sampling_offsets(query).view(bs, num_query, self.heads, self.nq, self.levels, self.points, 2)
        aw = self.attention_weights(query).view(bs, num_query, self.heads, self.nq, self.levels * self.points)
        aw = aw.softmax(-1).view(bs, num_query, self.heads, self.nq, self.levels, self.points)
        aw = aw.permute(0, 3, 1, 2, 4, 5).reshape(bs * self.nq, num_query, self.heads, self.levels, self.points)
        off = off.permute(0, 3, 1, 2, 4, 5, 6).reshape(bs * self.nq, num_query, self.heads, self.levels, self.points, 2)
        norm = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
        loc = reference_points[:, :, None, :, None, :] + off / norm[None, None, None, :, None, :]
        out = ms_deform_attn(value, spatial_shapes, loc, aw)  # (bs*2, nq, c)
        out = out.permute(1, 2, 0).view(num_query, c, bs, self.nq).mean(-1).permute(2, 0, 1)
        return self.output_proj(out) + identity


class MSDeformableAttention3D(nn.Module):
    def __init__(self, dim=256, heads=8, levels=1, points=8):
        super().__init__()
        self.dim, self.heads, self.levels, self.points = dim, heads, levels, points
        self.sampling_offsets = nn.Linear(dim, heads * levels * points * 2)
        self.attention_weights = nn.Linear(dim, heads * levels * points)
        self.value_proj = nn.Linear(dim, dim)

    def forward(self, query, value, reference_points, spatial_shapes):
        bs, nq, _ = query.shape
        nv = value.shape[1]
        value = self.value_proj(value).view(bs, nv, self.heads, -1)
        off = self.sampling_offsets(query).view(bs, nq, self.heads, self.levels, self.points, 2)
        aw = self.attention_weights(query).view(bs, nq, self.heads, self.levels * self.points).softmax(-1)
        aw = aw.view(bs, nq, self.heads, self.levels, self.points)
        norm = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
        _, _, nz, _ = reference_points.shape
        ref = reference_points[:, :, None, None, None, :, :]
        off = (off / norm[None, None, None, :, None, :]).view(bs, nq, self.heads, self.levels, self.points // nz, nz, 2)
        loc = (ref + off).view(bs, nq, self.heads, self.levels, self.points, 2)
        return ms_deform_attn(value, spatial_shapes, loc, aw)


class SpatialCrossAttention(nn.Module):
    def __init__(self, dim=256, num_cams=6):
        super().__init__()
        self.dim, self.num_cams = dim, num_cams
        self.deformable_attention = MSDeformableAttention3D(dim)
        self.output_proj = nn.Linear(dim, dim)

    def forward(self, query, feats, reference_points_cam, bev_mask, spatial_shapes):
        # query (bs, nq, c); feats (num_cam, L, bs, c); reference_points_cam (num_cam, bs, nq, D, 2)
        bs, nq, _ = query.shape
        D = reference_points_cam.size(3)
        slots = torch.zeros_like(query)
        indexes = [m[0].sum(-1).nonzero().squeeze(-1) for m in bev_mask]
        max_len = max(len(i) for i in indexes)
        q_re = query.new_zeros(bs, self.num_cams, max_len, self.dim)
        r_re = reference_points_cam.new_zeros(bs, self.num_cams, max_len, D, 2)
        for j in range(bs):
            for i, rp in enumerate(reference_points_cam):
                idx = indexes[i]
                q_re[j, i, :len(idx)] = query[j, idx]
                r_re[j, i, :len(idx)] = rp[j, idx]
        num_cams, L, _, c = feats.shape
        v = feats.permute(2, 0, 1, 3).reshape(bs * num_cams, L, c)
        out = self.deformable_attention(q_re.view(bs * num_cams, max_len, c), v,
                                        r_re.view(bs * num_cams, max_len, D, 2), spatial_shapes)
        out = out.view(bs, num_cams, max_len, c)
        for j in range(bs):
            for i, idx in enumerate(indexes):
                slots[j, idx] += out[j, i, :len(idx)]
        count = (bev_mask.sum(-1) > 0).permute(1, 2, 0).sum(-1).clamp(min=1.0)
        slots = slots / count[..., None]
        return self.output_proj(slots) + query


class BEVFormerLayer(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.attentions = nn.ModuleList([TemporalSelfAttention(dim), SpatialCrossAttention(dim)])
        self.ffns = nn.ModuleList([FFN(dim)])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(3)])

    def forward(self, q, feats, bev_pos, ref_2d, prev_bev, bev_hw, ref_cam, bev_mask, spatial_shapes):
        bev_shape = torch.tensor([list(bev_hw)], device=q.device)
        q = self.attentions[0](q, prev_bev, bev_pos, ref_2d, bev_shape)
        q = self.norms[0](q)
        q = self.attentions[1](q, feats, ref_cam, bev_mask, spatial_shapes)
        q = self.norms[1](q)
        q = self.ffns[0](q)
        return self.norms[2](q)


class BEVFormerEncoder(nn.Module):
    def __init__(self, dim=256, num_layers=3, pc_range=PC_RANGE, num_points_in_pillar=4):
        super().__init__()
        self.layers = nn.ModuleList([BEVFormerLayer(dim) for _ in range(num_layers)])
        self.pc_range = pc_range
        self.num_points_in_pillar = num_points_in_pillar

    @staticmethod
    def ref_3d(H, W, Z, n, device, dtype):
        zs = torch.linspace(0.5, Z - 0.5, n, dtype=dtype, device=device).view(-1, 1, 1).expand(n, H, W) / Z
        xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device).view(1, 1, W).expand(n, H, W) / W
        ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device).view(1, H, 1).expand(n, H, W) / H
        ref = torch.stack((xs, ys, zs), -1).permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
        return ref[None]

    @staticmethod
    def ref_2d(H, W, device, dtype):
        ry, rx = torch.meshgrid(torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
                                torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device), indexing="ij")
        return torch.stack((rx.reshape(-1)[None] / W, ry.reshape(-1)[None] / H), -1).unsqueeze(2)

    def point_sampling(self, ref_3d, lidar2img, img_hw):
        pc = self.pc_range
        ref = ref_3d.clone()
        ref[..., 0:1] = ref[..., 0:1] * (pc[3] - pc[0]) + pc[0]
        ref[..., 1:2] = ref[..., 1:2] * (pc[4] - pc[1]) + pc[1]
        ref[..., 2:3] = ref[..., 2:3] * (pc[5] - pc[2]) + pc[2]
        ref = torch.cat((ref, torch.ones_like(ref[..., :1])), -1).permute(1, 0, 2, 3)
        D, B, nq = ref.shape[:3]
        num_cam = lidar2img.size(1)
        ref = ref.view(D, B, 1, nq, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)
        l2i = lidar2img.view(1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, nq, 1, 1)
        cam = torch.matmul(l2i.float(), ref.float()).squeeze(-1)
        eps = 1e-5
        mask = cam[..., 2:3] > eps
        cam = cam[..., 0:2] / torch.maximum(cam[..., 2:3], torch.ones_like(cam[..., 2:3]) * eps)
        cam[..., 0] /= img_hw[1]
        cam[..., 1] /= img_hw[0]
        mask = mask & (cam[..., 1:2] > 0) & (cam[..., 1:2] < 1) & (cam[..., 0:1] < 1) & (cam[..., 0:1] > 0)
        mask = torch.nan_to_num(mask)
        return cam.permute(2, 1, 3, 0, 4), mask.permute(2, 1, 3, 0, 4).squeeze(-1)

    def forward(self, bev_query, feats, bev_hw, bev_pos, spatial_shapes, prev_bev, shift, lidar2img, img_hw):
        H, W = bev_hw
        bs = bev_query.size(1)
        r3 = self.ref_3d(H, W, self.pc_range[5] - self.pc_range[2], self.num_points_in_pillar,
                         bev_query.device, bev_query.dtype).repeat(bs, 1, 1, 1)
        r2 = self.ref_2d(H, W, bev_query.device, bev_query.dtype).repeat(bs, 1, 1, 1)
        prev_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        ref_cam, bev_mask = self.point_sampling(r3, lidar2img, img_hw)
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32
        shift_r2 = r2.clone() + shift[:, None, None, :]
        q = bev_query.permute(1, 0, 2)
        bev_pos = bev_pos.permute(1, 0, 2)
        _, L, nl, _ = r2.shape
        if prev_bev is not None:
            prev_bev = torch.stack([prev_bev.permute(1, 0, 2), q], 1).reshape(bs * 2, L, -1)
            hyb = torch.stack([shift_r2, r2], 1).reshape(bs * 2, L, nl, 2)
        else:
            hyb = torch.stack([r2, r2], 1).reshape(bs * 2, L, nl, 2)
        for layer in self.layers:
            q = layer(q, feats, bev_pos, hyb, prev_bev, (H, W), ref_cam, bev_mask, spatial_shapes)
        return q


class MHAWrap(nn.Module):
    def __init__(self, dim=256, heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=0.1)

    def forward(self, q, query_pos):
        x = q + query_pos
        return q + self.attn(x, x, q)[0]


class CustomMSDeformableAttention(nn.Module):
    def __init__(self, dim=256, heads=8, levels=1, points=4):
        super().__init__()
        self.heads, self.levels, self.points = heads, levels, points
        self.sampling_offsets = nn.Linear(dim, heads * levels * points * 2)
        self.attention_weights = nn.Linear(dim, heads * levels * points)
        self.value_proj = nn.Linear(dim, dim)
        self.output_proj = nn.Linear(dim, dim)

    def forward(self, query, value, query_pos, reference_points, spatial_shapes):
        identity = query
        query = (query + query_pos).permute(1, 0, 2)
        value = value.permute(1, 0, 2)
        bs, nq, _ = query.shape
        nv = value.shape[1]
        value = self.value_proj(value).view(bs, nv, self.heads, -1)
        off = self.sampling_offsets(query).view(bs, nq, self.heads, self.levels, self.points, 2)
        aw = self.attention_weights(query).view(bs, nq, self.heads, self.levels * self.points).softmax(-1)
        aw = aw.view(bs, nq, self.heads, self.levels, self.points)
        norm = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
        loc = reference_points[:, :, None, :, None, :] + off / norm[None, None, None, :, None, :]
        out = self.output_proj(ms_deform_attn(value, spatial_shapes, loc, aw)).permute(1, 0, 2)
        return out + identity


class DecoderLayer(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.attentions = nn.ModuleList([MHAWrap(dim), CustomMSDeformableAttention(dim)])
        self.ffns = nn.ModuleList([FFN(dim)])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(3)])

    def forward(self, q, value, query_pos, ref, spatial_shapes):
        q = self.norms[0](self.attentions[0](q, query_pos))
        q = self.norms[1](self.attentions[1](q, value, query_pos, ref, spatial_shapes))
        return self.norms[2](self.ffns[0](q))


class Decoder(nn.Module):
    def __init__(self, dim=256, num_layers=6):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(dim) for _ in range(num_layers)])

    def forward(self, query, value, query_pos, reference_points, reg_branches, spatial_shapes):
        out, inter, inter_ref = query, [], []
        for lid, layer in enumerate(self.layers):
            out = layer(out, value, query_pos, reference_points[..., :2].unsqueeze(2), spatial_shapes)
            o = out.permute(1, 0, 2)
            tmp = reg_branches[lid](o)
            new = torch.zeros_like(reference_points)
            new[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points[..., :2])
            new[..., 2:3] = tmp[..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])
            reference_points = new.sigmoid().detach()
            inter.append(out)
            inter_ref.append(reference_points)
        return torch.stack(inter), torch.stack(inter_ref)


class PerceptionTransformer(nn.Module):
    def __init__(self, dim=256, num_cams=6, rotate_center=(100, 100)):
        super().__init__()
        self.dim = dim
        self.encoder = BEVFormerEncoder(dim)
        self.decoder = Decoder(dim)
        self.level_embeds = nn.Parameter(torch.zeros(4, dim))
        self.cams_embeds = nn.Parameter(torch.zeros(num_cams, dim))
        self.reference_points = nn.Linear(dim, 3)
        self.can_bus_mlp = nn.Sequential(nn.Linear(18, dim // 2), nn.ReLU(inplace=True),
                                         nn.Linear(dim // 2, dim), nn.ReLU(inplace=True))
        self.can_bus_mlp.add_module("norm", nn.LayerNorm(dim))
        self.rotate_center = list(rotate_center)

    def get_bev_features(self, feat, bev_queries, bev_hw, grid_length, bev_pos, prev_bev, can_bus, lidar2img, img_hw):
        H, W = bev_hw
        bs = feat.size(0)
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)
        cb = can_bus.detach().cpu().double().numpy()
        dx, dy = cb[:, 0], cb[:, 1]
        ego_angle = cb[:, -2] / np.pi * 180
        trans_len = np.sqrt(dx ** 2 + dy ** 2)
        trans_ang = np.arctan2(dy, dx) / np.pi * 180
        bev_angle = ego_angle - trans_ang
        shift_y = trans_len * np.cos(bev_angle / 180 * np.pi) / grid_length[0] / H
        shift_x = trans_len * np.sin(bev_angle / 180 * np.pi) / grid_length[1] / W
        shift = bev_queries.new_tensor(np.stack([shift_x, shift_y])).permute(1, 0)
        if prev_bev is not None:
            if prev_bev.shape[1] == H * W:
                prev_bev = prev_bev.permute(1, 0, 2)
            prev_bev = prev_bev.clone()
            for i in range(bs):
                t = prev_bev[:, i].reshape(H, W, -1).permute(2, 0, 1)
                t = rotate(t, float(cb[i, -1]), center=self.rotate_center)
                prev_bev[:, i] = t.permute(1, 2, 0).reshape(H * W, -1)
        bev_queries = bev_queries + self.can_bus_mlp(can_bus.to(bev_queries.dtype))[None]
        _, num_cam, c, h, w = feat.shape
        f = feat.flatten(3).permute(1, 0, 3, 2)
        f = f + self.cams_embeds[:, None, None, :].to(f.dtype)
        f = f + self.level_embeds[None, None, 0:1, :].to(f.dtype)
        spatial_shapes = torch.as_tensor([(h, w)], dtype=torch.long, device=feat.device)
        f = f.permute(0, 2, 1, 3)  # (num_cam, HW, bs, c)
        return self.encoder(bev_queries, f, bev_hw, bev_pos, spatial_shapes, prev_bev, shift, lidar2img, img_hw)

    def forward(self, feat, bev_queries, object_query_embed, bev_hw, grid_length, bev_pos, reg_branches,
                prev_bev, can_bus, lidar2img, img_hw):
        bev_embed = self.get_bev_features(feat, bev_queries, bev_hw, grid_length, bev_pos, prev_bev,
                                          can_bus, lidar2img, img_hw)
        bs = feat.size(0)
        query_pos, query = torch.split(object_query_embed, self.dim, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        ref = self.reference_points(query_pos).sigmoid()
        init_ref = ref
        H, W = bev_hw
        inter, inter_ref = self.decoder(query.permute(1, 0, 2), bev_embed.permute(1, 0, 2),
                                        query_pos.permute(1, 0, 2), ref, reg_branches,
                                        torch.tensor([[H, W]], device=feat.device))
        return bev_embed, inter, init_ref, inter_ref


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, num_feats=128, row=50, col=50):
        super().__init__()
        self.row_embed = nn.Embedding(row, num_feats)
        self.col_embed = nn.Embedding(col, num_feats)

    def forward(self, bs, h, w, device):
        x = self.col_embed(torch.arange(w, device=device))
        y = self.row_embed(torch.arange(h, device=device))
        pos = torch.cat((x.unsqueeze(0).repeat(h, 1, 1), y.unsqueeze(1).repeat(1, w, 1)), -1)
        return pos.permute(2, 0, 1).unsqueeze(0).repeat(bs, 1, 1, 1)


class BEVFormerHead(nn.Module):
    def __init__(self, dim=256, bev_h=50, bev_w=50, num_query=900, num_classes=10, code_size=10,
                 num_layers=6, pc_range=PC_RANGE):
        super().__init__()
        self.bev_h, self.bev_w, self.pc_range = bev_h, bev_w, pc_range
        self.real_w = pc_range[3] - pc_range[0]
        self.real_h = pc_range[4] - pc_range[1]
        self.code_weights = nn.Parameter(torch.ones(code_size), requires_grad=False)
        self.positional_encoding = LearnedPositionalEncoding(dim // 2, bev_h, bev_w)
        self.transformer = PerceptionTransformer(dim)
        cls = []
        for _ in range(2):
            cls += [nn.Linear(dim, dim), nn.LayerNorm(dim), nn.ReLU(inplace=True)]
        cls.append(nn.Linear(dim, num_classes))
        reg = []
        for _ in range(2):
            reg += [nn.Linear(dim, dim), nn.ReLU()]
        reg.append(nn.Linear(dim, code_size))
        self.cls_branches = nn.ModuleList([copy.deepcopy(nn.Sequential(*cls)) for _ in range(num_layers)])
        self.reg_branches = nn.ModuleList([copy.deepcopy(nn.Sequential(*reg)) for _ in range(num_layers)])
        self.bev_embedding = nn.Embedding(bev_h * bev_w, dim)
        self.query_embedding = nn.Embedding(num_query, dim * 2)

    def forward(self, feat, prev_bev, can_bus, lidar2img, img_hw):
        bs = feat.size(0)
        bev_pos = self.positional_encoding(bs, self.bev_h, self.bev_w, feat.device).to(feat.dtype)
        bev_embed, hs, init_ref, inter_ref = self.transformer(
            feat, self.bev_embedding.weight, self.query_embedding.weight, (self.bev_h, self.bev_w),
            (self.real_h / self.bev_h, self.real_w / self.bev_w), bev_pos, self.reg_branches,
            prev_bev, can_bus, lidar2img, img_hw)
        hs = hs.permute(0, 2, 1, 3)
        pc = self.pc_range
        lvl = hs.shape[0] - 1  # only the last decoder layer is decoded at test time
        reference = inverse_sigmoid(inter_ref[lvl - 1])
        cls = self.cls_branches[lvl](hs[lvl])
        tmp = self.reg_branches[lvl](hs[lvl]).clone()
        tmp[..., 0:2] = (tmp[..., 0:2] + reference[..., 0:2]).sigmoid()
        tmp[..., 4:5] = (tmp[..., 4:5] + reference[..., 2:3]).sigmoid()
        tmp[..., 0:1] = tmp[..., 0:1] * (pc[3] - pc[0]) + pc[0]
        tmp[..., 1:2] = tmp[..., 1:2] * (pc[4] - pc[1]) + pc[1]
        tmp[..., 4:5] = tmp[..., 4:5] * (pc[5] - pc[2]) + pc[2]
        return bev_embed, cls, tmp


class BEVFormerTiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_backbone = ResNet50C5()
        self.img_neck = FPN1()
        self.pts_bbox_head = BEVFormerHead()

    def forward(self, img, lidar2img, can_bus, img_hw, prev_bev=None):
        """img (bs, 6, 3, H, W) normalized; lidar2img (bs, 6, 4, 4); can_bus (bs, 18) with deltas applied.

        Returns bev_embed, cls_logits (bs, 900, 10), bbox_preds (bs, 900, 10).
        """
        bs, n = img.shape[:2]
        feat = self.img_neck(self.img_backbone(img.flatten(0, 1)))
        feat = feat.view(bs, n, *feat.shape[1:])
        return self.pts_bbox_head(feat, prev_bev, can_bus, lidar2img, img_hw)


def decode(cls_logits, bbox_preds, max_num=300, post_center_range=POST_CENTER_RANGE):
    """NMSFreeCoder: top-k over (query, class) scores; returns boxes in LiDAR frame.

    Output boxes: (x, y, z_center, w, l, h, yaw_mmdet3d, vx, vy), scores, labels.
    """
    scores, idx = cls_logits.sigmoid().view(-1).topk(max_num)
    labels = idx % cls_logits.shape[-1]
    b = bbox_preds[idx // cls_logits.shape[-1]]
    rot = torch.atan2(b[:, 6:7], b[:, 7:8])
    boxes = torch.cat([b[:, 0:2], b[:, 4:5], b[:, 2:4].exp(), b[:, 5:6].exp(), rot, b[:, 8:10]], -1)
    r = torch.tensor(post_center_range, device=boxes.device)
    keep = (boxes[:, :3] >= r[:3]).all(1) & (boxes[:, :3] <= r[3:]).all(1)
    return boxes[keep], scores[keep], labels[keep]


def load_bevformer_tiny(ckpt_path: str, device="cuda") -> BEVFormerTiny:
    model = BEVFormerTiny()
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = sd.get("state_dict", sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # the 4-level level_embeds is kept (only level 0 is used); nothing else may be missing
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing} unexpected={unexpected}")
    return model.eval().to(device)


class TemporalRunner:
    """Replicates BEVFormer.forward_test: carries prev_bev + ego deltas within a scene."""

    def __init__(self, model, video_test_mode=True):
        self.model = model
        self.video = video_test_mode
        self.reset()

    def reset(self):
        self.prev = {"bev": None, "scene": None, "pos": 0, "angle": 0}

    @torch.no_grad()
    def step(self, img, lidar2img, can_bus: np.ndarray, scene_token, img_hw):
        if scene_token != self.prev["scene"]:
            self.prev["bev"] = None
        self.prev["scene"] = scene_token
        if not self.video:
            self.prev["bev"] = None
        can = can_bus.copy()
        tmp_pos, tmp_angle = can[:3].copy(), float(can[-1])
        if self.prev["bev"] is not None:
            can[:3] -= self.prev["pos"]
            can[-1] -= self.prev["angle"]
        else:
            can[-1] = 0
            can[:3] = 0
        dev = img.device
        bev, cls, box = self.model(img, lidar2img.float().to(dev),
                                   torch.as_tensor(can[None], dtype=torch.float32, device=dev), img_hw,
                                   prev_bev=self.prev["bev"])
        self.prev.update(pos=tmp_pos, angle=tmp_angle, bev=bev)
        return cls[0], box[0]
