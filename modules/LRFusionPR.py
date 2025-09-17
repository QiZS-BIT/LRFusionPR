import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.netvlad import NetVLADLoupe
from torchvision.models.resnet import resnet18


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.normalize(input, p=2, dim=self.dim)


class LearnedPositionalEncoding(nn.Module):
    def __init__(self,
                 num_feats,
                 row_num_embed=50,
                 col_num_embed=50):
        super(LearnedPositionalEncoding, self).__init__()
        self.row_embed = nn.Embedding(row_num_embed, num_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_feats)
        self.num_feats = num_feats
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed

    def forward(self, mask):
        h, w = mask.shape[-2:]
        x = torch.arange(w, device=mask.device)
        y = torch.arange(h, device=mask.device)
        x_embed = self.col_embed(x)
        y_embed = self.row_embed(y)
        pos = torch.cat((x_embed.unsqueeze(0).repeat(h, 1, 1), y_embed.unsqueeze(1).repeat(1, w, 1)), dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(mask.shape[0], 1, 1, 1)
        return pos


class LRFusionPR(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        self.conv1_l = pretrained.conv1
        self.bn1_l = pretrained.bn1
        self.maxpool1_l = pretrained.maxpool
        self.layer1_l = pretrained.layer1
        self.layer2_l = pretrained.layer2
        self.layer3_l = pretrained.layer3
        self.layer4_l = pretrained.layer4

        self.conv1_r = copy.deepcopy(pretrained.conv1)
        self.bn1_r = copy.deepcopy(pretrained.bn1)
        self.maxpool1_r = copy.deepcopy(pretrained.maxpool)
        self.layer1_r = copy.deepcopy(pretrained.layer1)
        self.layer2_r = copy.deepcopy(pretrained.layer2)

        self.pos_emb_l = LearnedPositionalEncoding(256, 7, 29)
        self.pos_emb_r = LearnedPositionalEncoding(64, 7, 29)

        self.conv_align_l = nn.Conv2d(512, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.encoder_radar = nn.MultiheadAttention(128, num_heads=8, batch_first=True)
        self.encoder_lidar = nn.MultiheadAttention(128, num_heads=8, batch_first=True)

        self.net_vlad = NetVLADLoupe(
            feature_size=640,
            cluster_size=64,
            output_dim=256,
            gating=True,
            add_batch_norm=False
        )
        self.net_vlad_r = NetVLADLoupe(
            feature_size=128,
            cluster_size=32,
            output_dim=256,
            gating=True,
            add_batch_norm=False
        )

        self.relu = nn.ReLU(inplace=True)
        self.norm1 = nn.LayerNorm(128)
        self.norm2 = nn.LayerNorm(128)

    def forward(self, l_bev, r_bev):
        l_bev = l_bev.view(l_bev.shape[0] * l_bev.shape[1], 3, 200, 900)
        r_bev = r_bev.view(r_bev.shape[0] * r_bev.shape[1], 3, 50, 225)

        feat_r_bev = self.conv1_r(r_bev)
        feat_r_bev = self.bn1_r(feat_r_bev)
        feat_r_bev = self.maxpool1_r(feat_r_bev)
        feat_r_bev = self.layer1_r(feat_r_bev)
        feat_r_bev = self.layer2_r(feat_r_bev)

        feat_l_bev = self.conv1_l(l_bev)
        feat_l_bev = self.bn1_l(feat_l_bev)
        feat_l_bev = self.relu(feat_l_bev)
        feat_l_bev = self.maxpool1_l(feat_l_bev)
        feat_l_bev = self.layer1_l(feat_l_bev)
        feat_l_bev = self.layer2_l(feat_l_bev)
        feat_l_bev = self.layer3_l(feat_l_bev)
        feat_l_bev = self.layer4_l(feat_l_bev)

        # positional embedding
        feat_l_bev_shape = feat_r_bev.shape
        bev_mask_l = torch.zeros((feat_l_bev_shape[0], feat_l_bev_shape[2], feat_l_bev_shape[3]), device=feat_r_bev.device, dtype=feat_r_bev.dtype)
        pos_r = self.pos_emb_r(bev_mask_l)
        feat_r_bev = feat_r_bev + pos_r
        pos_l = self.pos_emb_l(bev_mask_l)
        feat_l_bev = feat_l_bev + pos_l

        feat_l_bev_align = self.conv_align_l(feat_l_bev)
        feat_l_bev_align = feat_l_bev_align.flatten(2).permute(0, 2, 1)
        feat_l_bev = feat_l_bev.flatten(2).permute(0, 2, 1)
        feat_r_bev = feat_r_bev.flatten(2).permute(0, 2, 1)
        feat_r_bev_fuse = self.encoder_radar(feat_r_bev, feat_l_bev_align, feat_l_bev_align)[0]
        feat_r_bev_fuse = self.norm1(feat_r_bev_fuse)
        feat_l_bev_fuse = self.encoder_lidar(feat_l_bev_align, feat_r_bev, feat_r_bev)[0]
        feat_l_bev_fuse = self.norm2(feat_l_bev_fuse)
        feat_r_bev_fuse = feat_r_bev_fuse + feat_l_bev_fuse

        fuse_l_bev = torch.cat([feat_l_bev, feat_r_bev_fuse], dim=-1)
        fuse_l_bev = fuse_l_bev.reshape(feat_l_bev_shape[0], -1, 640)

        # aggregation
        descriptor = self.net_vlad(fuse_l_bev)
        descriptor = F.normalize(descriptor, dim=1)
        descriptor_r = self.net_vlad_r(feat_r_bev)
        descriptor_r = F.normalize(descriptor_r, dim=1)

        descriptor_concat = torch.cat([descriptor, descriptor_r], dim=-1)

        return descriptor, descriptor_r, descriptor_concat

    @classmethod
    def create(cls, weights=None):
        if weights is not None:
            pretrained = resnet18(weights=weights)
        else:
            pretrained = resnet18()
        model = cls(pretrained)
        return model
