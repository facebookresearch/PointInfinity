# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# MCC: https://github.com/facebookresearch/MCC
# Point-E: https://github.com/openai/point-e
# RIN: https://arxiv.org/pdf/2212.11972
# This code includes the implementation of our default two-stream model.
# Our default two-stream implementation is based on RIN and MCC,
# Other backbone in the two-stream family such as PerceiverIO will also work.
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from timm.models.vision_transformer import PatchEmbed, Block
from utils import get_2d_sincos_pos_embed, preprocess_img
from modules import Denoiser_backbone

class XYZPosEmbed(nn.Module):
    """
    A Masked Autoencoder with VisionTransformer backbone.
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim

        self.two_d_pos_embed = nn.Parameter(
            torch.zeros(1, 64 + 1, embed_dim), requires_grad=False)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.win_size = 8

        self.pos_embed = nn.Linear(3, embed_dim)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads=num_heads, mlp_ratio=2.0, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
            for _ in range(1)
        ])

        self.invalid_xyz_token = nn.Parameter(torch.zeros(embed_dim,))

        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.cls_token, std=.02)

        two_d_pos_embed = get_2d_sincos_pos_embed(self.two_d_pos_embed.shape[-1], 8, cls_token=True)
        self.two_d_pos_embed.data.copy_(torch.from_numpy(two_d_pos_embed).float().unsqueeze(0))

        torch.nn.init.normal_(self.invalid_xyz_token, std=.02)

    def forward(self, seen_xyz, valid_seen_xyz):
        emb = self.pos_embed(seen_xyz)

        emb[~valid_seen_xyz] = 0.0
        emb[~valid_seen_xyz] += self.invalid_xyz_token

        B, H, W, C = emb.shape
        emb = emb.view(B, H // self.win_size, self.win_size, W // self.win_size, self.win_size, C)
        emb = emb.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.win_size * self.win_size, C)

        emb = emb + self.two_d_pos_embed[:, 1:, :]
        cls_token = self.cls_token + self.two_d_pos_embed[:, :1, :]

        cls_tokens = cls_token.expand(emb.shape[0], -1, -1)
        emb = torch.cat((cls_tokens, emb), dim=1)
        for _, blk in enumerate(self.blocks):
            emb = blk(emb)
        return emb[:, 0].view(B, (H // self.win_size) * (W // self.win_size), -1)

class MCCEncoder(nn.Module):
    """ 
    MCC's RGB and XYZ encoder
    """
    def __init__(self,
                 img_size=224, patch_size=16, in_chans=3, embed_dim=1024, depth=24, 
                 num_heads=16, mlp_ratio=4., norm_layer=nn.LayerNorm, drop_path=0.1):
        super().__init__()

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.n_tokens = num_patches + 1
        self.embed_dim = embed_dim

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

        self.blocks = nn.ModuleList([
            Block(
                embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                drop_path=drop_path
            ) for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        self.cls_token_xyz = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.xyz_pos_embed = XYZPosEmbed(embed_dim, num_heads)

        self.blocks_xyz = nn.ModuleList([
            Block(
                embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                drop_path=drop_path
            ) for i in range(depth)])

        self.norm_xyz = norm_layer(embed_dim)

        self.initialize_weights()

    def initialize_weights(self):

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.cls_token_xyz, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, seen_xyz, valid_seen_xyz):

        # get tokens
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        y = self.xyz_pos_embed(seen_xyz, valid_seen_xyz)

        ##### forward E_XYZ #####
        # append cls token
        cls_token_xyz = self.cls_token_xyz
        cls_tokens_xyz = cls_token_xyz.expand(y.shape[0], -1, -1)

        y = torch.cat((cls_tokens_xyz, y), dim=1)
        # apply Transformer blocks
        for blk in self.blocks_xyz:
            y = blk(y)
        y = self.norm_xyz(y)

        ##### forward E_RGB #####
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)

        x = torch.cat((cls_tokens, x), dim=1)
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # combine encodings
        return torch.cat([x, y], dim=2)

class TwoStreamDenoiser(nn.Module):
    '''
    Full Point diffusion model using MCC's encoders with the Two Stream backbone
    '''
    def __init__(
        self,
        num_points: int = 1024,
        num_latents: int = 256,
        cond_drop_prob: float = 0.1,
        input_channels: int = 6,
        output_channels: int = 6,
        latent_dim: int = 768,
        num_blocks: int = 6,
        num_compute_layers: int = 4,
        **kwargs,
    ):
        super().__init__()
        # define encoders
        self.mcc_encoder = MCCEncoder(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
                                      norm_layer=partial(nn.LayerNorm, eps=1e-6))
        # define backbone
        self.denoiser_backbone = Denoiser_backbone(input_channels=input_channels, output_channels=output_channels, 
                                      num_x=num_points, num_z=num_latents, z_dim=latent_dim, 
                                      num_blocks=num_blocks, num_compute_layers=num_compute_layers)
        self.cond_embed = nn.Sequential(
            nn.LayerNorm(
                normalized_shape=(self.mcc_encoder.embed_dim*2,)
            ),
            nn.Linear(self.mcc_encoder.embed_dim*2, self.denoiser_backbone.z_dim),
        )
        self.cond_drop_prob = cond_drop_prob
        self.num_points = num_points

    def cached_model_kwargs(self, model_kwargs):
        with torch.no_grad():
            cond_dict = {}
            images = preprocess_img(model_kwargs["images"])
            embeddings = self.mcc_encoder(
                images,
                model_kwargs["seen_xyz"],
                model_kwargs["seen_xyz_mask"],
            )
            cond_dict["embeddings"] = embeddings
            if "prev_latent" in model_kwargs:
                cond_dict["prev_latent"] = model_kwargs["prev_latent"]
            return cond_dict

    def forward(
        self,
        x,
        t,
        images=None,
        seen_xyz=None,
        seen_xyz_mask=None,
        embeddings=None,
        prev_latent=None,
    ):
        """
        Forward pass through the model.

        Parameters:
        x: Tensor of shape [B, C, N_points], raw input point cloud.
        t: Tensor of shape [B], time step.
        images (Tensor, optional): A batch of images to condition on.
        seen_xyz (Tensor, optional): A batch of xyz maps to condition on.
        seen_xyz_mask (Tensor, optional): Validity mask for xyz maps.
        embeddings (Tensor, optional): A batch of conditional latent (avoid duplicate 
                                        computation of MCC encoder in diffusion inference)
        prev_latent (Tensor, optional): Self-conditioning latent.

        Returns:
        x_denoised: Tensor of shape [B, C, N_points], denoised point cloud/noise.
        """
        assert images is not None or embeddings is not None, "must specify images or embeddings"
        assert images is None or embeddings is None, "cannot specify both images and embeddings"
        assert x.shape[-1] == self.num_points

        # get the condition vectors with MCC encoders
        if images is not None:
            images = preprocess_img(images)
            cond_vec = self.mcc_encoder(images, seen_xyz, seen_xyz_mask)
        else:
            cond_vec = embeddings
        # condition dropout
        if self.training:
            mask = torch.rand(size=[len(x)]) >= self.cond_drop_prob
            cond_vec = cond_vec * mask[:, None, None].to(cond_vec)
        cond_vec = self.cond_embed(cond_vec)

        # denoiser forward
        x_denoised, latent = self.denoiser_backbone(x.permute(0, 2, 1).contiguous(), t, cond_vec, prev_latent=prev_latent)
        x_denoised = x_denoised.permute(0, 2, 1).contiguous()
        return x_denoised, latent