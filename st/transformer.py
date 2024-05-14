import torch
import torch.nn.functional as F
from torch import nn, Tensor
import numpy as np
import os
from typing import Optional


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int = 512, n_heads: int = 8, dim_FFN: int = 2048, dropout: int = 0.1, normalize_before: bool = False) -> None:
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout)
        self.FFN = nn.Sequential(
            nn.Linear(d_model, dim_FFN), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim_FFN, d_model))
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.normalize_before = normalize_before

    def forward_post(self, x: Tensor, mask: Optional[Tensor] = None, key_padding_mask: Optional[Tensor] = None, pos: Optional[Tensor] = None) -> Tensor:
        Q = K = V = x
        if pos is not None:
            Q = K = x + pos
        g_x, _ = self.multi_head_attention(
            Q, K, V, attn_mask=mask, key_padding_mask=key_padding_mask)
        x = x + self.dropout(g_x)
        x = self.layer_norm1(x)
        x = x + self.dropout(self.FFN(x))
        x = self.layer_norm2(x)
        return x

    def forward_pre(self, x: Tensor, mask: Optional[Tensor] = None, key_padding_mask: Optional[Tensor] = None, pos: Optional[Tensor] = None) -> Tensor:
        x_1 = self.layer_norm1(x)
        Q = K = V = x_1
        if pos is not None:
            Q = K = x_1 + pos
        x_1, _ = self.multi_head_attention(
            Q, K, V, attn_mask=mask, key_padding_mask=key_padding_mask)
        x = x + self.dropout(x_1)
        x_1 = self.layer_norm2(x)
        x = x + self.dropout(self.FFN(x_1))
        return x

    def forward(self, x: Tensor, mask: Optional[Tensor] = None, key_padding_mask: Optional[Tensor] = None, pos: Optional[Tensor] = None) -> Tensor:
        if self.normalize_before:
            return self.forward_pre(x, mask, key_padding_mask, pos)
        return self.forward_post(x, mask, key_padding_mask, pos)


class Encoder(nn.Module):
    def __init__(self, encoder_layer: EncoderLayer, n_layers: int = 3, norm=None) -> None:
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(n_layers)])
        self.n_layers = n_layers
        self.norm = norm

    def forward(self, x: Tensor, mask: Optional[Tensor] = None, key_padding_mask: Optional[Tensor] = None, pos: Optional[Tensor] = None) -> Tensor:
        for i in range(self.n_layers):
            x = self.layers[i](x, mask, key_padding_mask, pos)
        if self.norm is not None:
            x = self.norm(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int = 512, n_heads: int = 8, dim_FFN: int = 2048, dropout: int = 0.1, normalize_before: bool = False) -> None:
        super(DecoderLayer, self).__init__()
        self.multi_head_attn1 = nn.MultiheadAttention(
            d_model, n_heads, dropout)
        self.multi_head_attn2 = nn.MultiheadAttention(
            d_model, n_heads, dropout)
        self.FFN = nn.Sequential(
            nn.Linear(d_model, dim_FFN), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim_FFN, d_model))
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.layer_norm4 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.normalize_before = normalize_before

    def forward_post(self, content_enc, style_enc, content_mask: Optional[Tensor] = None, style_mask: Optional[Tensor] = None,
                     content_key_padding_mask: Optional[Tensor] = None, style_key_padding_mask: Optional[Tensor] = None,
                     K_pos: Optional[Tensor] = None, Q_pos: Optional[Tensor] = None) -> Tensor:

        Q = content_enc
        K = V = style_enc
        if Q_pos is not None:
            Q = content_enc + Q_pos
        if K_pos is not None:
            K = style_enc + K_pos

        g_x_1, _ = self.multi_head_attn1(
            Q, K, V, attn_mask=content_mask, key_padding_mask=content_key_padding_mask)
        content_enc = content_enc + self.dropout(g_x_1)
        content_enc = self.layer_norm1(content_enc)

        Q = content_enc
        if Q_pos is not None:
            Q = content_enc + Q_pos
        g_x_2, _ = self.multi_head_attn2(
            Q, K, V, attn_mask=style_mask, key_padding_mask=style_key_padding_mask)
        mixed_enc = content_enc + self.dropout(g_x_2)
        mixed_enc = self.layer_norm2(mixed_enc)

        mixed_enc = mixed_enc + self.dropout(self.FFN(mixed_enc))
        mixed_enc = self.layer_norm3(mixed_enc)
        return mixed_enc

    def forward_pre(self, content_enc, style_enc, content_mask: Optional[Tensor] = None, style_mask: Optional[Tensor] = None,
                    content_key_padding_mask: Optional[Tensor] = None, style_key_padding_mask: Optional[Tensor] = None,
                    K_pos: Optional[Tensor] = None, Q_pos: Optional[Tensor] = None) -> Tensor:

        content_enc_1 = self.layer_norm1(content_enc)
        style_enc_1 = self.layer_norm2(style_enc)
        Q = content_enc_1
        K = V = style_enc_1
        if Q_pos is not None:
            Q = content_enc_1 + Q_pos
        if K_pos is not None:
            K = style_enc_1 + K_pos

        g_x_1, _ = self.multi_head_attn1(
            Q, K, V, attn_mask=content_mask, key_padding_mask=content_key_padding_mask)
        content_enc = content_enc + self.dropout(g_x_1)
        content_enc_1 = self.layer_norm3(content_enc)

        Q = content_enc_1
        if Q_pos is not None:
            Q = content_enc_1 + Q_pos
        g_x_2, _ = self.multi_head_attn2(
            Q, K, V, attn_mask=style_mask, key_padding_mask=style_key_padding_mask)
        mixed_enc = content_enc + self.dropout(g_x_2)
        mixed_enc_1 = self.layer_norm4(mixed_enc)

        mixed_enc = mixed_enc + self.dropout(self.FFN(mixed_enc_1))
        return mixed_enc

    def forward(self, content_enc, style_enc, content_mask: Optional[Tensor] = None, style_mask: Optional[Tensor] = None,
                content_key_padding_mask: Optional[Tensor] = None, style_key_padding_mask: Optional[Tensor] = None,
                K_pos: Optional[Tensor] = None, Q_pos: Optional[Tensor] = None) -> Tensor:
        if self.normalize_before:
            return self.forward_pre(content_enc, style_enc, content_mask, style_mask, content_key_padding_mask, style_key_padding_mask, K_pos, Q_pos)
        return self.forward_post(content_enc, style_enc, content_mask, style_mask, content_key_padding_mask, style_key_padding_mask, K_pos, Q_pos)


class Decoder(nn.Module):
    def __init__(self, decoder_layer: DecoderLayer, n_layers: int = 3, norm=None, return_intermediate: bool = False) -> None:
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(n_layers)])
        self.n_layers = n_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, content_enc, style_enc, content_mask: Optional[Tensor] = None, style_mask: Optional[Tensor] = None,
                content_key_padding_mask: Optional[Tensor] = None, style_key_padding_mask: Optional[Tensor] = None,
                K_pos: Optional[Tensor] = None, Q_pos: Optional[Tensor] = None) -> Tensor:
        intermediate = []
        for i in range(self.n_layers):
            content_enc = self.layers[i](content_enc, style_enc, content_mask, style_mask,
                                         content_key_padding_mask, style_key_padding_mask, K_pos, Q_pos)
            if self.return_intermediate:
                if self.norm:
                    intermediate.append(self.norm(content_enc))
                else:
                    intermediate.append(content_enc)

        if self.norm is not None:
            output = self.norm(content_enc)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class Transformer(nn.Module):
    def __init__(self, d_model: int = 512, n_heads: int = 8, dim_FFN: int = 2048, dropout: int = 0.1, normalize_before: bool = False,
                 n_layers: int = 3, return_intermediate: bool = False) -> None:

        super(Transformer, self).__init__()

        encoder_layer = EncoderLayer(
            d_model, n_heads, dim_FFN, dropout, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder_c = Encoder(encoder_layer, n_layers, encoder_norm)
        self.encoder_s = Encoder(encoder_layer, n_layers, encoder_norm)

        decoder_layer = DecoderLayer(
            d_model, n_heads, dim_FFN, dropout, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = Decoder(decoder_layer, n_layers,
                               decoder_norm, return_intermediate)
        
        self.d_model = d_model

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.positional_encoding = nn.Conv2d(512, 512, kernel_size=(1, 1))
        self.average_pool = nn.AdaptiveAvgPool2d(18)

    def forward(self, style, mask, content, pos_embed_c, pos_embed_s):
        content_pool = self.average_pool(content)
        pos_c = self.positional_encoding(content_pool)
        pos_c = F.interpolate(pos_c, mode='bilinear', size=style.shape[-2:])

        style = style.flatten(2).permute(2, 0, 1)
        if pos_embed_s is not None:
            pos_embed_s = pos_embed_s.flatten(2).permute(2, 0, 1)

        content = content.flatten(2).permute(2, 0, 1)
        if pos_embed_c is not None:
            pos_embed_c = pos_embed_c.flatten(2).permute(2, 0, 1)

        style_enc = self.encoder_s(
            style, key_padding_mask=mask, pos=pos_embed_s)
        content_enc = self.encoder_c(content, key_padding_mask=mask, pos=pos_embed_c)
        output = self.decoder(content_enc, style_enc, style_key_padding_mask=mask, K_pos=pos_embed_s, Q_pos=pos_embed_c)[0]

        N, B, C = output.shape
        H = int(np.sqrt(N))
        output = output.permute(1,2,0).view(B, C, -1, H)

        return output
