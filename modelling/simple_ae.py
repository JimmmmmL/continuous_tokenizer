import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List
from modelling.modules import Encoder, Decoder, TimmViTEncoder, TimmViTDecoder

@dataclass
class SimpleModelArgs:
    image_size: int = 256
    codebook_embed_dim: int = 32
    
    # encoder/decoder type
    enc_type: str = 'vit'  # 'cnn' or 'vit'
    dec_type: str = 'vit'  # 'cnn' or 'vit'
    
    # for cnn
    encoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    decoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    z_channels: int = 256
    dropout_p: float = 0.0
    
    # for vit
    encoder_model: str = 'vit_base_patch14_dinov2.lvd142m'
    decoder_model: str = 'vit_base_patch14_dinov2.lvd142m'
    num_latent_tokens: int = 256
    enc_patch_size: int = 16
    dec_patch_size: int = 16
    enc_pretrained: bool = True
    dec_pretrained: bool = False
    to_pixel: str = 'linear'

class SimpleAE(nn.Module):
    def __init__(self, config: SimpleModelArgs):
        super().__init__()
        self.config = config
        
        # Setup encoder
        if config.enc_type == 'cnn':
            self.encoder = Encoder(
                ch_mult=config.encoder_ch_mult, 
                z_channels=config.z_channels, 
                dropout=config.dropout_p
            )
            self.quant_conv = nn.Conv2d(config.z_channels, config.codebook_embed_dim, 1)
        elif config.enc_type == 'vit':
            self.encoder = TimmViTEncoder(
                in_channels=3, 
                num_latent_tokens=config.num_latent_tokens,
                model_name=config.encoder_model,
                model_kwargs={'img_size': config.image_size, 'patch_size': config.enc_patch_size},
                pretrained=config.enc_pretrained,
                tuning_method='full',
            )
            self.quant_conv = nn.Linear(self.encoder.embed_dim, config.codebook_embed_dim)
        
        # Setup decoder
        if config.dec_type == 'cnn':
            self.decoder = Decoder(
                ch_mult=config.decoder_ch_mult, 
                z_channels=config.z_channels, 
                dropout=config.dropout_p
            )
            self.post_quant_conv = nn.Conv2d(config.codebook_embed_dim, config.z_channels, 1)
        elif config.dec_type == 'vit':
            self.decoder = TimmViTDecoder(
                in_channels=3, 
                num_latent_tokens=config.num_latent_tokens,
                model_name=config.decoder_model,
                model_kwargs={'img_size': config.image_size, 'patch_size': config.dec_patch_size},
                pretrained=config.dec_pretrained,
                tuning_method='full',
                codebook_embed_dim=config.codebook_embed_dim,
                to_pixel=config.to_pixel,
            )
            self.post_quant_conv = nn.Linear(config.codebook_embed_dim, self.decoder.embed_dim)
        
        # Check if using movq
        self.use_movq = 'movq' in config.decoder_model

    def encode(self, x):
        h = self.encoder(x)
        latent = self.quant_conv(h)
        return latent

    def decode(self, latent, x=None, h=None, w=None):
        tmp_latent = latent
        latent = self.post_quant_conv(latent)
        if self.use_movq:
            dec = self.decoder(latent, tmp_latent, h, w)
        else:
            dec = self.decoder(latent, None, h, w)
        return dec

    def forward(self, x):
        b, _, h, w = x.size()
        
        # Encode
        latent = self.encode(x)
        
        # Decode
        reconstruction = self.decode(latent, x=x, h=h, w=w)
        
        return reconstruction

# 使用示例
if __name__ == '__main__':
    config = SimpleModelArgs(
        image_size=256,
        codebook_embed_dim=32,
        enc_type='vit',
        dec_type='vit',
        encoder_model='vit_small_patch14_dinov2.lvd142m',
        decoder_model='vit_small_patch14_dinov2.lvd142m',
    )
    
    model = SimpleAE(config)
    model = model.cuda()
    
    # 测试
    x = torch.randn(4, 3, 256, 256).cuda()
    z = model.encode(x)
    print(f"Encoded shape: {z.shape}")
    reconstruction = model(x)
    rec_from_z = model.decode(z, h=256, w=256)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {reconstruction.shape}")
    print(f"Reconstruction shape from z: {rec_from_z.shape}")
    
    # 计算重建损失
    loss = nn.MSELoss()(reconstruction, x)
    print(f"Reconstruction loss: {loss.item()}")