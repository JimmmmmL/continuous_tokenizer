import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import peft
from timm.models import create_model
from timm.layers import trunc_normal_
from modelling.modules.timm_vit.to_pixel import ToPixel
from modelling.modules.timm_vit.vision_transformer import Attention



def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
                nn.Linear(hidden_size, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, z_dim),
            )


class TimmViTEncoder(nn.Module):
    def __init__(self, in_channels=3, num_latent_tokens=32,
                 model_name='vit_small_patch14_dinov2.lvd142m',
                 model_kwargs={'img_size': 224, 'patch_size': 14, 'drop_path_rate': 0.0,},
                 pretrained=True, tuning_method='lora', tuning_kwargs={'r': 8},
                 ):
        super().__init__()

        self.model_name = model_name
        assert model_name in ['vit_small_patch14_dinov2.lvd142m', 'vit_base_patch14_dinov2.lvd142m',
                              'vit_large_patch14_dinov2.lvd142m', 'vit_giant_patch14_dinov2.lvd142m',
                              'vit_small_patch14_reg4_dinov2.lvd142m', 'vit_base_patch14_reg4_dinov2.lvd142m',
                              'vit_large_patch14_reg4_dinov2.lvd142m',
                              'vit_giant_patch14_reg4_dinov2.lvd142m', 'vit_base_patch16_clip_224.openai',
                              "vit_base_patch16_clip_224.laion2b", "samvit_base_patch16.sa1b", "eva02_base_patch16_clip_224.merged2b"], f"{model_name} not found"

        # parameters
        self.num_latent_tokens = num_latent_tokens

        # load model
        model = create_model(
            model_name,
            pretrained=pretrained,
            **model_kwargs
        )

        self.embed_dim = model.embed_dim
        # get num of img tokens
        self.num_img_tokens = model.patch_embed.num_patches
        self.num_prefix_tokens = model.num_prefix_tokens

        # tuning method
        if tuning_method == 'full':
            # doing nothing
            self.model = model
        elif tuning_method == 'lora':
            config = peft.LoraConfig(target_modules=r".*\.mlp\.fc\d",
                                     modules_to_save=['norm'], **tuning_kwargs)
            self.model = peft.get_peft_model(model, config)
            # self.model.base_model.model.pos_embed.requires_grad = True
            self.model.print_trainable_parameters()
        elif tuning_method == 'frozen':
            for param in model.parameters():
                param.requires_grad = False
            self.model = model

        if self.num_latent_tokens:
            # latent tokens
            self.latent_tokens = nn.Parameter(torch.zeros(1, self.num_latent_tokens, model.embed_dim))
            nn.init.normal_(self.latent_tokens, std=.02)

            self.latent_pos_embed = nn.Parameter(torch.zeros(1, self.num_latent_tokens, model.embed_dim))
            trunc_normal_(self.latent_pos_embed, std=.02)
        

    def no_weight_decay(self):
        return ['model.pos_embed', 'model.cls_token', 'model.dist_token', 'latent_tokens', 'latent_pos_embed']

    def forward(self, x, return_img_token=False):

        # get tokens
        x = self.model.patch_embed(x)
        
        
        if not 'eva02' in self.model_name:
            x = self.model._pos_embed(x)
            x = self.model.patch_drop(x)
        else:
            x, _ = self.model._pos_embed(x)

        if self.num_latent_tokens:
            # insert latent tokens
            z = self.latent_tokens.expand(x.size(0), -1, -1)
            x = torch.cat([x, z + self.latent_pos_embed], dim=1)
            
        # pre layer norm
        if not 'eva02' in self.model_name:
            x = self.model.norm_pre(x)
            
        for i, blk in enumerate(self.model.blocks):
            x = blk(x)
                
        # x = self.model.blocks(x)
        if not 'eva02' in self.model_name:
            x = self.model.norm(x)
        else:
            x = self.model.fc_norm(x)

        if self.num_latent_tokens:
            # get z tokens as out
            out = x[:, -self.num_latent_tokens:]
        else:
            # get img tokens as out
            out = x[:, self.num_prefix_tokens:]
        

        return out


class TimmViTDecoder(nn.Module):
    def __init__(self, in_channels=3,
                 model_name='vit_small_patch14_dinov2.lvd142m',
                 model_kwargs={'img_size': 224, 'patch_size': 14, 'drop_path_rate': 0.0}, pretrained=True,
                 tuning_method='lora', tuning_kwargs={'r': 8},
                 num_latent_tokens=32, to_pixel='linear',
                 ):
        super().__init__()

        assert model_name in ['vit_small_patch14_dinov2.lvd142m', 'vit_base_patch14_dinov2.lvd142m',
                              'vit_large_patch14_dinov2.lvd142m', 'vit_giant_patch14_dinov2.lvd142m',
                              'vit_small_patch14_reg4_dinov2.lvd142m', 'vit_base_patch14_reg4_dinov2.lvd142m',
                              'vit_large_patch14_reg4_dinov2.lvd142m', 'vit_giant_patch14_reg4_dinov2.lvd142m', 
                              'vit_base_patch16_clip_224.openai']

        # model_kwargs['num_latent_tokens'] = num_latent_tokens
        model = create_model(
            model_name,
            pretrained=pretrained,
            **model_kwargs
        )
        if model_name == "vit_base_patch16_clip_224.openai":
            model.head = None
        
        self.embed_dim = model.embed_dim
        # get num of img tokens
        self.num_img_tokens = model.patch_embed.num_patches
        self.num_prefix_tokens = model.num_prefix_tokens
        self.num_latent_tokens = num_latent_tokens
        
        # tuning method
        if tuning_method == 'full':
            # doing nothing
            self.model = model
        elif tuning_method == 'lora':
            config = peft.LoraConfig(target_modules=r".*\.mlp\.fc\d",
                                     modules_to_save=['patch_embed.proj', 'patch_embed.norm', 'norm'], **tuning_kwargs)
            self.model = peft.get_peft_model(model, config)
            # self.model.base_model.model.pos_embed.requires_grad = True
            self.model.print_trainable_parameters()
        elif tuning_method == 'frozen':
            for param in model.parameters():
                param.requires_grad = False
            self.model = model

        # latent tokens
        self.mask_token = nn.Parameter(torch.zeros(1, 1, model.embed_dim))
        nn.init.normal_(self.mask_token, std=.02)

        self.latent_pos_embed = nn.Parameter(torch.zeros(1, self.num_latent_tokens, model.embed_dim))
        trunc_normal_(self.latent_pos_embed, std=.02)

        # to pixel
        self.to_pixel = ToPixel(to_pixel=to_pixel, img_size=model_kwargs['img_size'], in_channels=in_channels,
                                in_dim=model.embed_dim, patch_size=model_kwargs['patch_size'])

        del self.model.patch_embed.proj.bias
        del self.model.patch_embed.proj.weight

    def no_weight_decay(self):
        return ['model.pos_embed', 'model.cls_token', 'model.dist_token', 'mask_token', 'latent_pos_embed']

    @property
    def last_layer(self):
        return self.to_pixel.model.weight

    def forward(self, z):

        # mask tokens
        x = self.mask_token.expand(z.size(0), self.num_img_tokens, -1)
        # x = self.mask_token.expand(z.size(0), -1, -1)
        x = self.model._pos_embed(x)
        x = self.model.patch_drop(x)
        
        z = z + self.latent_pos_embed
        x = torch.cat([x, z], dim=1)

        x = self.model.norm_pre(x)
        x = self.model.blocks(x)
        x = self.model.norm(x)

        x = x[:, self.num_prefix_tokens:self.num_img_tokens + self.num_prefix_tokens]

        out = self.to_pixel(x)

        return out


if __name__ == '__main__':
    encoder = TimmViTEncoder(num_latent_tokens=256)
    decoder = TimmViTDecoder(num_latent_tokens=256)
    
    x = torch.randn(1, 3, 224, 224)
    
    o = encoder(x)
    print(o.shape)
    r = decoder(o)