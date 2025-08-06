from dataclasses import dataclass, field
import torch
import torch.nn as nn
# import sys
# sys.path.append('/data1/jl14087/continuous_tokenizer/')
from transformers import AutoModel, AutoConfig
from modelling.modules.bert_mae_models import TextMAEEncoder, TextMAEDecoder

@dataclass
class TextMAEArgs:
    # --- Language Model Config ---
    vocab_size: int = 50257
    max_seq_len: int = 512
    
    # --- Latent Space Config ---
    codebook_embed_dim: int = 32
    num_latent_tokens: int = 128

    # --- Transformer Config ---
    num_layer: int = 12
    num_heads: int = 12
    width: int = 768
    
    # --- Masked Modeling Config ---
    token_drop_rate: float = 0.4  # Minimum mask rate
    token_drop_rate_max: float = 0.6  # Maximum mask rate

    

class TextMAE(nn.Module):
    """
    A Text-based Masked Autoencoder following the structure:
    Encoder -> Bottleneck -> Decoder
    """
    def __init__(self, config: TextMAEArgs):
        super().__init__()
        self.config = config
        
        # 1. Setup Encoder
        self.encoder = TextMAEEncoder(
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
            num_latent_tokens=config.num_latent_tokens,
            num_layers=config.num_layer,
            num_heads=config.num_heads,
            width=config.width,
            token_drop=config.token_drop_rate,
            token_drop_max=config.token_drop_rate_max,
        )
        
        # 2. Setup the bottleneck (quantization) layers
        # This layer reduces the dimension of the encoder's output latents
        self.quant_conv = nn.Linear(self.encoder.width, config.codebook_embed_dim)
        # This layer expands the dimension back for the decoder
        self.post_quant_conv = nn.Linear(config.codebook_embed_dim, self.encoder.width) # Assuming enc/dec have same dim
        
        # 3. Setup Decoder
        self.decoder = TextMAEDecoder(
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
            num_latent_tokens=config.num_latent_tokens,
            num_layers=config.num_layer,
            num_heads=config.num_heads,
            width=config.width,
        )

        # Ensure post_quant_conv output dimension matches decoder's expected input dimension
        if self.encoder.width != self.decoder.width:
            self.post_quant_conv = nn.Linear(config.codebook_embed_dim, self.decoder.width)

        # Latent tokens
        self.num_latent_tokens = config.num_latent_tokens
        scale = self.encoder.width ** -0.5
        self.latent_tokens = nn.Parameter(
            scale * torch.randn(config.num_latent_tokens, config.width)
        )

        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """ Initialize the weights.
            :param:
                module -> torch.nn.Module: module to initialize
        """
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def encode(self, input_ids, return_mask=False):
        """
        Encodes input_ids to a compressed latent representation.

        Args:
            input_ids (torch.Tensor): Tensor of token IDs of shape (B, S).
            return_mask (bool): If True, also returns the mask used for training.

        Returns:
            torch.Tensor: The compressed latent codes of shape (B, num_latent_tokens, codebook_embed_dim).
            torch.Tensor (optional): The boolean mask of shape (B, S).
        """
        # Get high-dimensional latents and the mask from the encoder
        # h has shape (B, num_latent_tokens, encoder_embed_dim)
        h, mask = self.encoder(input_ids, self.latent_tokens, return_mask=True)

        # Compress to the codebook dimension
        # latent has shape (B, num_latent_tokens, codebook_embed_dim)
        latent = self.quant_conv(h)
        
        if return_mask:
            return latent, mask
        return latent

    def decode(self, latent):
        """
        Decodes a latent representation back to vocabulary logits.

        Args:
            latent (torch.Tensor): The compressed latent codes of shape (B, num_latent_tokens, codebook_embed_dim).

        Returns:
            torch.Tensor: The reconstructed logits of shape (B, S, vocab_size).
        """
        # Expand the latent dimension back to the decoder's expected dimension
        # h_deco has shape (B, num_latent_tokens, decoder_embed_dim)
        h_deco = self.post_quant_conv(latent)
        
        # Decode to get logits
        logits = self.decoder(h_deco)
        return logits

    def forward(self, input_ids, return_mask=False):
        """
        Full forward pass of the autoencoder.

        Args:
            input_ids (torch.Tensor): Tensor of token IDs of shape (B, S).

        Returns:
            torch.Tensor: The reconstructed logits of shape (B, S, vocab_size).
            torch.Tensor: The boolean mask of shape (B, S) used for loss calculation.
        """
        # Encode the input text to get the compressed latent and the mask
        latent, mask = self.encode(input_ids, return_mask=True)
        
        # Decode the latent representation to get reconstruction logits
        reconstruction_logits = self.decode(latent)
        
        # Return both logits and the mask, as the mask is needed to compute the loss
        if return_mask:
            return reconstruction_logits, mask
        return reconstruction_logits
    
if __name__ == '__main__':
    config = TextMAEArgs()
    model = TextMAE(config).cuda()

    # Example input
    input_ids = torch.randint(0, config.vocab_size, (4, 512)) # Batch size of 4, sequence length of 1024
    input_ids = input_ids.cuda()

    # Check latent space encoding
    latent, mask = model.encode(input_ids, return_mask=True)
    print(f"Encoded latent shape: {latent.shape}") # Should be (4, 128, 32) for batch size 4, num_latent_tokens 128, codebook_embed_dim 32

    # Forward pass
    logits = model(input_ids)
    print(f"Logits shape: {logits.shape}") # Shape check. It should be (4, 512, 50257) for logits