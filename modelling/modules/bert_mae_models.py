import torch
import torch.nn as nn
import numpy as np
from scipy import stats
from einops import rearrange

# We need the transformers library for BERT and peft for LoRA
# pip install transformers peft accelerate
import peft
from collections import OrderedDict

def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model,
            n_head,
            mlp_ratio = 4.0,
            act_layer = nn.GELU,
            norm_layer = nn.LayerNorm
        ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.mlp_ratio = mlp_ratio
        # optionally we can disable the FFN
        if mlp_ratio > 0:
            self.ln_2 = norm_layer(d_model)
            mlp_width = int(d_model * mlp_ratio)
            self.mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(d_model, mlp_width)),
                ("gelu", act_layer()),
                ("c_proj", nn.Linear(mlp_width, d_model))
            ]))

    def attention(
            self,
            x: torch.Tensor
    ):
        return self.attn(x, x, x, need_weights=False)[0]

    def forward(
            self,
            x: torch.Tensor,
    ):
        attn_output = self.attention(x=self.ln_1(x))
        x = x + attn_output
        if self.mlp_ratio > 0:
            x = x + self.mlp(self.ln_2(x))
        return x

if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
    ATTENTION_MODE = 'flash'
else:
    try:
        import xformers
        import xformers.ops
        ATTENTION_MODE = 'xformers'
    except:
        ATTENTION_MODE = 'math'
print(f'attention mode is {ATTENTION_MODE}')

def _expand_token(token, batch_size: int):
    return token.unsqueeze(0).expand(batch_size, -1, -1)

class TextMAEEncoder(nn.Module):
    """
    An MAE Encoder adapted for Text, using a BERT-style model as the backbone.
    """
    def __init__(self,
                 vocab_size=50257, # GPT-2 Vocab Size
                 max_seq_len=256,
                 num_latent_tokens=32,
                 num_layers=12,
                 num_heads=12,
                 width=768,
                 token_drop=0.4,
                 token_drop_max=0.6,
                 ):
        super().__init__()

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.token_drop = token_drop
        self.token_drop_max = token_drop_max
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.num_latent_tokens = num_latent_tokens
        self.width = width


        self.token_embed = nn.Embedding(vocab_size, width)

        scale = width ** -0.5

        self.positional_embedding = nn.Parameter(scale * torch.randn(max_seq_len, width))
        self.latent_token_positional_embedding = nn.Parameter(scale * torch.randn(num_latent_tokens, width))

        self.ln_pre = nn.LayerNorm(width)

        self.transformer = nn.ModuleList()
        for i in range(num_layers):
            self.transformer.append(
                ResidualAttentionBlock(
                self.width, self.num_heads, mlp_ratio=4.0
            ))
        self.ln_post = nn.LayerNorm(width)


        # Token dropping (masking) logic is preserved
        self.token_drop = token_drop > 0.0
        if self.token_drop:
            self.mask_ratio_generator = stats.truncnorm((token_drop - token_drop_max) / 0.25, 0, loc=token_drop_max, scale=0.25)
            self.mask_token_embed = nn.Parameter(torch.zeros(1, 1, self.width))
            nn.init.normal_(self.mask_token_embed, std=.02)


    def sample_orders(self, bsz, seq_len):
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(seq_len)))
            np.random.shuffle(order)
            orders.append(order)
        return torch.from_numpy(np.array(orders)).long()

    def random_masking(self, x, orders):
        bsz, seq_len, embed_dim = x.shape
        mask_rate = self.mask_ratio_generator.rvs(1)[0]
        num_masked_tokens = int(np.ceil(seq_len * mask_rate))
        mask = torch.zeros(bsz, seq_len, device=x.device, dtype=torch.bool)
        masked_indices = orders[:, :num_masked_tokens]
        mask.scatter_(1, masked_indices, True)
        return mask

    def forward(self, input_ids, latent_tokens, return_mask=False):
        # input_ids shape: (batch_size, seq_len)
        
        # 1. Get token embeddings
        # The `model.embeddings` layer handles token, position, and token_type embeddings
        batch_size = input_ids.size(0)
        x = input_ids
        x = self.token_embed(x) # (B, S, D)
        
        # Add positional embeddings
        x = x + self.positional_embedding.to(x.device).to(x.dtype)  # (B, S, D)


        # 2. Apply token masking (if training)
        mask = None
        if self.token_drop and self.training:
            orders = self.sample_orders(bsz=x.size(0), seq_len=x.size(1)).to(x.device)
            mask = self.random_masking(x, orders).unsqueeze(-1)
            x = torch.where(mask.bool(), self.mask_token_embed, x)

        # 3. Add latent tokens
        latent_tokens = _expand_token(latent_tokens, batch_size).to(x.dtype).to(x.device)
        latent_tokens = latent_tokens + self.latent_token_positional_embedding.to(x.device).to(x.dtype)
        x = torch.cat([x, latent_tokens], dim=1)

        x = self.ln_pre(x)  # LayerNorm before transformer

        # 4. Pass through transformer layers
        x = x.permute(1, 0, 2)  # (S, B, D) for transformer compatibility
        for layer in self.transformer:
            x = layer(x)
        x = x.permute(1, 0, 2)  # (B, S, D) back to original shape

        latent_tokens = x[:, -self.num_latent_tokens:]  # Get the latent tokens
        latent_tokens = self.ln_post(latent_tokens)  # Final LayerNorm on latent tokens
        out = latent_tokens
        if return_mask:
            return out, mask
        else:
            return out


class TextMAEDecoder(nn.Module):
    """
    An MAE Decoder adapted for Text. It takes latent vectors and reconstructs
    the original token sequence by predicting token IDs.
    """
    def __init__(self,
                 vocab_size=50257,
                 max_seq_len=256,
                 num_latent_tokens=32,
                 width=768,
                 num_layers=12,
                 num_heads=12,
                 ):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.num_latent_tokens = num_latent_tokens
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.width = width
        
        scale = width ** -0.5

        self.positional_embedding = nn.Parameter(scale * torch.randn(max_seq_len, width))
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, width))
        self.latent_token_positional_embedding = nn.Parameter(scale * torch.randn(num_latent_tokens, width))

        self.ln_pre = nn.LayerNorm(width)

        self.transformer = nn.ModuleList()
        for i in range(num_layers):
            self.transformer.append(
                ResidualAttentionBlock(
                    width, num_heads, mlp_ratio=4.0
                )
            )
        self.ln_post = nn.LayerNorm(width)

        self.decoder_head = nn.Linear(self.width, self.vocab_size)


    def forward(self, z):
        batch_size, seq_len, _ = z.size()

        masked_tokens = self.mask_token.repeat(batch_size, self.max_seq_len, 1).to(z.device).to(z.dtype)
        masked_tokens = masked_tokens + self.positional_embedding.to(z.device).to(z.dtype)

        z = z + self.latent_token_positional_embedding[:seq_len].to(z.device).to(z.dtype)
        z = torch.cat([masked_tokens, z], dim=1)  # Concatenate masked tokens with latent tokens

        z = self.ln_pre(z)  # LayerNorm before transformer

        z = z.permute(1, 0, 2)  # (S, B, D) for transformer compatibility
        for layer in self.transformer:
            z = layer(z)
        z = z.permute(1, 0, 2)

        z = z[:, :self.max_seq_len]  # Keep only the relevant part
        z = self.ln_post(z)  # Final LayerNorm

        logits = self.decoder_head(z)

        return logits

if __name__ == '__main__':
    # --- Example Usage ---
    BATCH_SIZE = 4
    MAX_SEQ_LEN = 128
    VOCAB_SIZE = 50257 # GPT-2
    NUM_LATENT_TOKENS = 32

    # 1. Create dummy input data (token IDs)
    # In a real scenario, this comes from a tokenizer
    dummy_input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, MAX_SEQ_LEN))

    # 2. Instantiate the models
    encoder = TextMAEEncoder(
        vocab_size=VOCAB_SIZE,
        max_seq_len=MAX_SEQ_LEN,
        num_latent_tokens=NUM_LATENT_TOKENS,
        num_layers=12,  # BERT-style encoder
        num_heads=12,
        width=768,  # BERT base width
        token_drop=0.4,  # Example token drop rate
        token_drop_max=0.6  # Example max token drop rate
    )

    decoder = TextMAEDecoder(
        vocab_size=VOCAB_SIZE,
        max_seq_len=MAX_SEQ_LEN,
        num_latent_tokens=NUM_LATENT_TOKENS,
        width=768,  # BERT base width
        num_layers=12,  # BERT-style decoder
        num_heads=12
    )
    
    encoder.train() # Set to train mode to enable masking
    decoder.train()

    latent_tokens = torch.randn(NUM_LATENT_TOKENS, 768)  # Randomly initialized latent tokens
    # 3. Forward pass
    latent_representation, mask = encoder(dummy_input_ids, latent_tokens=latent_tokens, return_mask=True)
    reconstructed_logits = decoder(latent_representation)

    print("--- Encoder ---")
    print(f"Input IDs shape: {dummy_input_ids.shape}")
    print(f"Latent representation shape: {latent_representation.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Number of masked tokens in first example: {mask[0].sum()}")
    
    print("\n--- Decoder ---")
    print(f"Reconstructed logits shape: {reconstructed_logits.shape}")

    # 4. Calculate Loss (Example)
    # The loss is CrossEntropy, calculated only on the masked tokens
    loss_fn = nn.CrossEntropyLoss()
    
    # Reshape for CrossEntropyLoss: (N, C, ...) -> (B * S, C)
    logits_flat = reconstructed_logits.view(-1, VOCAB_SIZE)
    # Target should be the original tokens: (B * S)
    labels_flat = dummy_input_ids.view(-1)


    # Calculate loss regardless of masking 
    loss = loss_fn(logits_flat, labels_flat)
    print(f"Loss: {loss.item()}")