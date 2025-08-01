import torch
import torch.nn as nn
import numpy as np
from scipy import stats
from einops import rearrange

# We need the transformers library for BERT and peft for LoRA
# pip install transformers peft accelerate
from transformers import AutoModel, AutoConfig
import peft

# Note: The original RoPE helper functions are not needed as we rely on BERT's
# native positional embeddings. I've removed them for clarity.

def extend_positional_embeddings(model, new_max_length):
    """
    Extends the positional embeddings of a BERT-like model to a new maximum length.
    """
    # Get the original embedding layer and its properties
    original_embeddings = model.embeddings.position_embeddings
    original_max_length = original_embeddings.num_embeddings
    embedding_dim = original_embeddings.embedding_dim

    if new_max_length <= original_max_length:
        print(f"New length {new_max_length} is not greater than original {original_max_length}. No changes made.")
        return model

    # Create a new embedding layer with the new size
    new_embeddings = nn.Embedding(new_max_length, embedding_dim)
    # Initialize with the same device and dtype
    new_embeddings.to(original_embeddings.weight.device, dtype=original_embeddings.weight.dtype)
    
    # Copy the old weights into the new layer
    new_embeddings.weight.data[:original_max_length, :] = original_embeddings.weight.data
    
    # Initialize the new positional embeddings (positions > original_max_length)
    # A common practice is to just initialize them like the original model did.
    # We can copy the initialization from the last few existing vectors as a simple heuristic
    # or just let them be randomly initialized as per nn.Embedding default.
    # For fine-tuning, random initialization is usually sufficient.
    # Here, we will let them be randomly initialized by default.
    
    # Replace the old layer with the new one
    model.embeddings.position_embeddings = new_embeddings
    
    # Update the model's config to reflect the change
    model.config.max_position_embeddings = new_max_length
    
    print(f"Resized positional embeddings from {original_max_length} to {new_max_length}")
    return model

class TextMAEEncoder(nn.Module):
    """
    An MAE Encoder adapted for Text, using a BERT-style model as the backbone.
    """
    def __init__(self,
                 vocab_size=50257, # GPT-2 Vocab Size
                 max_seq_len=256,
                 num_latent_tokens=32,
                 model_name='bert-base-uncased',
                 pretrained=True,
                 tuning_method='full', # 'full', 'lora', 'frozen'
                 tuning_kwargs={'r': 8},
                 token_drop=0.4,
                 token_drop_max=0.6,
                 ):
        super().__init__()

        self.model_name = model_name

        # 1. Load a BERT-style model from HuggingFace Transformers
        config = AutoConfig.from_pretrained(model_name)
        if pretrained:
            model = AutoModel.from_pretrained(model_name)
        else:
            model = AutoModel.from_config(config)

        # Resize token embeddings to match the custom vocab_size
        if model.config.vocab_size != vocab_size:
            model.resize_token_embeddings(vocab_size)

        # --- FIX for sequence length ---
        if max_seq_len > model.config.max_position_embeddings:
            model = extend_positional_embeddings(model, max_seq_len)
            
        # --- Core Parameter Changes from Vision to Text ---
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embed_dim = model.config.hidden_size # e.g., 768 for bert-base
        self.num_prefix_tokens = 1 # For the [CLS] token

        # 2. Adapt the embedding layer
        # The original patch_embed is replaced by a standard token embedding.
        # We can just use the one from the BERT model itself.
        # Note: BERT's `embeddings` layer includes word, position, and token_type embeddings.
        self.token_embed = model.embeddings

        # --- Tuning Method ---
        if tuning_method == 'full':
            self.model = model
        elif tuning_method == 'lora':
            # Target modules in BERT are typically in the attention blocks
            lora_config = peft.LoraConfig(target_modules=r".*attention\.self\.(query|key|value)",
                                          modules_to_save=['embeddings', 'norm'], **tuning_kwargs)
            self.model = peft.get_peft_model(model, lora_config)
            self.model.print_trainable_parameters()
        elif tuning_method == 'frozen':
            for param in model.parameters():
                param.requires_grad = False
            self.model = model

        # --- Logic kept from the original implementation ---
        self.num_latent_tokens = num_latent_tokens
        if self.num_latent_tokens:
            self.latent_tokens = nn.Parameter(torch.zeros(1, self.num_latent_tokens, self.embed_dim))
            nn.init.normal_(self.latent_tokens, std=.02)
            # Positional embedding for the latent tokens
            self.latent_pos_embed = nn.Parameter(torch.zeros(1, self.num_latent_tokens, self.embed_dim))
            nn.init.normal_(self.latent_pos_embed, std=.02)

        # Token dropping (masking) logic is preserved
        self.token_drop = token_drop > 0.0
        if self.token_drop:
            self.mask_ratio_generator = stats.truncnorm((token_drop - token_drop_max) / 0.25, 0, loc=token_drop_max, scale=0.25)
            self.mask_token_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            nn.init.normal_(self.mask_token_embed, std=.02)

        # RoPE/APE logic is removed, as we rely on BERT's internal positional embeddings.
        # The architecture is simpler and more standard for text models this way.

    def no_weight_decay(self):
        # Typical parameters to exclude from weight decay for BERT-style models
        return ['model.embeddings.word_embeddings.weight', 'model.embeddings.position_embeddings.weight', 
                'model.embeddings.token_type_embeddings.weight', 'bias', 'LayerNorm.weight',
                'latent_tokens', 'latent_pos_embed']

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

    def forward(self, input_ids, return_mask=False):
        # input_ids shape: (batch_size, seq_len)
        
        # 1. Get token embeddings
        # The `model.embeddings` layer handles token, position, and token_type embeddings
        x = self.token_embed(input_ids=input_ids) # (B, S, D)

        # 2. Apply token masking (if training)
        mask = None
        if self.token_drop and self.training:
            orders = self.sample_orders(bsz=x.size(0), seq_len=x.size(1)).to(x.device)
            mask = self.random_masking(x, orders).unsqueeze(-1)
            x = torch.where(mask.bool(), self.mask_token_embed, x)

        # 3. Add latent tokens
        if self.num_latent_tokens:
            z = self.latent_tokens.expand(x.size(0), -1, -1)
            z = z + self.latent_pos_embed
            x = torch.cat([x, z], dim=1) # (B, S + num_latents, D)

        # Create attention mask for the transformer
        # All tokens (real and latent) can attend to each other
        attention_mask = torch.ones(x.shape[:2], device=x.device)
        extended_attention_mask = self.model.get_extended_attention_mask(attention_mask, x.shape, x.device)

        # 4. Pass through the BERT encoder layers
        # We use the encoder part of the BERT model
        encoder_outputs = self.model.encoder(
            x,
            attention_mask=extended_attention_mask,
        )

        x = encoder_outputs.last_hidden_state

        # 5. Extract output tokens
        if self.num_latent_tokens:
            # Get z tokens as output
            out = x[:, -self.num_latent_tokens:]
        else:
            # Get all text tokens as output (excluding [CLS] if we want)
            out = x[:, self.num_prefix_tokens:self.max_seq_len+self.num_prefix_tokens]
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
                 model_name='bert-base-uncased',
                 num_decoder_layers=6,
                 pretrained=True,
                 tuning_method='full',
                 tuning_kwargs={'r': 8}
                 ):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.num_latent_tokens = num_latent_tokens
        
        # To create a decoder with a different number of layers, we first load the
        # pretrained model and then modify its layer structure.
        if pretrained:
            decoder_model = AutoModel.from_pretrained(model_name)
        else:
            config = AutoConfig.from_pretrained(model_name)
            decoder_model = AutoModel.from_config(config)

        # Resize token embeddings to match the custom vocab_size ---
        if decoder_model.config.vocab_size != vocab_size:
            decoder_model.resize_token_embeddings(vocab_size)
        
        # --- FIX for sequence length ---
        # The decoder needs its input sequence length, not the encoder's.
        # Its input is `max_seq_len` mask tokens + `num_latent_tokens`
        decoder_input_len = max_seq_len + num_latent_tokens
        if decoder_input_len > decoder_model.config.max_position_embeddings:
            decoder_model = extend_positional_embeddings(decoder_model, decoder_input_len)
            
        # --- Modify the number of layers for the decoder ---
        # Truncate the list of layers to the desired number
        decoder_model.encoder.layer = decoder_model.encoder.layer[:num_decoder_layers]
        decoder_model.config.num_hidden_layers = num_decoder_layers # Update config for consistency

        self.embed_dim = decoder_model.config.hidden_size
        self.num_prefix_tokens = 0 # Decoder doesn't need a CLS token for its sequence

        # --- Tuning Method ---
        if tuning_method == 'full':
            self.model = decoder_model
        elif tuning_method == 'lora':
            lora_config = peft.LoraConfig(target_modules=r".*attention\.self\.(query|key|value)",
                                          modules_to_save=['embeddings', 'norm'], **tuning_kwargs)
            self.model = peft.get_peft_model(decoder_model, lora_config)
            self.model.print_trainable_parameters()
        elif tuning_method == 'frozen':
            for param in decoder_model.parameters():
                param.requires_grad = False
            self.model = decoder_model

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.normal_(self.mask_token, std=.02)

        self.latent_pos_embed = nn.Parameter(torch.zeros(1, self.num_latent_tokens, self.embed_dim))
        nn.init.normal_(self.latent_pos_embed, std=.02)

        self.decoder_norm = nn.LayerNorm(self.embed_dim)
        self.decoder_head = nn.Linear(self.embed_dim, self.vocab_size)
        
    def no_weight_decay(self):
        return ['model.embeddings.word_embeddings.weight', 'model.embeddings.position_embeddings.weight', 
                'model.embeddings.token_type_embeddings.weight', 'bias', 'LayerNorm.weight',
                'mask_token', 'latent_pos_embed']

    @property
    def last_layer(self):
        return self.decoder_head

    def forward(self, z):
        x = self.mask_token.expand(z.size(0), self.max_seq_len, -1)
        
        position_ids = torch.arange(self.max_seq_len, device=z.device).expand(z.size(0), -1)
        
        # We need to manually add positional embeddings since we're not passing `input_ids`
        pos_embed = self.model.embeddings.position_embeddings(position_ids)
        x = x + pos_embed

        z = z + self.latent_pos_embed
        
        x = torch.cat([x, z], dim=1)

        attention_mask = torch.ones(x.shape[:2], device=x.device)
        extended_attention_mask = self.model.get_extended_attention_mask(attention_mask, x.shape)
        
        decoder_outputs = self.model.encoder(
            x,
            attention_mask=extended_attention_mask,
        )
        x = decoder_outputs.last_hidden_state
        
        x = x[:, :self.max_seq_len]

        x = self.decoder_norm(x)

        logits = self.decoder_head(x)

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
        model_name='bert-base-uncased' # Could be 'distilbert-base-uncased' for a lighter model
    )

    decoder = TextMAEDecoder(
        vocab_size=VOCAB_SIZE,
        max_seq_len=MAX_SEQ_LEN,
        num_latent_tokens=NUM_LATENT_TOKENS,
        model_name='bert-base-uncased', # Using same arch for simplicity
        num_decoder_layers=4 # A shallower decoder
    )
    
    encoder.train() # Set to train mode to enable masking
    decoder.train()

    # 3. Forward pass
    latent_representation, mask = encoder(dummy_input_ids, return_mask=True)
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