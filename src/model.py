import jax
import haiku as hk
import dataclasses
import jax.numpy as jnp

def layer_norm(x: jnp.ndarray) -> jnp.ndarray:
    ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
    return ln(x)


@dataclasses.dataclass
class DecoderBlock(hk.Module):
    num_heads: int

    def __call__(self, embs: jnp.ndarray, is_training: bool = True):
        _, seq_len, emb_dim = embs.shape

        # Create attention mask
        attn_mask = jnp.full((seq_len, seq_len), float('-inf'))
        attn_mask = jnp.triu(attn_mask, diag=1)

        # MultiHeadAttention
        attn_block = hk.MultiHeadAttention(self.num_heads, emb_dim, model_size=emb_dim)

        # FFN
        ffn = hk.Sequential([
            hk.Linear(emb_dim * 4),
            jax.nn.gelu,
            hk.Linear(emb_dim)
        ])

        # Pass through transformer block
        x = embs

        x1 = attn_block(x, x, x, mask=attn_mask)
        x1 = layer_norm(x + x1)

        x2 = ffn(x1)
        x2 = layer_norm(x1 + x2)

        return x2

@dataclasses.dataclass
class Transformer(hk.Module):
    num_layers: int
    num_heads: int
    emb_dim: int
    drop_rate: float
    num_tokens: int

    def __call__(self, idx: jnp.ndarray, is_training: bool = True):
        _, seq_len = idx.shape

        # Create input embeddings (token_embs + pos_embs)
        token_embs_map = hk.Embed(self.num_tokens, self.emb_dim)

        pos_embs_map = hk.Embed(seq_len, self.emb_dim)
        pos_idx = jnp.arange(seq_len)

        embs = token_embs_map(idx) + pos_embs_map(pos_idx)

        # Define decoder transformer models
        model = hk.Sequential([
            *[DecoderBlock(self.num_heads) for _ in range(self.num_layers)],
            layer_norm,
            hk.Linear(self.num_tokens)
        ])

        return model(embs)
