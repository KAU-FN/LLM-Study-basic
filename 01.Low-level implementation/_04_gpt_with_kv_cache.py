"""
04_1_kv-cache.ipynb에서 구현한 내용을 다른 기법들과의 비교용으로 재 정리.
"""

import tiktoken
import time
import torch
import torch.nn as nn

from _04_gpt import LayerNorm, FeedForward, generate_text_simple_cached

# 03_multihead_attention.py에서 작성했던 MultiHeadAttention 사용
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        
        self.d_out = d_out
        self.num_heads = num_heads
        self.d_head = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)    # Query weight
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)      # Key weight
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)    # Value weight
        self.out_projection = nn.Linear(d_out, d_out)           # Last output projection(head의 ouput들을 concat한 결과물)
        self.dropout = nn.Dropout(dropout)

        # module을 GPU로 보낼 때, mask도 함께 GPU로 이동.
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

        ####################################
        # NEW
        self.register_buffer("cache_k", None, persistent=False)  # persistent=False로 설정하여 state_dict에 저장되지 않도록 함
        self.register_buffer("cache_v", None, persistent=False)
        self.ptr_current_pos = 0
        ####################################

    def forward(self, x, use_cache=False):
        b, num_tokens, d_in = x.shape   # batch 단위로 처리하므로 batch dimension인 B가 추가

        # input projection (Q, K, V 생성)
            # [B, n_tokens, d_in] -> [B, n_tokens, d_out]
        Q, K_new, V_new = self.W_query(x), self.W_key(x), self.W_value(x)

        # split 함수 사용(head 단위로 분할)
            # [B, n_tokens, d_out] -> [B, num_heads, n_tokens, d_head]
        Q, K_new, V_new = self.split(Q), self.split(K_new), self.split(V_new)  
        ####################################
        # NEW
        if use_cache:
            if self.cache_k is None:
                # cache가 비어있는 경우, 현재 K, V를 그대로 캐시에 저장
                self.cache_k, self.cache_v = K_new, V_new
            else:
                # cache에 이전 K, V가 있는 경우, 현재 K, V를 이어붙임
                self.cache_k = torch.cat([self.cache_k, K_new], dim=2)  # n_tokens 차원(2번째 차원)을 따라 이어붙임
                self.cache_v = torch.cat([self.cache_v, V_new], dim=2)
            K, V = self.cache_k, self.cache_v
        else:
            K, V = K_new, V_new
        ####################################

        # self attention 연산
            # Q * K^T -> [B, num_heads, n_tokens, n_tokens]
        attn_scores = Q @ K.transpose(-2, -1)


        ####################################
        # NEW
        num_tokens_Q = Q.shape[-2]
        num_tokens_K = K.shape[-2]
        if use_cache:
            mask_bool = self.mask.bool()[
                self.ptr_current_pos : self.ptr_current_pos + num_tokens_Q, :num_tokens_K
            ]
            self.ptr_current_pos += num_tokens_Q
        else:
            mask_bool = self.mask.bool()[:num_tokens_Q, :num_tokens_K]

        ####################################

        # masking 처리 -> future token을 보지 못하도록.
        attn_scores.masked_fill_(
            mask_bool, -torch.inf
        )

        # softmax scaling 및 dropout
        attn_weights = torch.softmax(attn_scores / K.shape[-2]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # (Q * K^T) * V 연산
            # [B, num_heads, n_tokens, d_head]
        context_vector = attn_weights @ V

        # concat 함수 사용, head 단위로 분할된 context_vector를 다시 concat
            # [B, num_heads, n_tokens, d_head] -> [B, n_tokens, d_out]
        context_vector = self.concat(context_vector)  
        context_vector = self.out_projection(context_vector)  # output projection

        return context_vector

    def split(self, tensor):
        """
        split tensor by number of heads

        Input shape: [B, n_tokens, d_out]
        Output shape: [B, num_heads, n_tokens, d_tensor]
        """

        b, n_tokens, d_out = tensor.shape

        d_tensor = d_out // self.num_heads
        tensor = tensor.view(b, n_tokens, self.num_heads, d_tensor).transpose(1, 2)

        return tensor
    
    def concat(self, tensor):
        """
        concat tensor by number of heads

        Input shape: [B, num_heads, n_tokens, d_tensor]
        Output shape: [B, n_tokens, d_out]
        """

        b, num_heads, n_tokens, d_tensor = tensor.size()

        d_out = num_heads * d_tensor
        tensor = tensor.transpose(1, 2).contiguous().view(b, n_tokens, d_out)

        return tensor
    
    ####################################
    # NEW
    def reset_cache(self):
        self.cache_k, self.cache_v, = None, None
        self.ptr_current_pos = 0
    ####################################

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.attention = MultiHeadAttention(
            d_in = cfg['embed_dim'],
            d_out = cfg['embed_dim'],
            context_length = cfg['context_length'],
            num_heads = cfg['num_heads'],
            dropout = cfg['drop_rate'],
            qkv_bias = cfg['qkv_bias']
        )

        self.ffn = FeedForward(cfg)

        self.norm1 = LayerNorm(cfg['embed_dim'])
        self.norm2 = LayerNorm(cfg['embed_dim'])

        self.drop_shortcut = nn.Dropout(cfg['drop_rate'])

    def forward(self, x, use_cache=False):
        """
        input -> LayerNorm -> MHA -> Dropout -> skip connection
        -> LayerNorm -> FFN -> Dropout -> skip connection
        -> output
        """
        # attention with skip connection
        residual = x
        x = self.norm1(x)           # LayerNorm

        ####################################
        # NEW
        x = self.attention(x, use_cache=use_cache)       # MHA, [batch, context_length, embed_dim]
        ####################################

        x = self.drop_shortcut(x)   # Dropout
        x = x + residual            # skip(residual) connection

        # FFN with skip connection
        residual = x
        x = self.norm2(x)           # LayerNorm
        x = self.ffn(x)             # FeedForward
        x = self.drop_shortcut(x)   # Dropout
        x = x + residual            # skip(residual) connection

        return x

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.token_embedding = nn.Embedding(cfg['vocab_size'], cfg['embed_dim'])
        self.position_embedding = nn.Embedding(cfg['context_length'], cfg['embed_dim'])
        self.drop_embedding = nn.Dropout(cfg['drop_rate'])

        ####################################
        # NEW
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg['num_layers'])]
        )
        self.current_pos = 0
        ####################################

        self.final_norm = LayerNorm(cfg['embed_dim'])
        self.out_head = nn.Linear(cfg['embed_dim'], cfg['vocab_size'], bias=False)

    def forward(self, in_idx, use_cache=False):
        batch_size, seq_length = in_idx.shape

        # token embedding, positional embedding을 더해서 최종 input embedding 구성
        token_embeddings = self.token_embedding(in_idx)
        
        ####################################
        # NEW
        if use_cache:
            pos_ids = torch.arange(
                self.current_pos, self.current_pos + seq_length, device=in_idx.device, dtype=torch.long
            )
            self.current_pos += seq_length
        else:
            pos_ids = torch.arange(
                0, seq_length, device=in_idx.device, dtype=torch.long
            )
        pos_embeddings = self.position_embedding(pos_ids).unsqueeze(0)
        ####################################

        x = token_embeddings + pos_embeddings   # [batch_size, num_tokens, embed_dim]

        x = self.drop_embedding(x)

        # Transformer block forward pass
        ####################################
        # NEW
        for block in self.transformer_blocks:
            x = block(x, use_cache=use_cache)
        ####################################

        # last layer norm
        x = self.final_norm(x)

        logits = self.out_head(x)

        return logits
    
    ####################################
    # NEW
    def reset_kv_cache(self):
        for block in self.transformer_blocks:
            block.attention.reset_cache()
        self.current_pos = 0
    ####################################

def main():
    GPT_CONFIG_124M = {
        'vocab_size': 50257,        # Vocabulary size
        'context_length': 1024,     # Context(max sequence) length
        'embed_dim': 768,           # Embedding dimension
        'num_heads': 12,            # Number of attention heads
        'num_layers': 12,           # Number of layers(transformer blocks)
        'drop_rate': 0.1,           # Dropout rate
        'qkv_bias': False,          # Q,K,V bias
    }

    torch.manual_seed(62)

    model = GPTModel(GPT_CONFIG_124M)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    model.to(device)
    model.eval()  # disable dropout

    start_context = "O say can you see,"

    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded, device=device).unsqueeze(0)

    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)
        
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()

    token_ids = generate_text_simple_cached(
        model = model,
        idx = encoded_tensor,
        max_new_tokens = 200
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time.time() - start

    decoded_text = tokenizer.decode(token_ids.squeeze(0).tolist())
    print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
    print("\nOutput:", token_ids)
    print("Output length:", len(token_ids[0]))
    print("Output text:", decoded_text)

    print(f"\nTime: {total_time:.2f} sec")
    print(f"{int(len(token_ids[0])/total_time)} tokens/sec")

    if torch.cuda.is_available():
        max_mem_bytes = torch.cuda.max_memory_allocated()
        max_mem_gb = max_mem_bytes / (1024 ** 3)
        print(f"Max memory allocated: {max_mem_gb:.2f} GB")

if __name__ == "__main__":
    main()