import tiktoken
import torch
import torch.nn as nn
import time
from torch.utils.data import DataLoader, Dataset

#########################################
# from Ch.02
#########################################
class GPTDatasetV1(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # 전체 text를 Tokenize
        token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        assert len(token_ids) > max_length, "Number of tokenized inputs must at least be equal to max_length+1"

        # sliding window를 통해 text를 max_length 길이의 겹치는 sequence로 분할
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    

def create_dataloader_v1(text, batch_size=4, max_length=256, stride=128, 
                         shuffle=True, drop_last=True, num_workers=0):
    # tokenizer 초기화
    tokenizer = tiktoken.get_encoding("gpt2")

    # dataset 생성
    dataset = GPTDatasetV1(text, tokenizer, max_length, stride)

    # dataloader 생성
    dataloader = DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        drop_last = drop_last,
        num_workers = num_workers
    )

    return dataloader

#########################################
# from Ch.03 & Ch.04(KV-cache)
#########################################
class LayerNorm(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(embed_dim))    # gamma
        self.shift = nn.Parameter(torch.zeros(embed_dim))   # beta
        
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        out = self.scale * x_norm + self.shift

        return out
    
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(cfg['embed_dim'], 4 * cfg['embed_dim']),
            nn.GELU(),
            nn.Linear(4 * cfg['embed_dim'], cfg['embed_dim'])
        )
    
    def forward(self, x):
        return self.layers(x)

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
        self.register_buffer("cache_k", None, persistent=False)  # persistent=False로 설정하여 state_dict에 저장되지 않도록 함
        self.register_buffer("cache_v", None, persistent=False)
        self.ptr_current_pos = 0

    def forward(self, x, use_cache=False):
        b, num_tokens, d_in = x.shape   # batch 단위로 처리하므로 batch dimension인 B가 추가

        # input projection (Q, K, V 생성)
            # [B, n_tokens, d_in] -> [B, n_tokens, d_out]
        Q, K_new, V_new = self.W_query(x), self.W_key(x), self.W_value(x)

        # split 함수 사용(head 단위로 분할)
            # [B, n_tokens, d_out] -> [B, num_heads, n_tokens, d_head]
        Q, K_new, V_new = self.split(Q), self.split(K_new), self.split(V_new)  

        # KV-cache handling
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

        # self attention 연산
            # Q * K^T -> [B, num_heads, n_tokens, n_tokens]
        attn_scores = Q @ K.transpose(-2, -1)

        # mask 생성
        num_tokens_Q = Q.shape[-2]
        num_tokens_K = K.shape[-2]
        if use_cache:
            mask_bool = self.mask.bool()[
                self.ptr_current_pos : self.ptr_current_pos + num_tokens_Q, :num_tokens_K
            ]
            self.ptr_current_pos += num_tokens_Q
        else:
            mask_bool = self.mask.bool()[:num_tokens_Q, :num_tokens_K]

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
    
    def reset_cache(self):
        self.cache_k, self.cache_v, = None, None
        self.ptr_current_pos = 0

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
        x = self.attention(x, use_cache=use_cache)       # MHA, [batch, context_length, embed_dim]
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

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg['num_layers'])]
        )
        self.current_pos = 0

        self.final_norm = LayerNorm(cfg['embed_dim'])
        self.out_head = nn.Linear(cfg['embed_dim'], cfg['vocab_size'], bias=False)

    def forward(self, in_idx, use_cache=False):
        batch_size, seq_length = in_idx.shape

        # token embedding, positional embedding을 더해서 최종 input embedding 구성
        token_embeddings = self.token_embedding(in_idx)
        
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
        x = token_embeddings + pos_embeddings   # [batch_size, num_tokens, embed_dim]
        x = self.drop_embedding(x)

        # Transformer block forward pass
        for block in self.transformer_blocks:
            x = block(x, use_cache=use_cache)

        # last layer norm
        x = self.final_norm(x)

        logits = self.out_head(x)

        return logits
    
    def reset_kv_cache(self):
        for block in self.transformer_blocks:
            block.attention.reset_cache()
        self.current_pos = 0
    
def generate_text_simple_cached(model, idx, max_new_tokens, context_size=None, use_cache=True):
    """
    Greedy decoding with KV-cache support
    """
    model.eval()
    context_length = context_size or model.position_embedding.num_embeddings

    with torch.no_grad():
        if use_cache:
            # Initialize cache with full prompt
            model.reset_kv_cache()
            logits = model(idx[:, -context_length:], use_cache=True)

            for _ in range(max_new_tokens):
                # 가장 높은 log-probability를 가진 token 선택 (greedy sampling)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)

                # 새로운 token을 입력 sequence에 추가
                idx = torch.cat([idx, next_idx], dim=1)

                # model에는 새 token만을 전달
                logits = model(next_idx, use_cache=True)
        
        else:
            for _ in range(max_new_tokens):
                logits = model(idx[:, -context_length:], use_cache=False)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_idx], dim=1)

    return idx

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
    model.eval()  # disable dropout

    start_context = "O say can you see,"

    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    out = generate_text_simple_cached(
        model = model,
        idx = encoded_tensor,
        max_new_tokens = 30,
        context_size = GPT_CONFIG_124M['context_length'],
        use_cache = True
    )

    decoded_text = tokenizer.decode(out.squeeze(0).tolist())

    print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
    print("\nOutput:", out)
    print("Output length:", len(out[0]))
    print("Output text:", decoded_text)

if __name__ == "__main__":
    main()