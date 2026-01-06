import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 02_working_with_text_data.py에서 작성했던 GPTDatasetV1, create_dataloader 을 사용

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

def create_dataloader_v1(text, batch_size=4, max_length=256, stride=128, shuffle=True):
    # tokenizer 초기화
    tokenizer = tiktoken.get_encoding("gpt2")

    # dataset 생성
    dataset = GPTDatasetV1(text, tokenizer, max_length, stride)

    # dataloader 생성
    dataloader = DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        shuffle = shuffle,
    )

    return dataloader

# text file read
with open("small-text-sample.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# tokenizer 및 encoded text 생성
tokenizer = tiktoken.get_encoding("gpt2")
encoded_text = tokenizer.encode(raw_text)

vocab_size = 50257  # GPT-2의 vocabulary size
output_dim = 256
max_len = 1024
context_length = max_len

# embedding layer 생성
token_embedding_layer = nn.Embedding(vocab_size, output_dim)
pos_embedding_layer = nn.Embedding(context_length, output_dim)

# dataloader 생성
max_length = 4
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=max_length)

# # 생성되는 batch shape 확인 -> (8, 4, 256)
# for batch in dataloader:
#     x, y = batch

#     token_embeddings = token_embedding_layer(x)  # [batch_size, seq_length, output_dim]
#     pos_embeddings = pos_embedding_layer(torch.arange(max_length))

#     input_embeddings = token_embeddings + pos_embeddings  # [batch_size, seq_length, output_dim]

#     print(input_embeddings.shape)
#     break


# Simple Multi-head Attention
class CasualSelfAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)    # Query weight
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)      # Key weight
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)    # Value weight
        self.dropout = nn.Dropout(dropout)

        # register buffer: https://analytics4everything.tistory.com/313
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        B, n_tokens, d_in = x.shape   # batch 단위로 처리하므로 batch dimension인 B가 추가

        # input -> Q, K, V projection
        Q = self.W_query(x)  # [B, n_tokens, d_out]
        K = self.W_key(x)    # [B, n_tokens, d_out]
        V = self.W_value(x)  # [B, n_tokens, d_out]

        attention_scores = Q @ K.transpose(1, 2)  # Q * K^T -> [B, n_tokens, n_tokens]
        attention_scores.masked_fill_(
            self.mask.bool()[:n_tokens, :n_tokens], -torch.inf
        )                                         # future token을 보지 못하도록 masking 처리
        attention_weights = torch.softmax(attention_scores / K.shape[-1]**0.5, dim=-1)  # scale dot-product attention
        attention_weights = self.dropout(attention_weights) # dropout

        context_vector = attention_weights @ V # [B, n_tokens, d_out]

        return context_vector

class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        
        self.heads = nn.ModuleList(
            [CasualSelfAttention(d_in, d_out, context_length, dropout, qkv_bias)
             for _ in range(num_heads)]
        )
        self.out_projection = nn.Linear(d_out * num_heads, d_out * num_heads)
    
    def forward(self, x):
        context_vector = torch.cat([head(x) for head in self.heads], dim=-1)  # 각 head의 출력을 concat
        output = self.out_projection(context_vector)  # output projection

        return output

# # 실제 동작 확인
# torch.manual_seed(62)

# for batch in dataloader:
#     x, y = batch

#     token_embeddings = token_embedding_layer(x)  # [batch_size, seq_length, output_dim]
#     pos_embeddings = pos_embedding_layer(torch.arange(max_length))

#     input_embeddings = token_embeddings + pos_embeddings  # [batch_size, seq_length, output_dim]
#     break

# context_length = max_length
# d_in = output_dim

# num_heads = 2
# d_out = d_in // num_heads

# MHA = MultiHeadAttentionWrapper(d_in, d_out, context_length, dropout=0.0, num_heads=num_heads)

# batch = input_embeddings    # 앞서 생성한 batch 사용 [batch_size, seq_length, output_dim]
# context_vector = MHA(batch) # [batch_size, seq_length, output_dim]

# # context_vector shape: torch.Size([8, 4, 256])
# print("context_vector shape:", context_vector.shape)    # [8, 4, 256]