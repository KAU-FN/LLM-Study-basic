"""
02, 03에서 작성한 module을 간편히 import 하기 위한 파일.
"""
import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 02_working_with_text_data.py에서 작성했던 GPTDatasetV1, create_dataloader 사용
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

def create_dataloader_v1(text, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
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

    def forward(self, x):
        b, num_tokens, d_in = x.shape   # batch 단위로 처리하므로 batch dimension인 B가 추가

        # input projection (Q, K, V 생성)
            # [B, n_tokens, d_in] -> [B, n_tokens, d_out]
        Q, K, V = self.W_query(x), self.W_key(x), self.W_value(x)

        # split 함수 사용(head 단위로 분할)
            # [B, n_tokens, d_out] -> [B, num_heads, n_tokens, d_head]
        Q, K, V = self.split(Q), self.split(K), self.split(V)  
        
        # self attention 연산
            # Q * K^T -> [B, num_heads, n_tokens, n_tokens]
        attn_scores = Q @ K.transpose(-2, -1)

        # masking 처리 -> future token을 보지 못하도록.
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
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