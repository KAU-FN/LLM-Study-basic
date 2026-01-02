import os
import requests
import re

# 파일을 직접 다운로드 하지 않은 경우 다운로드.
if not os.path.exists(os.getcwd() + 'the-verdict.txt'):
    url = (
        "https://raw.githubusercontent.com/rasbt/"
        "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
        "the-verdict.txt"
    )

    file_path = 'the-verdict.txt'

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    with open(file_path, 'wb') as f:
        f.write(response.content)

with open('the-verdict.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# print('total number of characters:', len(text)) # 20479

# 읽어들인 text 간단한 전처리 (with regex) => tokenize
preprocessed_text = re.split(r'([,.:;?_!"()\']|--|\s)', text)
preprocessed_text = [item.strip() for item in preprocessed_text if item.strip()]
# print(preprocessed_text[:20])
# print(len(preprocessed_text))   # 4690

# token을 ID로 변환 -> embedding layer에 추가하기 위함.
all_words = sorted(set(preprocessed_text))
vocab_size = len(all_words)
# print('vocab size:', vocab_size)    # 1130

vocab = {token:integer for integer, token in enumerate(all_words)}
# print(vocab)    # {..., 'Made': 61, 'Miss': 62, ...} 와 같은 dict 형태의 단어사전 완성


# 이 모든걸 Tokenizer 클래스로 묶기 -> 재사용성을 위함
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}
    
    def encode(self, text):
        """
        text를 token ID로 변환
        """
        tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        tokens = [item.strip() for item in tokens if item.strip()]
        token_ids = [self.str_to_int[token] for token in tokens]
        return token_ids

    def decode(self, token_ids):
        """
        token ID를 다시 text로 변환
        """
        text = " ".join([self.int_to_str[i] for i in token_ids])

        # 특정 토큰 앞의 공백 제거
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

# tokenizer의 사용 예시
tokenizer = SimpleTokenizerV1(vocab)

text = """"It's the last he painted, you know," 
           Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)

# ## 다음과 같이 raw text가 id로 mapping(encode) 됨.
# ## [1, 56, 2, 850, 988, 602, 533, 746, 5, 1126, 596, 5, 1, 67, 7, 38, 851, 1108, 754, 793, 7]
# print(ids)

# ## decode를 하면 다시 원본 text가 나옴.
# ## " It' s the last he painted, you know," Mrs. Gisburn said with pardonable pride.
# print(tokenizer.decode(ids))

# Unknown words 처리 기능 추가 -> special context token 추가
# [BOS](beginning of sentence), [EOS](end of sentence), [UNK](unknown token), [PAD](padding token) 등...
# vocab에 위와 같은 special token 추가
all_tokens = sorted(list(set(preprocessed_text)))
all_tokens.extend(['<|bos|>', '<|eos|>', '<|unk|>', '<|pad|>', '<|endoftext|>'])
vocab = {token:integer for integer,token in enumerate(all_tokens)}

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}
    
    def encode(self, text):
        tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        tokens = [item.strip() for item in tokens if item.strip()]
        tokens = [
            item if item in self.str_to_int 
            else '<|unk|>' for item in tokens
        ]
        token_ids = [self.str_to_int[token] for token in tokens]
        return token_ids

    def decode(self, token_ids):
        text = " ".join([self.int_to_str[i] for i in token_ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

# Unknown words 처리가 추가된 tokenizer 사용 예시
tokenizer = SimpleTokenizerV2(vocab)

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."

text = " <|endoftext|> ".join((text1, text2))

# ## 사전에 없던 단어(Hello)는 <|unk|> 토큰으로 대체되는 것을 확인.
# ## Hello, do you like tea? <|endoftext|> In the sunlit terraces of the palace.
# print(text)
# ## <|unk|>, do you like tea? <|endoftext|> In the sunlit terraces of the <|unk|>.
# print(tokenizer.decode(tokenizer.encode(text)))


# GPT-2가 tokenizer로 사용하는 BytePair Encoding (BPE) 기법 간단 구현
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     "of someunknownPlace."
)

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

# # [15496, 11, 466, 345, 588, 8887, 30, 220, 50256, 554, 262, 4252, 18250, 8812, 2114, 1659, 617, 34680, 27271, 13]
# print(integers)

strings = tokenizer.decode(integers)
# # Hello, do you like tea? <|endoftext|> In the sunlit terracesof someunknownPlace.
# print(strings)


# Sliding window를 통한 data sampling
# LLM은 한번에 하나의 단어를 생성, 따라서 training data를 시퀀스로 나눠서 target을 예측하도록 데이터를 준비해야 함.

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text)
# print(len(enc_text))  # 5145

## sliding window 예시
## [290] ----> 4920
##  and ---->  established
## [290, 4920] ----> 2241
##  and established ---->  himself
## [290, 4920, 2241] ----> 287
##  and established himself ---->  in
## [290, 4920, 2241, 287] ----> 257
##  and established himself in ---->  a
context_size = 4
enc_sample = enc_text[50:]

for i in range(1, context_size+1):
    context = enc_sample[:i]
    to_predict = enc_sample[i]

    # print(context, "---->", to_predict)
    # print(tokenizer.decode(context), "---->", tokenizer.decode([to_predict]))

# data loder를 이용해 input dataset을 iterate하며, input 과 target이 하나씩 shift 되는 형태로 batch를 생성
import torch
from torch.utils.data import Dataset, DataLoader

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

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# ## context size가 4이고 batch size가 1인 dataloader 예시
# dataloader = create_dataloader_v1(
#     raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
# )

# data_iter = iter(dataloader)
# first_batch = next(data_iter)
# print(first_batch)  # [tensor([[  40,  367, 2885, 1464]]), tensor([[ 367, 2885, 1464, 1807]])]

## batch size를 늘린 예시
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=4, stride=4, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)

# # Inputs:
# #  tensor([[   40,   367,  2885,  1464],
# #         [ 1807,  3619,   402,   271],
# #         [10899,  2138,   257,  7026],
# #         [15632,   438,  2016,   257],
# #         [  922,  5891,  1576,   438],
# #         [  568,   340,   373,   645],
# #         [ 1049,  5975,   284,   502],
# #         [  284,  3285,   326,    11]])

# # Targets:
# #  tensor([[  367,  2885,  1464,  1807],
# #         [ 3619,   402,   271, 10899],
# #         [ 2138,   257,  7026, 15632],
# #         [  438,  2016,   257,   922],
# #         [ 5891,  1576,   438,   568],
# #         [  340,   373,   645,  1049],
# #         [ 5975,   284,   502,   284],
# #         [ 3285,   326,    11,   287]])
# print("Inputs:\n", inputs)
# print("\nTargets:\n", targets)


# token embedding 생성
# 현재 raw text -> token ID(tensor)로의 변환은 완성됨.
# 하지만, 모델(LLM)에 입력하기 위해서는 token ID를 dense vector로 변환하는 embedding layer가 필요.
# 이러한 embedding layer는 LLM 자체의 일부이고, 모델 학습 과정에서 함께 업데이트(학습)됨.

vocab_size = 50257  # GPT-2의 vocab size
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

# # 앞서 생성한 input은 [8, 4] shape의 tensor이고,
# # 이를 embedding layer에 통과시키면 [8, 4, 256] shape의 tensor가 생성됨.
# # Token IDs:
# #  tensor([[   40,   367,  2885,  1464],
# #         [ 1807,  3619,   402,   271],
# #         [10899,  2138,   257,  7026],
# #         [15632,   438,  2016,   257],
# #         [  922,  5891,  1576,   438],
# #         [  568,   340,   373,   645],
# #         [ 1049,  5975,   284,   502],
# #         [  284,  3285,   326,    11]])

# # Inputs shape:
# #  torch.Size([8, 4])

# # Token Embeddings shape:
# #  torch.Size([8, 4, 256])
# print("Token IDs:\n", inputs)
# print("\nInputs shape:\n", inputs.shape)

# token_embeddings = token_embedding_layer(inputs)
# print("\nToken Embeddings shape:\n", token_embeddings.shape)