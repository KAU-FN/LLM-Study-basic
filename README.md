# LLM-Study-basic
LLM study repository for basic code implementation

-------------------

Notion link는 카카오 단체방 상단 게시물 확인

- 개별로 branch 생성 $\rightarrow$ 맡은 파트는 해당 파트에서 진행 
- main branch는 세미나를 위한 완성된 정리본 만을 commit
  - (가능하면) 해당 파트의 발표자가 하단의 `.md` 파일 내용에 작성한 code/repository의 링크 삽입

-------------------

## Table of Contents

### 1. Low-Level Implementation

> LLM의 핵심이 되는 attention mechanism, Transformer block을 직접 구현하며 이해하는 것을 목표로 함.

- Working with text data
- Coding attention mechanism
- Implementing a GPT model from scratch to generate text
  - KV-cache
  - GQA(Grouped Query Attention)
  - SWA(Sliding Window Attention)
  - MoE(Mixture of Experts)


### 2. LLM techniques

> 이전 study & seminar 에서 진행했던 LLM 관련 기법들을 code 작성을 통해 이해하는 것을 목표로 함.

- Pretraining on unlabeled data
- Finetuning for classification
- Finetuning to follow instructions
- SFT + Llama2 fine-tuning
- SFT + QLoRa + Gemma
- RAG techniques


### 3. MCP

> paper review를 통해 접했던 serving의 핵심 중 하나인 MCP server architecture를 code 작성을 통해 직접 이해하는 것을 목표로 함.

- in progress


### 4. Agent Building (Orchestration)

> 앞선 skill들과 open-weight model 활용, 실제 간단한 AI Agent를 build 해보는 것을 목표로 함.

- in progress