# LLM-Study-basic

LLM study repository for basic code implementation

---

Notion link는 카카오 단체방 상단 게시물 확인

- 개별로 branch 생성 $\rightarrow$ 맡은 파트는 해당 파트에서 진행
- main branch는 세미나를 위한 완성된 정리본 만을 commit
  - (가능하면) 해당 파트의 발표자가 하단의 `.md` 파일 내용에 작성한 code/repository의 링크 삽입

---

## Table of Contents

### 1. Low-Level Implementation

> LLM의 핵심이 되는 attention mechanism, Transformer block을 직접 구현하며 이해하는 것을 목표로 함.

- [Working with text data](https://github.com/KAU-FN/LLM-Study-basic/blob/db3256b390ae02f6e78f808fc5577fcc15814e70/01.Low-level%20implementation/02_working_with_text_data.ipynb)
- [Coding attention mechanism](https://github.com/KAU-FN/LLM-Study-basic/blob/main/01.Low-level%20implementation/03_multihead_attention.py)
- [Implementing a GPT model from scratch to generate text](https://github.com/KAU-FN/LLM-Study-basic/blob/db3256b390ae02f6e78f808fc5577fcc15814e70/01.Low-level%20implementation/04_implementing_GPT_from_scratch.ipynb)
  - [KV-cache](https://github.com/KAU-FN/LLM-Study-basic/blob/db3256b390ae02f6e78f808fc5577fcc15814e70/01.Low-level%20implementation/04_1_kv-cache.ipynb)
  - [GQA(Grouped Query Attention)](https://github.com/KAU-FN/LLM-Study-basic/blob/db3256b390ae02f6e78f808fc5577fcc15814e70/01.Low-level%20implementation/04_2_GQA.ipynb)
  - [SWA(Sliding Window Attention)](https://github.com/KAU-FN/LLM-Study-basic/blob/db3256b390ae02f6e78f808fc5577fcc15814e70/01.Low-level%20implementation/04_3_SWA.ipynb)
  - [MoE(Mixture of Experts)](https://github.com/KAU-FN/LLM-Study-basic/blob/db3256b390ae02f6e78f808fc5577fcc15814e70/01.Low-level%20implementation/04_4_MoE.ipynb)

### 2. LLM techniques

> 이전 study & seminar 에서 진행했던 LLM 관련 기법들을 code 작성을 통해 이해하는 것을 목표로 함.

- [Pretraining on unlabeled data](https://github.com/KAU-FN/LLM-Study-basic/blob/main/02.LLM%20techniques/05_pretraining_on_unlabeled_data.ipynb)
  - [Memory-efficient weight loading](https://github.com/KAU-FN/LLM-Study-basic/blob/main/02.LLM%20techniques/05_1_memory-efficient-state-dict.ipynb)
  - [Converting GPT to Llama 2](https://github.com/KAU-FN/LLM-Study-basic/blob/main/02.LLM%20techniques/05_2_converting-gpt-to-llama2.ipynb)
  - [Optimizing Hyperparameters for Pretraining](https://github.com/KAU-FN/LLM-Study-basic/blob/main/02.LLM%20techniques/05_3_hparam_search.ipynb)
  - [Learning Rate Schedulers](https://github.com/KAU-FN/LLM-Study-basic/blob/main/02.LLM%20techniques/05_4_learning_rate_schedulers.ipynb)
- [Finetuning for classification](https://github.com/KAU-FN/LLM-Study-basic/blob/main/02.LLM%20techniques/06_finetuning_for_classification.ipynb)
  - [IMDB classification with GPT](https://github.com/KAU-FN/LLM-Study-basic/blob/main/02.LLM%20techniques/06_1_IMDB_classification.py)
- [Finetuning to follow instructions](https://github.com/KAU-FN/LLM-Study-basic/blob/main/02.LLM%20techniques/07_finetuning_to_follow_instructions.ipynb)
  - [Creating preference dataset](https://github.com/KAU-FN/LLM-Study-basic/blob/main/02.LLM%20techniques/07_1_create_preference_data_ollama.ipynb)
  - [Preference Tuning with DPO](https://github.com/KAU-FN/LLM-Study-basic/blob/main/02.LLM%20techniques/07_3_dpo_from_scratch.ipynb)
  - [Creating Passive Voice Entries](https://github.com/KAU-FN/LLM-Study-basic/blob/main/02.LLM%20techniques/07_2_create-passive-voice-entries.ipynb)
  - [Generating Datasets for Instruction Tuning](https://github.com/KAU-FN/LLM-Study-basic/blob/main/02.LLM%20techniques/07_4_llama3_ollama.ipynb)
  - [Instruction dataset refinement based on reflection-tuning](https://github.com/KAU-FN/LLM-Study-basic/blob/main/02.LLM%20techniques/07_5_reflection-gpt4.ipynb)
- SFT + Llama2 fine-tuning (HW issue로 인해 link 대체)
  - [SFT](https://github.com/adithya-s-k/AI-Engineering.academy/blob/main/docs/LLM/HandsOnWithFinetuning/SFT/SFT_finetuning_notebook.ipynb)
  - [Llama2 fine-tuning](https://github.com/adithya-s-k/AI-Engineering.academy/blob/main/docs/LLM/LLama2/Llama2_finetuning_notebook.ipynb)
- SFT + QLoRa + Gemma (HW issue로 인해 link 대체)
  - [Instruction fine-tuning Gemma with qLora & SFT](https://github.com/adithya-s-k/AI-Engineering.academy/blob/main/docs/LLM/LLama2/Llama2_finetuning_notebook.ipynb)
- RAG techniques
  - [RAG from scratch](https://github.com/KAU-FN/LLM-Study-basic/blob/main/02.LLM%20techniques/09_1_RAG_from_scratch.ipynb)
  - [Basic RAG](https://github.com/KAU-FN/LLM-Study-basic/blob/main/02.LLM%20techniques/09_2_Basic_RAG.ipynb)
    - [Basic RAG(standalone script)](https://github.com/KAU-FN/LLM-Study-basic/blob/main/02.LLM%20techniques/09_2_Basic_RAG.ipynb)
  - [Evaluating RAG Systems with DeepEval](https://github.com/KAU-FN/LLM-Study-basic/blob/main/02.LLM%20techniques/09_3_DeepEval.ipynb)
  - [Hybrid RAG with Qdrant Hybrid Search](https://github.com/KAU-FN/LLM-Study-basic/blob/main/02.LLM%20techniques/09_4_HybridRAG.ipynb)
  - [GraphRAG with LlamaIndex](https://github.com/KAU-FN/LLM-Study-basic/blob/main/02.LLM%20techniques/09_5_GraphRAG.ipynb)

### 3. MCP

> paper review를 통해 접했던 serving의 핵심 중 하나인 MCP server architecture를 code 작성을 통해 직접 이해하는 것을 목표로 함.

- Toy MCP server practice
- HuggingFace MCP Course
  - End-to-End MCP Application
  - Advanced MCP Development: Custom Workflow Services

### 4. Agent Building (Orchestration)

> 앞선 skill들과 open-weight model 활용, 실제 간단한 AI Agent를 build 해보는 것을 목표로 함.

- HuggingFace AI Agents Course
- AI Engineering Hub tutorial
