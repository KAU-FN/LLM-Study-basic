# References

[AI Engineering Hub - Intermediate Projects](https://github.com/patchy631/ai-engineering-hub/tree/main/llamaindex-mcp)

## Requires packages

```
llama-index                        0.14.20
llama-index-core                   0.14.20
llama-index-embeddings-fastembed   0.6.0
llama-index-embeddings-openai      0.6.0
llama-index-instrumentation        0.5.0
llama-index-llms-google-genai      0.9.1
llama-index-llms-ollama            0.10.1
llama-index-llms-openai            0.7.5
llama-index-readers-file           0.6.0
llama-index-tools-mcp              0.4.8
llama-index-vector-stores-qdrant   0.10.0
llama-index-workflows              2.17.1
mcp                                1.27.1
```

- open source model인 llama3.2(Ollama) 사용
  - 사용을 위해 Ollama 설치 필요.
  - [설치 참고 link](https://goddaehee.tistory.com/381)

- 설치가 되었다면 cmd에서 `ollama list`를 통해 현재 PC에 있는 LLM 종류를 볼 수 있음.

```
NAME               ID              SIZE
llama3.2:latest    ************    2.0 GB
```

## Run example

- 별도의(2개의) cmd 창을 열고, 한쪽에선 `server.py` 실행.

```
uv run server.py
```

- server 실행 후, `ollama_client.py` 실행하여 agent와 대화.

```
uv run ollama_client.py
```

- 실제 동작은 아래와 같음.

![example](./agent_response_example.png)
