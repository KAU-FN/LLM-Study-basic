# References

https://github.com/patchy631/ai-engineering-hub/tree/main/qwen-2.5VL-ocr


Window 설치 및 실행 방법

```
# 가상환경 생성
python -m venv qwen-ocr

### 가상환경 실행 (매 실행시마다)
.\qwen-ocr\Scripts\activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 허깅페이스 트랜스포머 최신 소스 설치 및 가속 라이브러리
pip install --upgrade --force-reinstall --no-cache-dir git+https://github.com/huggingface/transformers.git

# Qwen 비전 모델용 유틸리티 및 이미지 처리 라이브러리
pip install qwen-vl-utils pillow

# Web 기반 UI
pip install streamlit openai

# 실행
streamlit run ocr.py
```


Python 스크립트 수정부분
1. FlashAttention 패키지 미사용 (에러 회피)
2. 이미지 크기 및 해상도 조건 완화