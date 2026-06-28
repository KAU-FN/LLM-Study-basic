import json
import gradio as gr

# 감정분석을 위해 TextBlob 라이브러리 사용
# 주의) 영문 입력만 받음
from textblob import TextBlob


def sentiment_analysis(text: str) -> str:
    #인자 및 출력에 대한 타입힌트를 줘서 스키마 정의하는데 힌트 주기
    """
    Analyze the sentiment of the given text.

    Args:
        text (str): The text to analyze

    Returns:
        str: A JSON string containing polarity, subjectivity, and assessment
    """
    
    blob = TextBlob(text)
    sentiment = blob.sentiment
    
    result = {
        "polarity": round(sentiment.polarity, 2),  # -1 부정 to 1 긍정
        "subjectivity": round(sentiment.subjectivity, 2),  # 0 객관적 to 1 주관적
        "assessment": "positive" if sentiment.polarity > 0 else "negative" if sentiment.polarity < 0 else "neutral"
    }

    return json.dumps(result)

# Gradio 를 사용하기 위한 인터페이스 생성
demo = gr.Interface(
    fn=sentiment_analysis,
    inputs=gr.Textbox(placeholder="Enter text to analyze..."),
    outputs=gr.Textbox(), 
    title="Text Sentiment Analysis",
    description="Analyze the sentiment of text using TextBlob"
)

# 인터페이스 및 MCP 서버 실행
# MCP 서버의 엔드포인트는  http://localhost:7860/gradio_api/mcp/sse
# 접근가능 url http://localhost:7860
# tool 스키마 확인 http://localhost:7860/gradio_api/mcp/schema
# 환경변수를 사용하여 실행도 가능
if __name__ == "__main__":
    demo.launch(mcp_server=True)
