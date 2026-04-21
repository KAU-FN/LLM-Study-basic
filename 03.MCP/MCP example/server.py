# # server.py
# from mcp.server.fastmcp import FastMCP

# # MCP 서버 생성
# mcp = FastMCP("Demo")


# # 추가 도구 추가
# @mcp.tool()
# def add(a: int, b: int) -> int:
#     """Add two numbers"""
#     return a + b


# # 동적 인사말 리소스 추가
# @mcp.resource("greeting://{name}")
# def get_greeting(name: str) -> str:
#     """Get a personalized greeting"""
#     return f"Hello, {name}!"

# # 메인 실행 블록 - 서버를 실행하려면 필요합니다
# if __name__ == "__main__":
#     mcp.run()

import json
import urllib.request
from urllib.parse import quote
from mcp.server.fastmcp import FastMCP

# MCP 서버 생성
mcp = FastMCP("Demo")


# 1. 기존 계산기 도구
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


# 2. ✨ 날씨 조회 도구
@mcp.tool()
def get_weather(city: str) -> str:
    """Get the current weather and temperature for a specific city.
    Args:
        city: The name of the city (e.g., Seoul, Tokyo, New York)
    """
    try:
        # 1단계: 도시 이름으로 위도/경도 검색 (Open-Meteo Geocoding API)
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={quote(city)}&count=1"
        with urllib.request.urlopen(geo_url) as response:
            geo_data = json.loads(response.read())
            
        if not geo_data.get("results"):
            return f"'{city}' 도시를 찾을 수 없습니다."
            
        lat = geo_data["results"][0]["latitude"]
        lon = geo_data["results"][0]["longitude"]
        
        # 2단계: 위도/경도로 날씨 조회 (Open-Meteo Weather API)
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        with urllib.request.urlopen(weather_url) as response:
            weather_data = json.loads(response.read())
            
        current = weather_data["current_weather"]
        return f"{city}의 현재 날씨: 온도 {current['temperature']}°C, 풍속 {current['windspeed']}km/h"
    except Exception as e:
        return f"날씨 조회 실패: {str(e)}"


# 3. ✨ 환율 조회 도구
@mcp.tool()
def get_exchange_rate(base: str, target: str) -> str:
    """Get the real-time exchange rate between two currencies.
    Args:
        base: Base currency code (e.g., USD, EUR, KRW)
        target: Target currency code (e.g., KRW, JPY, USD)
    """
    try:
        base = base.upper()
        target = target.upper()
        
        url = f"https://api.frankfurter.app/latest?from={base}&to={target}"
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read())
            
        rate = data["rates"].get(target)
        return f"1 {base} = {rate} {target} 입니다. (기준일: {data['date']})"
    except Exception as e:
        return f"환율 조회 실패: {str(e)}"


# 4. ✨ 무작위 농담 도구
@mcp.tool()
def get_joke() -> str:
    """Get a random programming or general joke."""
    try:
        url = "https://official-joke-api.appspot.com/random_joke"
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read())
            
        return f"{data['setup']} ... {data['punchline']}"
    except Exception as e:
        return f"농담 가져오기 실패: {str(e)}"


# 기존 동적 인사말 리소스 유지
@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    return f"Hello, {name}!"


# 메인 실행 블록
if __name__ == "__main__":
    mcp.run()