import os
import requests
import json

# 从环境变量读取，或者直接写死
API_KEY = os.getenv("ANTHROPIC_AUTH_TOKEN", "sk-pBh8KJJ9drUHaDIERwzkebmqqKOQym3P0K2nq6NOfoalIQYa")
BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "https://anyrouter.top")

url = f"{BASE_URL}/v1/messages"

headers = {
    "Content-Type": "application/json",
    "x-api-key": API_KEY,
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36 Edg/129.0.0.0',
}

data = {
    "model": "claude-3.5-sonnet-20241022",   # 模型名
    "max_tokens": 200,                       # 最大生成 token 数
    "system": "You are a coding assistant that completes code snippets.",
    "messages": [
        {"role": "user", "content": "写一个Python函数，计算斐波那契数列前10项"}
    ]
}

try:
    response = requests.post(url, headers=headers, data=json.dumps(data), timeout=30)
    response.raise_for_status()
    result = response.json()
    print(json.dumps(result, indent=2, ensure_ascii=False))
except requests.exceptions.RequestException as e:
    print(f"请求失败: {e}")
