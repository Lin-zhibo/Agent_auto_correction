import os
import requests

# 与 config.py:32-34 一致
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_API_EMBEDDING_KEY = os.getenv(
    "OPENAI_API_EMBEDDING_KEY",
    "sk-ADw7Ylh7r6JTQJfBAb4e16F9Ed01488eA16aA0Fa1eB26d3e",
)
OPENAI_BASE_EMBEDDING_URL = os.getenv(
    "OPENAI_BASE_EMBEDDING_URL",
    "https://api.gpt.ge/v1/",
)

# 与 judge_eval.py:28-30 一致
KEY = os.getenv("JUDGE_API_KEY", "sk-ADw7Ylh7r6JTQJfBAb4e16F9Ed01488eA16aA0Fa1eB26d3e")
BASE_URL = os.getenv("JUDGE_BASE_URL", "https://api.gpt.ge/v1/")
# JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-5-chat-latest")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4.1-mini")


def _join_url(base_url: str, endpoint: str) -> str:
    return f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"


def test_embedding_api() -> bool:
    url = _join_url(OPENAI_BASE_EMBEDDING_URL, "/embeddings")
    headers = {
        "Authorization": f"Bearer {OPENAI_API_EMBEDDING_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OPENAI_EMBEDDING_MODEL,
        "input": "这是一条用于测试 embedding API 的文本。",
    }

    print("=== 测试 Embedding API ===")
    print(f"URL: {url}")
    print(f"MODEL: {OPENAI_EMBEDDING_MODEL}")
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        emb = (data.get("data") or [{}])[0].get("embedding", [])
        print(f"[PASS] Embedding API 可用，向量维度: {len(emb)}")
        return True
    except Exception as e:
        print(f"[FAIL] Embedding API 失败: {e}")
        return False


def test_judge_api() -> bool:
    url = _join_url(BASE_URL, "/chat/completions")
    headers = {
        "Authorization": f"Bearer {KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": JUDGE_MODEL,
        "messages": [{"role": "user", "content": "请回复：judge ok"}],
        "temperature": 0,
    }

    print("\n=== 测试 Judge API ===")
    print(f"URL: {url}")
    print(f"MODEL: {JUDGE_MODEL}")
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        content = (
            ((data.get("choices") or [{}])[0].get("message") or {}).get("content", "")
        )
        print(f"[PASS] Judge API 可用，返回内容: {content[:120]}")
        return True
    except Exception as e:
        print(f"[FAIL] Judge API 失败: {e}")
        return False


if __name__ == "__main__":
    embedding_ok = test_embedding_api()
    judge_ok = test_judge_api()
    print("\n=== 测试汇总 ===")
    print(f"Embedding API: {'PASS' if embedding_ok else 'FAIL'}")
    print(f"Judge API: {'PASS' if judge_ok else 'FAIL'}")