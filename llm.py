"""Minimal client for an OpenAI-compatible chat endpoint (Bionic / Ollama / LM Studio)."""
import json
import os
import requests

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
DEFAULTS = {"base_url": "http://localhost:3000/v1", "api_key": "", "model": "gemma-4-e4b", "timeout": 180}


def load_config():
    cfg = dict(DEFAULTS)
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg.update(json.load(f))
    return cfg


def save_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def list_models(cfg):
    h = {"Authorization": f"Bearer {cfg['api_key']}"} if cfg.get("api_key") else {}
    r = requests.get(cfg["base_url"].rstrip("/") + "/models", headers=h, timeout=15)
    r.raise_for_status()
    return [m["id"] for m in r.json().get("data", [])]


def ask(cfg, system, user, temperature=0.3, max_tokens=900):
    h = {"Content-Type": "application/json"}
    if cfg.get("api_key"):
        h["Authorization"] = f"Bearer {cfg['api_key']}"
    body = {"model": cfg["model"], "temperature": temperature, "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    r = requests.post(cfg["base_url"].rstrip("/") + "/chat/completions", headers=h, json=body, timeout=cfg["timeout"])
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


SYSTEM = (
    "You are a concise equity-market technical analyst. Use ONLY the numbers provided; never invent prices, "
    "news or fundamentals. Give: 1) market/sector read, 2) per-ticker one-liners (trend, key level, risk), "
    "3) what would change the view. Max ~250 words. This is educational analysis, not financial advice."
)
