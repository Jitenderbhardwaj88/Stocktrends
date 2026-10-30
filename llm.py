"""Minimal client for an OpenAI-compatible chat endpoint (Bionic / Ollama / LM Studio / hosted APIs)."""
import ipaddress
import json
import os
import socket
from urllib.parse import urlparse

import requests

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
LOCAL_DEFAULTS = {"base_url": "http://localhost:3000/v1", "api_key": "", "model": "gemma-4-e4b", "timeout": 180}
CLOUD_DEFAULTS = {"base_url": "", "api_key": "", "model": "", "timeout": 120}


def is_cloud():
    """True when running online (Streamlit Community Cloud).
    Override with the environment variable AI_CLOUD_MODE=1 (force online) or AI_CLOUD_MODE=0 (force local)."""
    flag = os.environ.get("AI_CLOUD_MODE")
    if flag in ("0", "1"):
        return flag == "1"
    here = os.path.abspath(os.path.dirname(__file__)).replace("\\", "/")
    return here.startswith("/mount/src") or os.environ.get("HOSTNAME", "").startswith("streamlit")


def load_config():
    """Local: defaults + saved config.json. Online: blank defaults, never read from disk."""
    if is_cloud():
        return dict(CLOUD_DEFAULTS)
    cfg = dict(LOCAL_DEFAULTS)
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg.update(json.load(f))
    return cfg


def save_config(cfg):
    """Local only. Online nothing is written, so one visitor's key can never reach another visitor."""
    if is_cloud():
        raise RuntimeError("Saving is disabled online.")
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def check_url(url):
    """Online only: allow https addresses that resolve to public hosts, so visitors can't point the
    server at localhost or internal networks."""
    if not url:
        raise ValueError("Enter a Base URL first.")
    if not is_cloud():
        return
    p = urlparse(url)
    if p.scheme != "https" or not p.hostname:
        raise ValueError("Online, the Base URL must start with https://")
    try:
        infos = socket.getaddrinfo(p.hostname, p.port or 443)
    except socket.gaierror:
        raise ValueError("Could not resolve that host name.")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if not ip.is_global:
            raise ValueError("Base URL must point to a public host.")


def list_models(cfg):
    check_url(cfg["base_url"])
    h = {"Authorization": f"Bearer {cfg['api_key']}"} if cfg.get("api_key") else {}
    r = requests.get(cfg["base_url"].rstrip("/") + "/models", headers=h, timeout=15, allow_redirects=False)
    r.raise_for_status()
    return [m["id"] for m in r.json().get("data", [])]


def ask(cfg, system, user, temperature=0.3, max_tokens=900):
    check_url(cfg["base_url"])
    h = {"Content-Type": "application/json"}
    if cfg.get("api_key"):
        h["Authorization"] = f"Bearer {cfg['api_key']}"
    body = {"model": cfg["model"], "temperature": temperature, "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    r = requests.post(cfg["base_url"].rstrip("/") + "/chat/completions", headers=h, json=body,
                      timeout=cfg["timeout"], allow_redirects=False)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


SYSTEM = (
    "You are a concise equity-market technical analyst. Use ONLY the numbers provided; never invent prices, "
    "news or fundamentals. Give: 1) market/sector read, 2) per-ticker one-liners (trend, key level, risk), "
    "3) what would change the view. Max ~250 words. This is educational analysis, not financial advice."
)
