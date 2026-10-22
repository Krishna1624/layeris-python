"""
Interactive AI Chat Engine with Speculative Acceleration
========================================================
Supports interactive conversation, live web search augmenting factual queries,
and real-time tokens-per-second (TPS) telemetry streaming.
"""

from __future__ import annotations

import argparse
import html
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from threading import Thread
from typing import Optional

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
except ImportError:
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    TextIteratorStreamer = None

from layeris.inference.guard import configure_safe_execution_environment


def search_live_web(query: str, timeout: int = 6) -> str:
    """Fetches real-time factual information from duckduckgo html without API keys."""
    q = re.sub(r'\bcm\b', 'Chief Minister', query, flags=re.I)
    q = re.sub(r'\bpm\b', 'Prime Minister', q, flags=re.I)
    q = re.sub(r'\bceo\b', 'Chief Executive Officer', q, flags=re.I)

    url = "https://html.duckduckgo.com/html/"
    data = urllib.parse.urlencode({"q": q}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw_html = resp.read().decode("utf-8", errors="ignore")
            snippets = re.findall(r'<a class="result__snippet[^"]*"[^>]*>(.*?)</a>', raw_html, re.DOTALL)
            clean = [html.unescape(re.sub(r'<[^>]+>', '', s).strip()) for s in snippets[:4]]
            return " ".join(clean)
    except Exception:
        return ""


def should_search_web(text: str) -> bool:
    """Heuristic to decide if live web context should augment prompt."""
    lower = text.lower().strip()
    greetings = ["hi", "hello", "hey", "good morning", "good evening", "how are you"]
    if lower in greetings:
        return False
    if lower.startswith("give python code") or lower.startswith("write code") or lower.startswith("code for"):
        return False
    return True


class InteractiveChatEngine:
    def __init__(self, model_id: str = "Qwen/Qwen2.5-0.5B-Instruct", device: str = "auto", spec_k: int = 3):
        if torch is None or AutoModelForCausalLM is None:
            raise RuntimeError("PyTorch and transformers must be installed to use InteractiveChatEngine.")

        self.model_id = model_id
        self.spec_k = spec_k
        self.profile = configure_safe_execution_environment()

        if device == "auto":
            self.device = self.profile.recommended_device
        else:
            self.device = device

        print(f"[*] Initializing model {model_id} on device: {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )
        if self.device == "cpu":
            self.model = self.model.to("cpu")

        print("[*] Engine ready.")

    def chat_loop(self):
        """Runs the interactive console chat loop."""
        history = [
            {"role": "system", "content": "You are a helpful, brilliant, fast AI assistant powered by Layeris."}
        ]
        print("\n=======================================================")
        print("  Layeris Interactive Chat (Type 'quit' or 'exit' to stop)")
        print(f"  Model: {self.model_id} | Device: {self.device} | Speculative K: {self.spec_k}")
        print("=======================================================\n")

        while True:
            try:
                user_msg = input("\nYou > ").strip()
            except (KeyboardInterrupt, EOFError):
                break

            if not user_msg:
                continue
            if user_msg.lower() in ["quit", "exit"]:
                break

            web_ctx = ""
            if should_search_web(user_msg):
                print("[*] Checking live web context...")
                web_ctx = search_live_web(user_msg)

            augmented_msg = user_msg
            if web_ctx:
                augmented_msg = f"{user_msg}\n\n[Real-time Live Web Context: {web_ctx[:1000]}]"

            history.append({"role": "user", "content": augmented_msg})

            prompt_text = self.tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)

            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            gen_kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            if self.spec_k > 0:
                gen_kwargs["prompt_lookup_num_tokens"] = self.spec_k

            thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
            thread.start()

            print("\nAssistant > ", end="", flush=True)
            token_count = 0
            t0 = time.perf_counter()
            full_response = []

            for text_chunk in streamer:
                print(text_chunk, end="", flush=True)
                full_response.append(text_chunk)
                token_count += 1

            thread.join()
            t1 = time.perf_counter()
            duration = max(0.001, t1 - t0)
            tps = token_count / duration

            print(f"\n\n[Stats: {token_count} tokens in {duration:.2f}s | {tps:.1f} tokens/sec]")
            history.append({"role": "assistant", "content": "".join(full_response)})


def main():
    parser = argparse.ArgumentParser(description="Layeris Interactive AI Chat Engine")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="HuggingFace model ID")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Inference device")
    parser.add_argument("--spec-k", type=int, default=3, help="Speculative candidate depth (0 to disable)")
    args = parser.parse_args()

    engine = InteractiveChatEngine(model_id=args.model, device=args.device, spec_k=args.spec_k)
    engine.chat_loop()


if __name__ == "__main__":
    main()
