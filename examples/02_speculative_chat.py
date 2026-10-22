"""
Example 02: Interactive Speculative Chat
========================================
Runs real-time interactive chat with speculative prompt-lookup and live web augmentation.
"""

import sys
from pathlib import Path

# Add src to sys.path if running without pip install -e .
src_path = Path(__file__).resolve().parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from layeris.inference.chat import InteractiveChatEngine


def main():
    print("Launching Layeris Interactive Chat...")
    engine = InteractiveChatEngine(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        device="auto",
        spec_k=3,
    )
    engine.chat_loop()


if __name__ == "__main__":
    main()
