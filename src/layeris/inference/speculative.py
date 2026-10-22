"""
Speculative Decoding Engine for Layeris Python
==============================================
Provides:
1. PromptLookupDraftGenerator: N-gram prompt-lookup drafting (zero parameter overhead).
2. TwoModelSpeculativeEngine: Draft model assisted speculative decoding with verification.
3. SpeculativeStats: Detailed instrumentation of acceptance rate, tokens/forward, and TPS.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class SpeculativeStats:
    total_tokens_generated: int = 0
    total_forward_passes: int = 0
    total_draft_tokens: int = 0
    accepted_draft_tokens: int = 0
    start_time_s: float = 0.0
    end_time_s: float = 0.0

    @property
    def tokens_per_forward(self) -> float:
        if self.total_forward_passes == 0:
            return 1.0
        return self.total_tokens_generated / self.total_forward_passes

    @property
    def acceptance_rate(self) -> float:
        if self.total_draft_tokens == 0:
            return 0.0
        return self.accepted_draft_tokens / self.total_draft_tokens

    @property
    def elapsed_seconds(self) -> float:
        return max(0.0001, self.end_time_s - self.start_time_s)

    @property
    def tokens_per_second(self) -> float:
        return self.total_tokens_generated / self.elapsed_seconds


class PromptLookupDraftGenerator:
    """Extracts candidate draft tokens using prompt-lookup n-gram matching."""

    def __init__(self, max_draft_tokens: int = 4, ngram_size: int = 3):
        self.max_draft_tokens = max(1, max_draft_tokens)
        self.ngram_size = max(1, ngram_size)

    def generate_draft(self, input_ids: List[int]) -> List[int]:
        """Finds matching n-grams in the recent history and extracts continuation candidates."""
        if len(input_ids) <= self.ngram_size:
            return []

        suffix = input_ids[-self.ngram_size:]
        candidates: List[int] = []

        # Search backwards through input_ids for matching n-gram suffix
        search_limit = len(input_ids) - self.ngram_size
        for idx in range(search_limit - 1, -1, -1):
            if input_ids[idx:idx + self.ngram_size] == suffix:
                match_end = idx + self.ngram_size
                extract_len = min(self.max_draft_tokens, len(input_ids) - match_end)
                if extract_len > 0:
                    candidates = input_ids[match_end:match_end + extract_len]
                    break

        return candidates


class SpeculativeDecoder:
    """Manages speculative generation cycles with verify-and-accept logic."""

    def __init__(self, draft_k: int = 3):
        self.draft_k = draft_k
        self.lookup_generator = PromptLookupDraftGenerator(max_draft_tokens=draft_k)
        self.stats = SpeculativeStats()

    def step(self, prompt_tokens: List[int], target_verifier) -> Tuple[List[int], bool]:
        """Runs a speculative drafting and target verification step."""
        if self.stats.start_time_s == 0.0:
            self.stats.start_time_s = time.perf_counter()

        drafts = self.lookup_generator.generate_draft(prompt_tokens)
        accepted, next_token, is_eos = target_verifier(prompt_tokens, drafts)

        self.stats.total_draft_tokens += len(drafts)
        self.stats.accepted_draft_tokens += len(accepted)
        self.stats.total_forward_passes += 1

        new_tokens = list(accepted)
        if next_token is not None:
            new_tokens.append(next_token)

        self.stats.total_tokens_generated += len(new_tokens)
        self.stats.end_time_s = time.perf_counter()

        return new_tokens, is_eos
