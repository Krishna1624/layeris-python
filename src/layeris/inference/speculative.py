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

    def reset(self) -> None:
        self.total_tokens_generated = 0
        self.total_forward_passes = 0
        self.total_draft_tokens = 0
        self.accepted_draft_tokens = 0
        self.start_time_s = 0.0
        self.end_time_s = 0.0

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
        self.max_draft_tokens = max(0, max_draft_tokens)
        self.ngram_size = max(1, ngram_size)

    def generate_draft(self, input_ids: List[int]) -> List[int]:
        """Finds matching n-grams in the recent history and extracts continuation candidates."""
        if self.max_draft_tokens <= 0 or len(input_ids) <= self.ngram_size:
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


def verify_candidates(
    target_predict_fn,
    prompt_tokens: List[int],
    drafts: List[int],
    eos_token_id: Optional[int] = None,
) -> Tuple[List[int], Optional[int], bool]:
    """
    Standard speculative decoding candidate verification.

    Evaluates draft tokens sequentially or in parallel against target predictions.
    - Matching tokens are accepted.
    - The first mismatch rejects all remaining drafts and emits the target's prediction as correction.
    - If all drafts match, queries the target model for one bonus continuation token.
    - Falls back to a single target token if drafts is empty.
    """
    if not drafts:
        next_tok = target_predict_fn(prompt_tokens)
        is_eos = (eos_token_id is not None and next_tok == eos_token_id)
        return [], next_tok, is_eos

    accepted: List[int] = []
    prefix = list(prompt_tokens)

    for draft_tok in drafts:
        target_pred = target_predict_fn(prefix)
        if target_pred == draft_tok:
            accepted.append(draft_tok)
            prefix.append(draft_tok)
            if eos_token_id is not None and draft_tok == eos_token_id:
                return accepted, None, True
        else:
            is_eos = (eos_token_id is not None and target_pred == eos_token_id)
            return accepted, target_pred, is_eos

    # All drafts accepted -> query bonus token
    bonus_tok = target_predict_fn(prefix)
    is_eos = (eos_token_id is not None and bonus_tok == eos_token_id)
    return accepted, bonus_tok, is_eos


def make_model_verifier(model, device: str = "cpu", eos_token_id: Optional[int] = None):
    """
    Creates a target verifier from a PyTorch CausalLM using batched parallel forward verification.
    Evaluates all candidate tokens in a SINGLE forward pass for maximum speculative throughput.
    """
    import torch

    def verifier(prompt_tokens: List[int], drafts: List[int]) -> Tuple[List[int], Optional[int], bool]:
        if not drafts:
            input_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
            with torch.no_grad():
                logits = model(input_tensor).logits
            next_tok = int(logits[0, -1, :].argmax().item())
            is_eos = (eos_token_id is not None and next_tok == eos_token_id)
            return [], next_tok, is_eos

        # Batched single forward pass over prefix + drafts
        full_tokens = prompt_tokens + drafts
        input_tensor = torch.tensor([full_tokens], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(input_tensor).logits

        p_len = len(prompt_tokens)
        accepted: List[int] = []

        for i, draft_tok in enumerate(drafts):
            target_pred = int(logits[0, p_len - 1 + i, :].argmax().item())
            if target_pred == draft_tok:
                accepted.append(draft_tok)
                if eos_token_id is not None and draft_tok == eos_token_id:
                    return accepted, None, True
            else:
                is_eos = (eos_token_id is not None and target_pred == eos_token_id)
                return accepted, target_pred, is_eos

        bonus_pred = int(logits[0, p_len + len(drafts) - 1, :].argmax().item())
        is_eos = (eos_token_id is not None and bonus_pred == eos_token_id)
        return accepted, bonus_pred, is_eos

    return verifier


class ModelDraftGenerator:
    """Generates K draft candidate tokens using a separate lightweight assistant model."""

    def __init__(self, draft_model, max_draft_tokens: int = 3, device: str = "cpu"):
        self.draft_model = draft_model
        self.max_draft_tokens = max(0, max_draft_tokens)
        self.device = device

    def generate_draft(self, input_ids: List[int]) -> List[int]:
        if self.max_draft_tokens <= 0:
            return []
        import torch

        cur = list(input_ids)
        drafts: List[int] = []
        with torch.no_grad():
            for _ in range(self.max_draft_tokens):
                t = torch.tensor([cur], dtype=torch.long, device=self.device)
                logits = self.draft_model(t).logits
                next_tok = int(logits[0, -1, :].argmax().item())
                drafts.append(next_tok)
                cur.append(next_tok)
        return drafts


class SpeculativeDecoder:
    """Manages speculative generation cycles with verify-and-accept logic."""

    def __init__(self, draft_k: int = 3, draft_generator=None):
        self.draft_k = max(0, draft_k)
        if draft_generator is not None:
            self.draft_generator = draft_generator
        else:
            self.draft_generator = PromptLookupDraftGenerator(max_draft_tokens=self.draft_k)
        self.stats = SpeculativeStats()

    def reset_stats(self) -> None:
        self.stats.reset()

    def step(self, prompt_tokens: List[int], target_verifier) -> Tuple[List[int], bool]:
        """Runs a speculative drafting and target verification step."""
        if self.stats.start_time_s == 0.0:
            self.stats.start_time_s = time.perf_counter()

        if self.draft_k <= 0:
            drafts = []
        else:
            drafts = self.draft_generator.generate_draft(prompt_tokens)

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

    def generate(
        self,
        prompt_tokens: List[int],
        target_verifier,
        max_new_tokens: int = 64,
        eos_token_id: Optional[int] = None,
    ) -> List[int]:
        """Generates tokens until max_new_tokens is reached or EOS is hit."""
        current_tokens = list(prompt_tokens)
        generated_tokens: List[int] = []

        while len(generated_tokens) < max_new_tokens:
            step_tokens, is_eos = self.step(current_tokens, target_verifier)
            if not step_tokens:
                break

            remaining = max_new_tokens - len(generated_tokens)
            to_add = step_tokens[:remaining]
            generated_tokens.extend(to_add)
            current_tokens.extend(to_add)

            if is_eos or (eos_token_id is not None and eos_token_id in to_add):
                break

        return generated_tokens


class TwoModelSpeculativeEngine:
    """
    Two-model speculative decoding engine using an assistant draft model
    and a primary target model with parallel verification.
    """

    def __init__(
        self,
        target_model,
        draft_model,
        draft_k: int = 3,
        device: str = "cpu",
        eos_token_id: Optional[int] = None,
    ):
        self.target_model = target_model
        self.draft_model = draft_model
        self.draft_k = draft_k
        self.device = device
        self.eos_token_id = eos_token_id

        self.draft_generator = ModelDraftGenerator(draft_model, max_draft_tokens=draft_k, device=device)
        self.decoder = SpeculativeDecoder(draft_k=draft_k, draft_generator=self.draft_generator)
        self.verifier = make_model_verifier(target_model, device=device, eos_token_id=eos_token_id)

    @property
    def stats(self) -> SpeculativeStats:
        return self.decoder.stats

    def reset_stats(self) -> None:
        self.decoder.reset_stats()

    def generate(self, prompt_tokens: List[int], max_new_tokens: int = 64) -> List[int]:
        return self.decoder.generate(
            prompt_tokens=prompt_tokens,
            target_verifier=self.verifier,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.eos_token_id,
        )
