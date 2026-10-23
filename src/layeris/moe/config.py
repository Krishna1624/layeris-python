"""
Layeris MoE Configuration & Capability Verification
===================================================

Provides strict pre-flight validation and capability detection for Mixture-of-Experts (MoE)
models. Rejects dense-only architectures in accordance with the Layeris MoE-only mandate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class ValidationOutcome:
    """Canonical outcomes for model inspection."""
    SUPPORTED_MOE = "SUPPORTED_MOE"
    DENSE_REJECTED = "DENSE_REJECTED"
    UNSUPPORTED_MOE = "UNSUPPORTED_MOE"
    AMBIGUOUS_OR_INCOMPLETE = "AMBIGUOUS_OR_INCOMPLETE"


# Explicitly dense model types that must be rejected
DENSE_MODEL_TYPES = {
    "llama",
    "llama2",
    "llama3",
    "qwen2",
    "qwen2_5",
    "mistral",
    "gemma",
    "gemma2",
    "phi",
    "phi3",
    "bert",
    "modernbert",
    "roberta",
}

# Recognized MoE model type signatures
SUPPORTED_MOE_TYPES = {
    "qwen3_5_moe_text",
    "qwen3_moe",
    "qwen2_moe",
    "deepseek_v3",
    "deepseek_v2",
    "deepseek",
    "glm",
    "chatglm",
    "olmoe",
    "mimo",
    "kimi_k3",
}


@dataclass(frozen=True)
class MoEConfig:
    """Validated MoE architecture configuration."""
    model_type: str
    architectures: List[str]
    num_layers: int
    num_experts: int
    num_experts_per_tok: int
    shared_expert_intermediate_size: int = 0
    moe_intermediate_size: int = 0
    hidden_size: int = 0
    is_moe: bool = True
    validation_outcome: str = ValidationOutcome.SUPPORTED_MOE
    diagnostic: str = ""

    def validate(self) -> None:
        """Enforces structural invariants for MoE execution."""
        if not self.is_moe:
            raise ValueError(f"Dense model rejected: {self.diagnostic}")
        if self.num_experts <= 1:
            raise ValueError(f"Invalid MoE configuration: num_experts={self.num_experts} must be > 1.")
        if self.num_experts_per_tok <= 0:
            raise ValueError(f"Invalid MoE configuration: num_experts_per_tok={self.num_experts_per_tok} must be > 0.")
        if self.num_experts_per_tok > self.num_experts:
            raise ValueError(
                f"Contradictory routing: top_k ({self.num_experts_per_tok}) > total experts ({self.num_experts})"
            )


def detect_moe_architecture(config_or_path: Union[Dict[str, Any], str, Path]) -> MoEConfig:
    """Detects and validates whether the given configuration is a supported MoE architecture.
    
    Raises:
        ValueError: If the configuration represents a dense-only model or malformed configuration.
    """
    raw_cfg: Dict[str, Any]
    if isinstance(config_or_path, (str, Path)):
        p = Path(config_or_path)
        if p.is_dir():
            cfg_file = p / "config.json"
        else:
            cfg_file = p
        if not cfg_file.is_file():
            raise FileNotFoundError(f"Config file not found at {cfg_file}")
        with open(cfg_file, "r", encoding="utf-8") as f:
            raw_cfg = json.load(f)
    elif isinstance(config_or_path, dict):
        raw_cfg = config_or_path
    else:
        raise TypeError(f"Expected dict, str, or Path, got {type(config_or_path)}")

    # Handle nested text_config if present
    text_cfg = raw_cfg.get("text_config", raw_cfg)

    m_type = str(text_cfg.get("model_type", raw_cfg.get("model_type", ""))).lower().strip()
    archs = text_cfg.get("architectures", raw_cfg.get("architectures", []))
    if isinstance(archs, str):
        archs = [archs]

    # Explicit dense model check
    if m_type in DENSE_MODEL_TYPES:
        diag = f"Model type '{m_type}' is an explicitly dense architecture (MoE-only engine rejects dense models)."
        raise ValueError(diag)

    # Extract expert counts across vendor schemas
    num_experts = (
        text_cfg.get("num_experts")
        or text_cfg.get("n_routed_experts")
        or text_cfg.get("num_local_experts")
        or raw_cfg.get("num_experts")
        or raw_cfg.get("n_routed_experts")
        or 0
    )

    # Extract top-k routed experts
    top_k = (
        text_cfg.get("num_experts_per_tok")
        or text_cfg.get("num_experts_per_token")
        or text_cfg.get("num_selected_experts")
        or raw_cfg.get("num_experts_per_tok")
        or 0
    )

    layers = (
        text_cfg.get("num_hidden_layers")
        or text_cfg.get("n_layer")
        or text_cfg.get("num_layers")
        or raw_cfg.get("num_hidden_layers")
        or 0
    )

    hidden = (
        text_cfg.get("hidden_size")
        or text_cfg.get("d_model")
        or raw_cfg.get("hidden_size")
        or 0
    )

    shared_inter = (
        text_cfg.get("shared_expert_intermediate_size")
        or raw_cfg.get("shared_expert_intermediate_size")
        or 0
    )

    moe_inter = (
        text_cfg.get("moe_intermediate_size")
        or raw_cfg.get("moe_intermediate_size")
        or 0
    )

    # Check for dense rejection if no routed experts
    if num_experts is None or int(num_experts) <= 1:
        diag = f"Rejected: architecture '{m_type}' has num_experts={num_experts} (requires > 1 for MoE)."
        raise ValueError(diag)

    num_experts = int(num_experts)
    top_k = int(top_k)
    layers = int(layers)
    hidden = int(hidden)
    shared_inter = int(shared_inter)
    moe_inter = int(moe_inter)

    if top_k <= 0:
        raise ValueError(f"Invalid routing configuration: top_k={top_k} must be >= 1")
    if top_k > num_experts:
        raise ValueError(f"Contradictory routing: top_k={top_k} > num_experts={num_experts}")

    # Verify recognized MoE family
    is_known_moe = (
        m_type in SUPPORTED_MOE_TYPES
        or any("moe" in a.lower() for a in archs)
        or (m_type == "glm" and num_experts > 1)
    )

    outcome = ValidationOutcome.SUPPORTED_MOE if is_known_moe else ValidationOutcome.UNSUPPORTED_MOE
    diag = f"MoE model '{m_type}': {layers} layers, {num_experts} experts, top-{top_k} routed per token."

    cfg = MoEConfig(
        model_type=m_type,
        architectures=archs,
        num_layers=layers,
        num_experts=num_experts,
        num_experts_per_tok=top_k,
        shared_expert_intermediate_size=shared_inter,
        moe_intermediate_size=moe_inter,
        hidden_size=hidden,
        is_moe=True,
        validation_outcome=outcome,
        diagnostic=diag,
    )
    cfg.validate()
    return cfg
