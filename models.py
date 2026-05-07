from dataclasses import dataclass
from typing import List, Optional


@dataclass
class AgentGeneration:
    agent_id: str
    output_text: str
    token_logprobs: List[float]
    is_faulty: bool
    fault_type: Optional[str]  # None | 'F1_crash' | 'F2_byzantine' | 'F3_drifter'
    model_id: Optional[str] = None  # e.g. "meta-llama/Meta-Llama-3.1-8B-Instruct"
    provider: Optional[str] = None  # e.g. "llama" | "qwen" | "mistral" | "phi3"


@dataclass
class ConsensusResult:
    final_answer: str
    admitted_agents: List[str]
    is_low_confidence: bool
