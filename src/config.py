"""
Central configuration for BIS RAG Engine.
All tunable hyperparameters live here.

LLM Provider Priority:
  1. OpenRouter  (recommended — single key, access to 100+ models, cheapest)
  2. Anthropic   (direct)
  3. OpenAI      (direct)
  4. Groq        (fast/free tier)
  5. Template    (zero-cost fallback — no LLM needed, still functional)

Set LLM_PROVIDER=openrouter and OPENROUTER_API_KEY in your .env
"""
import os
from pathlib import Path
from dataclasses import dataclass, field

# Load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv optional — env vars can be set manually

BASE_DIR = Path(__file__).parent.parent


@dataclass
class Config:
    # ── Paths ──────────────────────────────────────────────────────────────
    data_dir: Path = field(default_factory=lambda: BASE_DIR / "data")
    index_dir: Path = field(default_factory=lambda: BASE_DIR / "data" / "index")
    pdf_path: Path = field(
        default_factory=lambda: BASE_DIR / "data" / "bis_sp21.pdf"
    )

    # ── Embedding model ────────────────────────────────────────────────────
    # BGE-base: best retrieval quality / speed tradeoff for technical text
    # Runs 100% locally — no API key needed for embeddings
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_dim: int = 768
    embedding_batch_size: int = 64
    # BGE requires this prefix for QUERY embeddings only (not documents)
    query_instruction: str = "Represent this sentence for searching relevant passages: "

    # ── BM25 ───────────────────────────────────────────────────────────────
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # ── Retrieval pipeline ─────────────────────────────────────────────────
    top_k_dense: int = 100     # candidates from FAISS dense retrieval
    top_k_bm25: int = 100      # candidates from BM25 sparse retrieval
    top_k_rerank: int = 30     # candidates passed to cross-encoder reranker
    top_k_final: int = 5       # final results returned to user

    # ── RRF (Reciprocal Rank Fusion) ───────────────────────────────────────
    rrf_k: int = 60            # standard RRF constant (don't change unless tuning)

    # ── Cross-encoder reranker ─────────────────────────────────────────────
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_batch_size: int = 32

    # ── LLM Provider ───────────────────────────────────────────────────────
    # Supported: "openrouter" | "anthropic" | "openai" | "groq" | "none"
    # "none" = skip LLM, use fast template rationale (good enough for eval)
    llm_provider: str = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "openrouter")
    )

    # ── OpenRouter (PRIMARY — recommended) ────────────────────────────────
    # Get key at: https://openrouter.ai/keys
    # Compatible with OpenAI SDK (drop-in replacement)
    openrouter_api_key: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_API_KEY", "")
    )
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    # Recommended free/cheap models on OpenRouter:
    #   "meta-llama/llama-3.1-8b-instruct:free"  ← completely free
    #   "google/gemma-3-27b-it:free"              ← free, stronger
    #   "anthropic/claude-haiku-4-5"              ← paid, best quality
    #   "openai/gpt-4o-mini"                      ← paid, reliable
    openrouter_model: str = field(
        default_factory=lambda: os.getenv(
            "OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free"
        )
    )
    # Site info for OpenRouter (required in headers for free tier)
    openrouter_site_url: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_SITE_URL", "http://localhost:3000")
    )
    openrouter_site_name: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_SITE_NAME", "BIS RAG Engine")
    )

    # ── Fallback providers ─────────────────────────────────────────────────
    anthropic_api_key: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", "")
    )
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    groq_api_key: str = field(
        default_factory=lambda: os.getenv("GROQ_API_KEY", "")
    )

    # ── LLM generation settings ────────────────────────────────────────────
    # llm_model is used for non-openrouter providers
    llm_model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "meta-llama/llama-3.1-8b-instruct:free")
    )
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.1

    # ── API server ─────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ── Chunking ───────────────────────────────────────────────────────────
    max_chunk_tokens: int = 512
    chunk_overlap: int = 50

    def get_active_llm_model(self) -> str:
        """Return the model string for the active provider."""
        if self.llm_provider == "openrouter":
            return self.openrouter_model
        return self.llm_model

    def has_llm(self) -> bool:
        """Check if any LLM is configured."""
        if self.llm_provider == "none":
            return False
        if self.llm_provider == "openrouter":
            return bool(self.openrouter_api_key)
        if self.llm_provider == "anthropic":
            return bool(self.anthropic_api_key)
        if self.llm_provider == "openai":
            return bool(self.openai_api_key)
        if self.llm_provider == "groq":
            return bool(self.groq_api_key)
        return False


# Singleton — imported everywhere
config = Config()
