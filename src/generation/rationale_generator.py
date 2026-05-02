"""
LLM Rationale Generator
=========================
Generates a brief, grounded rationale for why each BIS standard was
recommended — using only information from the retrieved context.

Anti-hallucination measures:
  1. System prompt explicitly forbids fabricating IS numbers.
  2. Rationale is generated per-standard, not as a free-form essay.
  3. Fallback template used if LLM call fails or returns empty.
  4. Response is post-processed to strip any IS number NOT in context.

Supports: Anthropic Claude, OpenAI GPT-4o-mini, Groq (llama-3.1-8b)
"""
import json
import logging
import re
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a BIS (Bureau of Indian Standards) compliance expert assistant.
Your job is to explain, briefly and accurately, why a specific BIS standard applies to a 
product or use-case described by the user.

STRICT RULES:
1. Only reference IS numbers that appear in the context provided — never fabricate.
2. Each rationale must be 1-3 sentences maximum. Be specific, not generic.
3. Focus on the match between the product description and the standard's scope.
4. Do NOT say "this standard covers" in every sentence — vary your language.
5. Output ONLY valid JSON. No markdown, no preamble."""

RATIONALE_PROMPT_TEMPLATE = """Product description: {query}

Retrieved BIS standards (use ONLY these):
{context}

For each standard, output a JSON array with objects having exactly these keys:
  "is_number": the exact IS number string (e.g., "IS 269 : 1989")
  "title": the standard title
  "rationale": 1-3 sentence explanation of why this standard applies to the product

Return ONLY a JSON array. Example:
[
  {{"is_number": "IS 269 : 1989", "title": "33 Grade OPC", "rationale": "..."}}
]"""


def _build_context_block(standards: List[Dict[str, Any]]) -> str:
    """Format retrieved standards as a numbered context block for the LLM."""
    lines = []
    for i, std in enumerate(standards, 1):
        lines.append(
            f"{i}. {std['is_number_full']} — {std['title']}\n"
            f"   Category: {std.get('category', 'N/A')}\n"
            f"   Section: {std.get('section_name', 'General')}"
        )
    return "\n\n".join(lines)


def _template_rationale(meta: Dict[str, Any], query: str) -> str:
    """
    Fast template-based rationale as fallback or when LLM is unavailable.
    Grounded entirely in metadata — zero hallucination risk.
    """
    title = meta.get("title", "")
    category = meta.get("category", "Building Materials")
    is_num = meta.get("is_number_full", "")

    # Extract key query terms
    query_lower = query.lower()
    match_hints = []
    if "cement" in query_lower:
        match_hints.append("cement manufacturing requirements")
    if "concrete" in query_lower:
        match_hints.append("concrete specification and testing")
    if "aggregate" in query_lower:
        match_hints.append("aggregate grading and quality requirements")
    if "steel" in query_lower or "reinforcement" in query_lower:
        match_hints.append("steel and reinforcement specifications")
    if "pipe" in query_lower:
        match_hints.append("pipe manufacturing and testing standards")
    if "masonry" in query_lower or "block" in query_lower:
        match_hints.append("masonry unit dimensions and strength")

    hint_str = " and ".join(match_hints) if match_hints else "your described product"

    return (
        f"{is_num} specifies the requirements for {title.lower() if title else 'this product type'}. "
        f"It directly governs {hint_str} under the {category} category of BIS SP 21."
    )


def _strip_hallucinated_is_numbers(
    rationale: str, valid_is_numbers: List[str]
) -> str:
    """
    Remove any IS numbers in the rationale that aren't in the valid set.
    Prevents the LLM from hallucinating adjacent standards.
    """
    is_pattern = re.compile(
        r"IS\s+\d+(?:\s*\(Part\s*\d+\))?(?:\s*\(Sec\s*\d+\))?\s*:\s*\d{4}",
        re.IGNORECASE,
    )
    valid_normalised = {
        s.replace(" ", "").lower() for s in valid_is_numbers
    }

    def check_replace(m):
        found = m.group(0).replace(" ", "").lower()
        if found in valid_normalised:
            return m.group(0)
        return "[standard]"

    return is_pattern.sub(check_replace, rationale)


class RationaleGenerator:
    """
    Generates per-standard rationales using an LLM (or fast templates).
    """

    def __init__(self, config):
        self.config = config

    def _call_openrouter(self, prompt: str) -> Optional[str]:
        """
        Call OpenRouter API (primary recommended provider).

        OpenRouter is an OpenAI-compatible proxy that gives access to 100+ models
        including free tiers of Llama, Gemma, Mistral etc.
        Uses the openai SDK as a drop-in — just swap base_url + api_key.

        Free models to try (no cost):
          meta-llama/llama-3.1-8b-instruct:free
          google/gemma-3-27b-it:free
          mistralai/mistral-7b-instruct:free
        """
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=self.config.openrouter_api_key,
                base_url=self.config.openrouter_base_url,
                default_headers={
                    # Required by OpenRouter for free tier usage tracking
                    "HTTP-Referer": self.config.openrouter_site_url,
                    "X-Title": self.config.openrouter_site_name,
                },
            )
            resp = client.chat.completions.create(
                model=self.config.openrouter_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.config.llm_max_tokens,
                temperature=self.config.llm_temperature,
            )
            text = resp.choices[0].message.content
            logger.debug(
                f"OpenRouter [{self.config.openrouter_model}] "
                f"tokens: {resp.usage.total_tokens if resp.usage else 'N/A'}"
            )
            return text
        except Exception as e:
            logger.warning(f"OpenRouter call failed: {e}")
            return None

    def _call_anthropic(self, prompt: str) -> Optional[str]:
        """Call Anthropic Claude API directly (fallback)."""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.config.anthropic_api_key)
            msg = client.messages.create(
                model=self.config.llm_model,
                max_tokens=self.config.llm_max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.llm_temperature,
            )
            return msg.content[0].text
        except Exception as e:
            logger.warning(f"Anthropic call failed: {e}")
            return None

    def _call_openai(self, prompt: str) -> Optional[str]:
        """Call OpenAI API directly (fallback)."""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.config.openai_api_key)
            resp = client.chat.completions.create(
                model=self.config.llm_model or "gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.config.llm_max_tokens,
                temperature=self.config.llm_temperature,
            )
            return resp.choices[0].message.content
        except Exception as e:
            logger.warning(f"OpenAI call failed: {e}")
            return None

    def _call_groq(self, prompt: str) -> Optional[str]:
        """Call Groq API (fast free inference, fallback)."""
        try:
            from groq import Groq
            client = Groq(api_key=self.config.groq_api_key)
            resp = client.chat.completions.create(
                model=self.config.llm_model or "llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.config.llm_max_tokens,
                temperature=self.config.llm_temperature,
            )
            return resp.choices[0].message.content
        except Exception as e:
            logger.warning(f"Groq call failed: {e}")
            return None

    def _call_llm(self, prompt: str) -> Optional[str]:
        """
        Route to the configured LLM provider.
        Falls back through providers if primary fails.
        """
        provider = self.config.llm_provider.lower()

        # Primary: OpenRouter (recommended)
        if provider == "openrouter":
            if not self.config.openrouter_api_key:
                logger.warning(
                    "OPENROUTER_API_KEY not set. "
                    "Set it in .env or use LLM_PROVIDER=none for template fallback."
                )
                return None
            result = self._call_openrouter(prompt)
            if result:
                return result
            # Auto-fallback to other providers if OpenRouter fails
            logger.warning("OpenRouter failed, attempting fallbacks...")

        # Fallback chain
        if self.config.anthropic_api_key:
            return self._call_anthropic(prompt)
        if self.config.openai_api_key:
            return self._call_openai(prompt)
        if self.config.groq_api_key:
            return self._call_groq(prompt)

        logger.warning(
            "No LLM provider available. Using template rationale. "
            "Set OPENROUTER_API_KEY in .env for AI-generated rationales."
        )
        return None

    def _parse_llm_response(
        self, response: str, valid_is_numbers: List[str]
    ) -> Dict[str, str]:
        """
        Parse LLM JSON response into {is_number_full: rationale} map.
        Handles common JSON formatting issues.
        """
        if not response:
            return {}

        # Strip markdown code fences
        response = re.sub(r"```(?:json)?", "", response).strip().rstrip("`")

        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON array with regex
            match = re.search(r"\[.*\]", response, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                except Exception:
                    return {}
            else:
                return {}

        result = {}
        for item in parsed:
            if not isinstance(item, dict):
                continue
            is_num = item.get("is_number", "")
            rationale = item.get("rationale", "")
            if is_num and rationale:
                # Anti-hallucination: strip invalid IS references
                rationale = _strip_hallucinated_is_numbers(rationale, valid_is_numbers)
                result[is_num] = rationale

        return result

    def generate(
        self,
        query: str,
        retrieved_standards: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Generate rationales for retrieved standards.

        Args:
            query: User's product description query
            retrieved_standards: List of metadata dicts from reranker

        Returns:
            List of dicts with keys: is_number_full, title, category,
            rationale, is_number, year
        """
        if not retrieved_standards:
            return []

        valid_is_numbers = [s["is_number_full"] for s in retrieved_standards]
        context_block = _build_context_block(retrieved_standards)
        prompt = RATIONALE_PROMPT_TEMPLATE.format(
            query=query, context=context_block
        )

        # Try LLM first
        rationale_map: Dict[str, str] = {}
        llm_response = self._call_llm(prompt)
        if llm_response:
            rationale_map = self._parse_llm_response(llm_response, valid_is_numbers)

        # Build output list
        output = []
        for meta in retrieved_standards:
            is_full = meta["is_number_full"]
            # Use LLM rationale if available, else fallback to template
            rationale = rationale_map.get(is_full) or _template_rationale(meta, query)

            output.append(
                {
                    "is_number_full": is_full,
                    "is_number": meta.get("is_number", ""),
                    "year": meta.get("year", ""),
                    "title": meta.get("title", ""),
                    "category": meta.get("category", ""),
                    "rationale": rationale,
                }
            )

        return output
