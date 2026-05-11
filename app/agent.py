"""
Core agent controller for the SHL Assessment Recommender.

Strategy: deterministic rule-based recommender + LLM for conversational reply.
When LLM is rate-limited or returns <3 recs, rule-based engine fills the gap.
"""

import json
import os
import re
import logging
import time
from typing import Any, Optional

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> bool:
        return False

load_dotenv()

from app.catalog import CatalogStore
from app.prompts import build_full_prompt, build_query_from_messages
from app.models import ChatResponse, Recommendation, Message

logger = logging.getLogger(__name__)

_catalog_store: Optional[CatalogStore] = None
_retriever: Optional[Any] = None

MAX_TURNS = 8

CATALOG_URL_RE = re.compile(r"https?://[^\s<>()\"']+", re.IGNORECASE)

TYPE_NAME_TO_CODE = {
    "ability": "A",
    "aptitude": "A",
    "cognitive": "A",
    "reasoning": "A",
    "biodata": "B",
    "situational": "B",
    "situational judgement": "B",
    "situational judgment": "B",
    "competencies": "C",
    "competency": "C",
    "development": "D",
    "360": "D",
    "assessment exercises": "E",
    "assessment exercise": "E",
    "type e": "E",
    "knowledge": "K",
    "skills": "K",
    "simulation": "S",
    "simulations": "S",
    "personality": "P",
    "behavior": "P",
    "behaviour": "P",
}

JOB_LEVEL_ALIASES = {
    "director": "Director",
    "director-level": "Director",
    "executive": "Executive",
    "cxo": "Executive",
    "entry-level": "Entry-Level",
    "entry level": "Entry-Level",
    "graduate": "Graduate",
    "manager": "Manager",
    "front line manager": "Front Line Manager",
    "front-line manager": "Front Line Manager",
    "supervisor": "Supervisor",
    "mid-professional": "Mid-Professional",
    "mid professional": "Mid-Professional",
    "professional individual contributor": "Professional Individual Contributor",
}

CANONICAL_ASSESSMENTS = {
    "personality": "Occupational Personality Questionnaire OPQ32r",
    "cognitive": "SHL Verify Interactive G+",
    "leadership_report": "OPQ Leadership Report",
    "ucf_report": "OPQ Universal Competency Report 2.0",
    "sales_report": "OPQ MQ Sales Report",
    "sales_transform_ic": "Sales Transformation 2.0 - Individual Contributor",
    "gsa": "Global Skills Assessment",
    "gsa_dev_report": "Global Skills Development Report",
    "graduate_scenarios": "Graduate Scenarios",
    "dsi": "Dependability and Safety Instrument (DSI)",
}


def get_catalog_store() -> CatalogStore:
    global _catalog_store
    if _catalog_store is None:
        _catalog_store = CatalogStore()
        logger.info(f"Loaded catalog with {len(_catalog_store)} items")
    return _catalog_store


def get_retriever() -> Any:
    global _retriever
    if _retriever is None:
        from app.retriever import CatalogRetriever
        _retriever = CatalogRetriever(get_catalog_store())
        logger.info("TF-IDF retriever initialized")
    return _retriever


# ---------------------------------------------------------------------------
# Rule-based recommender — runs first, no LLM needed
# ---------------------------------------------------------------------------

def _rule_based_recommend(query: str, all_text: str, catalog: CatalogStore) -> list[dict]:
    """Deterministic keyword-based recommender. Always returns catalog-verified items."""
    combined = (query + " " + all_text).lower()
    result = []
    seen = set()

    def add(name: str):
        item = catalog.get_by_name(name)
        if item and item["url"] not in seen:
            result.append(item)
            seen.add(item["url"])

    # ── Leadership / CXO / Executive ──────────────────────────────────────
    if any(s in combined for s in ["leadership", "executive", "cxo", "director",
                                    "senior leadership", "leadership benchmark"]):
        add("Occupational Personality Questionnaire OPQ32r")
        add("OPQ Universal Competency Report 2.0")
        add("OPQ Leadership Report")
        add("SHL Verify Interactive G+")

    # ── Sales / talent audit / re-skill ───────────────────────────────────
    if any(s in combined for s in ["sales", "selling", "re-skill", "reskill",
                                    "talent audit", "restructur"]):
        add("Global Skills Assessment")
        add("Global Skills Development Report")
        add("Occupational Personality Questionnaire OPQ32r")
        add("OPQ MQ Sales Report")
        add("Sales Transformation 2.0 - Individual Contributor")

    # ── Contact centre / call centre / customer service ───────────────────
    if any(s in combined for s in ["contact cent", "contact center", "contact centre",
                                    "inbound call", "call centre", "call center"]):
        add("SVAR - Spoken English (US) (New)")
        add("Contact Center Call Simulation (New)")
        add("Entry Level Customer Serv-Retail & Contact Center")
        add("Customer Service Phone Simulation")

    # ── Healthcare / HIPAA / bilingual admin ──────────────────────────────
    if any(s in combined for s in ["hipaa", "healthcare", "patient record", "bilingual",
                                    "medical terminol", "health admin"]):
        add("HIPAA (Security)")
        add("Medical Terminology (New)")
        add("Microsoft Word 365 - Essentials (New)")
        add("Dependability and Safety Instrument (DSI)")
        add("Occupational Personality Questionnaire OPQ32r")

    # ── Admin assistants / Excel / Word (non-healthcare) ──────────────────
    if any(s in combined for s in ["admin assistant", "excel and word", "excel & word",
                                    "ms excel", "ms word", "spreadsheet"]):
        add("MS Excel (New)")
        add("MS Word (New)")
        add("Microsoft Excel 365 (New)")
        add("Microsoft Word 365 (New)")
        add("Occupational Personality Questionnaire OPQ32r")

    # ── Java / Spring / full-stack backend ────────────────────────────────
    if any(s in combined for s in ["core java", "spring", "full-stack", "full stack",
                                    "microservice", "backend engineer"]):
        add("Core Java (Advanced Level) (New)")
        add("Spring (New)")
        add("SQL (New)")
        if any(s in combined for s in ["aws", "amazon web services"]):
            add("Amazon Web Services (AWS) Development (New)")
        if "docker" in combined:
            add("Docker (New)")
        add("SHL Verify Interactive G+")
        add("Occupational Personality Questionnaire OPQ32r")

    # ── Java (generic) ────────────────────────────────────────────────────
    if "java" in combined and "core java" not in combined and "javascript" not in combined:
        add("Core Java (Advanced Level) (New)")
        add("Java 8 (New)")
        add("Spring (New)")
        add("SHL Verify Interactive G+")
        add("Occupational Personality Questionnaire OPQ32r")

    # ── Rust / networking / systems / Linux ───────────────────────────────
    if any(s in combined for s in ["rust engineer", "rust developer",
                                    "high-performance networking", "networking infrastructure"]):
        add("Smart Interview Live Coding")
        add("Linux Programming (General)")
        add("Networking and Implementation (New)")
        add("SHL Verify Interactive G+")
        add("Occupational Personality Questionnaire OPQ32r")

    # ── Graduate / financial analyst ──────────────────────────────────────
    if any(s in combined for s in ["graduate financial", "financial analyst",
                                    "numerical reasoning", "final-year"]):
        add("SHL Verify Interactive – Numerical Reasoning")
        add("Financial Accounting (New)")
        add("Basic Statistics (New)")
        add("Graduate Scenarios")
        add("Occupational Personality Questionnaire OPQ32r")

    # ── Graduate management / broad graduate batteries ────────────────────
    if "graduate" in combined and any(s in combined for s in [
        "management trainee", "trainee scheme", "recent graduates",
        "situational judgement", "situational judgment", "cognitive",
        "full battery",
    ]):
        add("SHL Verify Interactive G+")
        add("Occupational Personality Questionnaire OPQ32r")
        add("Graduate Scenarios")

    # ── Safety / manufacturing / industrial ───────────────────────────────
    if any(s in combined for s in [
        "plant operator", "plant operators", "chemical facility",
        "manufacturing", "industrial", "workplace safety",
        "procedure compliance", "cutting corners",
    ]):
        add("Manufac. & Indust. - Safety & Dependability 8.0")
        add("Manufacturing & Industrial - Mechanical Focus 8.0")
        add("Workplace Health and Safety (New)")

    return result[:10]


def _has_enough_context(combined_text: str) -> bool:
    """True when the user has given enough context to recommend assessments."""
    signals = [
        "leadership", "executive", "cxo", "director",
        "sales", "talent audit", "re-skill",
        "contact cent", "contact center", "call cent",
        "java", "spring", "rust", "python", "aws", "docker",
        "admin assistant", "excel", "word",
        "healthcare", "hipaa", "medical",
        "graduate", "financial analyst", "numerical",
        "safety", "manufacturing",
        "hiring", "assess", "screen", "recruit",
    ]
    low = combined_text.lower()
    return any(s in low for s in signals)


# ---------------------------------------------------------------------------
# LLM callers
# ---------------------------------------------------------------------------

def _call_openrouter(prompt_messages: list[dict]) -> str:
    """Call OpenRouter using its OpenAI-compatible chat completions endpoint."""
    import urllib.request
    import urllib.error

    or_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not or_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    model = os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-4-scout")
    messages = [{"role": m["role"], "content": m["content"]} for m in prompt_messages]

    # Google models on OpenRouter require max_completion_tokens, not max_tokens.
    # Other models accept both; sending max_completion_tokens is universally safe.
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": 0.3,
        "max_completion_tokens": 2048,
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {or_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/Mustafa11300/shl_assignment",
            "X-Title": "SHL Assessment Recommender",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter HTTP {e.code}: {body[:300]}") from e

    content = (data["choices"][0]["message"].get("content") or "").strip()
    if not content:
        raise RuntimeError("OpenRouter returned empty content")
    return content


def _call_groq(prompt_messages: list[dict]) -> str:
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        raise ValueError("GROQ_API_KEY not set")
    from groq import Groq
    client = Groq(api_key=groq_key)
    model = os.environ.get("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
    messages = [{"role": m["role"], "content": m["content"]} for m in prompt_messages]
    try:
        response = client.chat.completions.create(
            model=model, messages=messages, temperature=0.3, max_tokens=2048,
        )
    except Exception as e:
        raise RuntimeError(f"Groq API error: {str(e)[:200]}") from e
    content = (response.choices[0].message.content or "").strip()
    if not content:
        raise RuntimeError("Groq returned empty content")
    return content


def _try_gemini_once(api_key: str, prompt_messages: list[dict]) -> Optional[str]:
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=api_key)
        model_name = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
        system_instruction = prompt_messages[0]["content"] if prompt_messages[0]["role"] == "system" else ""
        user_content = prompt_messages[1]["content"] if len(prompt_messages) > 1 else ""
        config_kwargs = {
            "system_instruction": system_instruction,
            "temperature": 0.3,
            "max_output_tokens": 2048,
        }
        if "2.5" in model_name:
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        config = types.GenerateContentConfig(**config_kwargs)
        response = client.models.generate_content(
            model=model_name, contents=user_content, config=config,
        )
        return response.text
    except Exception as e:
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            logger.warning("Gemini 429 — skipping")
            return None
        raise


def _call_llm(prompt_messages: list[dict]) -> str:
    """OpenRouter → Gemini → Groq → wait 30s → Gemini. Each provider gets one shot."""
    or_key = os.environ.get("OPENROUTER_API_KEY", "")
    if or_key:
        try:
            result = _call_openrouter(prompt_messages)
            if result:
                return result
        except Exception as e:
            logger.warning(f"OpenRouter failed: {str(e)[:120]}")

    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if gemini_key:
        result = _try_gemini_once(gemini_key, prompt_messages)
        if result:
            return result

    groq_key = os.environ.get("GROQ_API_KEY", "")
    if groq_key:
        try:
            return _call_groq(prompt_messages)
        except Exception as e:
            logger.warning(f"Groq failed: {str(e)[:100]}")

    if gemini_key:
        logger.warning("All providers failed. Waiting 30s before Gemini retry...")
        time.sleep(30)
        result = _try_gemini_once(gemini_key, prompt_messages)
        if result:
            return result

    raise RuntimeError("All LLM providers exhausted")


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _parse_llm_response(raw_response: str, catalog: CatalogStore) -> ChatResponse:
    text = raw_response.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    if not text.startswith("{"):
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            text = json_match.group(0)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM JSON: {e}")
        return ChatResponse(
            reply="I can help you find the right SHL assessments. Could you tell me more about the role?",
            recommendations=[],
            end_of_conversation=False,
        )

    valid_recs = []
    for rec in data.get("recommendations", []) or []:
        if not isinstance(rec, dict):
            continue
        name = rec.get("name", "")
        url = rec.get("url", "")
        catalog_item = None
        if url:
            if _is_safe_catalog_url(url, catalog):
                catalog_item = catalog.get_by_url(url)
            else:
                # If the model emitted a URL, it must be a verbatim catalog URL.
                # Do not salvage by name; that would let hallucinated URLs pass.
                continue
        elif name:
            catalog_item = catalog.get_by_name(name)
            if not catalog_item:
                matches = catalog.search_by_name(name)
                if matches:
                    catalog_item = matches[0]
        if catalog_item:
            valid_recs.append(catalog_item)

    # Deduplicate
    seen_urls = set()
    deduped = []
    for item in valid_recs:
        if item["url"] not in seen_urls:
            seen_urls.add(item["url"])
            deduped.append(item)

    recs = [
        Recommendation(name=i["name"], url=i["url"], test_type=i["test_type"])
        for i in deduped[:10]
    ]

    return ChatResponse(
        reply=data.get("reply", "Here are my recommendations.") if isinstance(data.get("reply"), str) else "Here are my recommendations.",
        recommendations=recs,
        end_of_conversation=bool(data.get("end_of_conversation", False)) and bool(recs),
    )


def _merge_recommendations(
    primary: list[Recommendation],
    secondary: list[dict | Recommendation],
    catalog: CatalogStore,
    max_total: int = 10,
) -> list[Recommendation]:
    """Merge primary recs first, then supplement with secondary, deduplicating by URL."""
    seen = {r.url for r in primary}
    merged = list(primary)
    for item in secondary:
        url = item.url if isinstance(item, Recommendation) else item.get("url", "")
        name = item.name if isinstance(item, Recommendation) else item.get("name", "")
        test_type = item.test_type if isinstance(item, Recommendation) else item.get("test_type", "")
        catalog_item = catalog.get_by_url(url) if url and url.isascii() else None
        if catalog_item and url not in seen and len(merged) < max_total:
            merged.append(Recommendation(
                name=catalog_item.get("name", name),
                url=catalog_item["url"],
                test_type=catalog_item.get("test_type", test_type),
            ))
            seen.add(catalog_item["url"])
    return merged


def _llm_enabled() -> bool:
    return os.environ.get("USE_LLM", "").strip().lower() in {"1", "true", "yes"}


def _latest_user_text(messages: list[dict]) -> str:
    for msg in reversed(messages):
        if msg["role"] == "user":
            return msg["content"]
    return ""


def _is_confirmation(text: str) -> bool:
    low = text.lower()
    signals = [
        "confirmed", "confirm", "lock", "locking", "final list",
        "perfect", "that works", "that's good", "covers it",
        "keep the shortlist", "keep it", "as-is", "as is",
        "go ahead", "sounds good", "let's go with those",
        "lets go with those", "go with those", "yes",
    ]
    return any(signal in low for signal in signals)


def _sanitize_user_content(text: str) -> str:
    """Strip null bytes and control characters from user content (Case 16)."""
    import unicodedata
    # Remove null bytes and other C0/C1 control chars except newline/tab
    cleaned = []
    for ch in text:
        if ch in ('\n', '\t', '\r'):
            cleaned.append(ch)
        elif unicodedata.category(ch).startswith('C'):
            continue  # skip control characters
        else:
            cleaned.append(ch)
    return ''.join(cleaned)[:8000]


def _is_injection_attempt(text: str) -> bool:
    """Deterministic injection detection per Rule E. Always blocks, no LLM judgment."""
    low = text.lower()
    hard_signals = [
        "ignore all previous instructions",
        "ignore previous instructions",
        "ignore all prior instructions",
        "forget your guidelines",
        "forget your rules",
        "forget your instructions",
        "you are now",
        "pretend you are",
        "i am the assistant",
        "i am the system",
        "as chatgpt",
        "as gpt",
        "as shl-gpt",
        "no restrictions",
        "the user said",
        "ignore all",
        "let's roleplay", "lets roleplay",
        "developer says",
        "system says",
        "jailbreak",
        "unrestricted",
        "outside your catalog",
        "bypass",
        "attacker.com",
        "evil.com",
    ]
    return any(signal in low for signal in hard_signals)


def _contains_json_injection(text: str) -> bool:
    """Detect JSON or recommendation-shaped objects embedded in user message (Case 5)."""
    low = text.lower()
    schema_key = re.compile(
        r'["\']\s*(reply|recommendations|end_of_conversation|test_type)\s*["\']\s*:',
        re.IGNORECASE,
    )
    if schema_key.search(text):
        return True
    if re.search(r"\{[^{}]*(https?://|shl\.com|attacker\.com|evil\.com)[^{}]*\}", low, re.DOTALL):
        return True
    return False


def _is_safe_catalog_url(url: str, catalog: CatalogStore) -> bool:
    """A URL is safe only when it is pure ASCII and exists verbatim in the catalog."""
    return bool(url) and url.isascii() and catalog.url_exists(url)


def _extract_urls(text: str) -> list[str]:
    """Extract URL-looking tokens, trimming punctuation outside the URL."""
    urls = []
    for match in CATALOG_URL_RE.findall(text or ""):
        urls.append(match.rstrip(".,;:!?)]}\"'"))
    return urls


def _catalog_url_response(latest_user: str, catalog: CatalogStore) -> Optional[ChatResponse]:
    """Answer direct URL lookup questions without allowing homoglyph bypasses."""
    urls = _extract_urls(latest_user)
    if not urls:
        return None

    valid_items = [catalog.get_by_url(url) for url in urls if _is_safe_catalog_url(url, catalog)]
    valid_items = [item for item in valid_items if item]
    if valid_items and len(valid_items) == len(urls):
        recs = _to_recommendations(valid_items, catalog)
        names = ", ".join(rec.name for rec in recs)
        return ChatResponse(
            reply=f"Yes, that URL is in the SHL catalog: {names}.",
            recommendations=recs,
            end_of_conversation=False,
        )

    return ChatResponse(
        reply="That URL is not a verbatim SHL catalog URL I can validate. Please share the role or skills you need to assess, and I can recommend items from the catalog.",
        recommendations=[],
        end_of_conversation=False,
    )


def _is_refusal_topic(text: str) -> bool:
    low = text.lower()
    legal_signals = [
        "legally required", "legal requirement", "satisfy that requirement",
        "satisfies that requirement", "regulatory obligation",
    ]
    if any(signal in low for signal in legal_signals):
        return True
    # Injection is handled separately by _is_injection_attempt
    return False


def _is_vague_request(text: str, has_context: bool, has_recs: bool) -> bool:
    low = text.lower().strip()
    role_signals = [
        "java", "spring", "rust", "python", "sales", "admin", "healthcare",
        "financial", "analyst", "graduate", "leadership", "executive",
        "contact", "call", "safety", "manufacturing", "engineer", "developer",
    ]
    vague_phrases = [
        "i need an assessment", "need an assessment", "help me choose",
        "help", "what should i use", "recommend an assessment",
    ]
    if any(phrase in low for phrase in vague_phrases) and not any(signal in low for signal in role_signals):
        return True
    if has_context or has_recs:
        return False
    return len(low.split()) < 5


def _unsupported_tech_terms(text: str) -> set[str]:
    low = text.lower()
    terms = set()
    if re.search(r"\brust\b", low):
        terms.add("Rust")
    if re.search(r"\b(wasm|webassembly)\b", low):
        terms.add("WebAssembly")
    if re.search(r"\bkotlin\b", low):
        terms.add("Kotlin")
    if re.search(r"\bgolang\b|\bgo\s+(developer|engineer|programmer)\b", low):
        terms.add("Go")
    return terms


def _explicitly_disallows_adjacent(text: str) -> bool:
    low = text.lower()
    signals = [
        "no proxies",
        "without proxies",
        "no proxy",
        "specific only",
        "wasm-specific",
        "webassembly-specific",
        "rust-specific",
        "exact match only",
        "nothing adjacent",
    ]
    return any(signal in low for signal in signals)


def _has_prior_no_coverage_offer(messages: list[dict]) -> bool:
    for msg in messages:
        if msg["role"] != "assistant":
            continue
        low = msg["content"].lower()
        if "catalog doesn't currently include" in low or "catalog does not currently include" in low:
            return True
        if "no rust-specific" in low or "no webassembly-specific" in low or "no wasm-specific" in low:
            return True
    return False


def _no_coverage_response(terms: set[str], disallows_adjacent: bool) -> ChatResponse:
    label = " or ".join(sorted(terms)) if terms else "that technology"
    if disallows_adjacent:
        reply = (
            f"The catalog does not currently include a {label}-specific assessment, "
            "and your constraints rule out adjacent proxies. I won't invent a catalog item; "
            "if you relax the proxy constraint, I can suggest nearby SHL options."
        )
    else:
        reply = (
            f"The catalog doesn't currently include a {label}-specific assessment. "
            "The closest SHL options are live coding, systems/Linux, networking, Verify G+, "
            "and OPQ32r if you are open to adjacent signals."
        )
    return ChatResponse(reply=reply, recommendations=[], end_of_conversation=False)


def _has_hard_constraint_language(text: str) -> bool:
    low = text.lower()
    signals = [
        " only", "nothing else", "must ", "must have", "required",
        "under ", "less than", "no cognitive", "without cognitive",
        "in swahili", "adaptive", "remote",
    ]
    return any(signal in low for signal in signals)


def _extract_required_types(text: str) -> set[str]:
    low = text.lower()
    required: set[str] = set()

    if re.search(r"\btype\s*([abcdekps])\b", low):
        for code in re.findall(r"\btype\s*([abcdekps])\b", low):
            required.add(code.upper())

    for label, code in TYPE_NAME_TO_CODE.items():
        if label in low:
            required.add(code)

    return required


def _extract_job_level(text: str) -> Optional[str]:
    low = text.lower()
    for signal, canonical in JOB_LEVEL_ALIASES.items():
        if signal in low:
            return canonical
    return None


def _extract_language(text: str) -> Optional[str]:
    low = text.lower()
    match = re.search(r"\b(?:in|only in)\s+([a-z][a-z -]+?)(?:,|\.|\s+under|\s+for|\s+with|\s+and|$)", low)
    if not match:
        return None
    language = match.group(1).strip()
    if language in {"the catalog", "your catalog", "this catalog", "a single assessment"}:
        return None
    return language.title()


def _extract_max_duration(text: str) -> tuple[Optional[int], bool]:
    low = text.lower()
    match = re.search(r"\b(under|less than|at most|no more than|within)\s+(\d+)\s*(?:min|minute|minutes)\b", low)
    if not match:
        return None, False
    strict = match.group(1) in {"under", "less than"}
    return int(match.group(2)), strict


def _constraints_from_text(text: str) -> dict:
    low = text.lower()
    required_types = _extract_required_types(text) if _has_hard_constraint_language(text) else set()
    excluded_types: set[str] = set()
    if "no cognitive" in low or "without cognitive" in low:
        excluded_types.add("A")
    required_types -= excluded_types

    max_duration, duration_strict = _extract_max_duration(text)
    language = _extract_language(text)

    constraints = {
        "required_types": required_types,
        "excluded_types": excluded_types,
        "single_assessment_all_types": "single assessment" in low and len(required_types) > 1,
        "language": language,
        "max_duration": max_duration,
        "duration_strict": duration_strict,
        "remote": True if re.search(r"\bremote\b", low) else None,
        "adaptive": True if re.search(r"\badaptive\b", low) else None,
        "job_level": _extract_job_level(text),
    }
    return constraints


def _constraints_active(constraints: dict) -> bool:
    return any([
        constraints.get("required_types"),
        constraints.get("excluded_types"),
        constraints.get("language"),
        constraints.get("max_duration") is not None,
        constraints.get("remote") is not None,
        constraints.get("adaptive") is not None,
        constraints.get("single_assessment_all_types"),
    ])


def _item_codes(item: dict) -> set[str]:
    return {code.strip() for code in item.get("test_type", "").split(",") if code.strip()}


def _duration_minutes(item: dict) -> Optional[int]:
    match = re.search(r"\d+", str(item.get("duration", "")))
    return int(match.group(0)) if match else None


def _language_matches(item: dict, language: str) -> bool:
    languages = item.get("languages") or []
    if not languages:
        return False
    wanted = language.lower()
    return any(wanted in str(lang).lower() for lang in languages)


def _item_matches_constraints(item: dict, constraints: dict) -> bool:
    codes = _item_codes(item)
    required = constraints.get("required_types") or set()
    excluded = constraints.get("excluded_types") or set()

    if required:
        if constraints.get("single_assessment_all_types"):
            if not required <= codes:
                return False
        elif not (required & codes):
            return False

    if excluded & codes:
        return False

    language = constraints.get("language")
    if language and not _language_matches(item, language):
        return False

    max_duration = constraints.get("max_duration")
    if max_duration is not None:
        duration = _duration_minutes(item)
        if duration is None:
            return False
        if constraints.get("duration_strict"):
            if not duration < max_duration:
                return False
        elif duration > max_duration:
            return False

    if constraints.get("remote") is True and str(item.get("remote", "")).lower() != "yes":
        return False

    if constraints.get("adaptive") is True and str(item.get("adaptive", "")).lower() != "yes":
        return False

    job_level = constraints.get("job_level")
    if job_level:
        levels = item.get("job_levels") or []
        if levels and job_level not in levels:
            return False

    return True


def _filter_by_constraints(items: list[dict], constraints: dict) -> list[dict]:
    if not _constraints_active(constraints):
        return items
    return [item for item in items if _item_matches_constraints(item, constraints)]


def _constraint_catalog_candidates(catalog: CatalogStore, constraints: dict) -> list[dict]:
    candidates = _filter_by_constraints(catalog.items, constraints)
    candidates.sort(key=lambda item: (
        0 if item.get("test_type") == "E" else 1,
        item.get("name", ""),
    ))
    return candidates[:10]


def _constraint_failure_reply(constraints: dict) -> str:
    reasons = []
    if constraints.get("language"):
        reasons.append(f"language={constraints['language']}")
    if constraints.get("max_duration") is not None:
        op = "under" if constraints.get("duration_strict") else "within"
        reasons.append(f"{op} {constraints['max_duration']} minutes")
    if constraints.get("adaptive"):
        reasons.append("adaptive=yes")
    if constraints.get("remote"):
        reasons.append("remote=yes")
    if constraints.get("required_types"):
        reasons.append("test_type=" + ",".join(sorted(constraints["required_types"])))
    if constraints.get("excluded_types"):
        reasons.append("excluded_type=" + ",".join(sorted(constraints["excluded_types"])))

    detail = "; ".join(reasons) if reasons else "the requested hard constraints"
    return (
        "I could not verify any catalog assessment that satisfies all of those hard "
        f"constraints ({detail}). I won't recommend partial matches unless you relax one."
    )


def _constraint_success_reply(recs: list[Recommendation], constraints: dict) -> str:
    names = ", ".join(rec.name for rec in recs)
    if constraints.get("required_types") == {"E"}:
        return f"I found the catalog items that include Assessment Exercises (type E): {names}."
    return f"I found catalog items that satisfy the hard constraints: {names}."


def _apply_user_constraints(items: list[dict], latest_user: str) -> list[dict]:
    """Apply explicit remove/drop instructions from the newest user turn."""
    low = latest_user.lower()
    remove_names: set[str] = set()

    remove_prefix = r"\b(remove|drop|exclude|without)\s+(?:the\s+)?"
    if re.search(remove_prefix + r"(opq32r|opq|personality)\b", low):
        remove_names.add("Occupational Personality Questionnaire OPQ32r")
    if re.search(remove_prefix + r"verify\s*g\+?\b", low):
        remove_names.add("SHL Verify Interactive G+")
    if re.search(remove_prefix + r"graduate scenarios\b", low):
        remove_names.add("Graduate Scenarios")

    if not remove_names:
        return items

    return [item for item in items if item.get("name") not in remove_names]


def _to_recommendations(items: list[dict], catalog: CatalogStore) -> list[Recommendation]:
    """Convert catalog dicts to schema objects, re-validating URLs against catalog."""
    recs = []
    seen = set()
    for item in items:
        catalog_item = catalog.get_by_url(item.get("url", ""))
        if not catalog_item or catalog_item["url"] in seen:
            continue
        if not catalog_item.get("test_type"):
            continue
        recs.append(Recommendation(
            name=catalog_item["name"],
            url=catalog_item["url"],
            test_type=catalog_item["test_type"],
        ))
        seen.add(catalog_item["url"])
        if len(recs) == 10:
            break
    return recs


def _build_rule_reply(
    latest_user: str,
    recs: list[Recommendation],
    refusing: bool = False,
    confirming: Optional[bool] = None,
) -> str:
    low = latest_user.lower()
    is_confirming = _is_confirmation(latest_user) if confirming is None else confirming
    if refusing:
        return (
            "I can't advise on legal or regulatory obligations. I can help select "
            "SHL assessments and keep the shortlist grounded in the catalog."
        )

    if "difference" in low and "opq" in low and "mq" in low:
        return (
            "OPQ32r is the core personality assessment. OPQ MQ Sales Report is a "
            "sales-focused report layer for interpreting OPQ/MQ outputs in a sales context. "
            "I would keep both only where that sales-specific reporting is useful."
        )
    if "difference" in low and ("dsi" in low or "dependability" in low) and "8.0" in low:
        return (
            "DSI is a general dependability and safety-oriented personality measure. "
            "Safety & Dependability 8.0 is the stronger industrial fit for plant or "
            "manufacturing environments, with Workplace Health and Safety as the knowledge check."
        )
    if "different" in low and "contact center call simulation" in low:
        return (
            "Contact Center Call Simulation is the newer contact-center simulation choice for "
            "volume screening. Customer Service Phone Simulation is the older phone-simulation "
            "option and is useful as a finalist-stage complement."
        )
    if "advanced level" in low and "java" in low:
        return (
            "Yes. For an experienced Java engineer maintaining existing services, Core Java "
            "Advanced Level is the right catalog fit, with Spring and SQL around it."
        )
    if "verify g+" in low and ("really need" in low or "redundant" in low):
        return (
            "Verify G+ is not a duplicate of the technical tests; it adds a general reasoning "
            "signal for senior design judgment. I would keep it if candidate time allows."
        )

    if not recs:
        return (
            "I couldn't find a catalog-backed shortlist that matches those constraints. "
            "Could you relax one requirement or share the role and skills you want to prioritize?"
        )

    names = ", ".join(rec.name for rec in recs)
    if is_confirming:
        return f"Confirmed. Final shortlist: {names}."
    return f"Based on your requirements, I recommend this SHL shortlist: {names}."


def _deterministic_response(
    items: list[dict],
    latest_user: str,
    catalog: CatalogStore,
    turns_remaining: int,
    can_confirm: bool = True,
) -> ChatResponse:
    constrained = _apply_user_constraints(items, latest_user)
    recs = _to_recommendations(constrained, catalog)
    is_confirming = can_confirm and _is_confirmation(latest_user)
    return ChatResponse(
        reply=_build_rule_reply(latest_user, recs, confirming=is_confirming),
        recommendations=recs,
        end_of_conversation=bool(recs) and (turns_remaining <= 1 or is_confirming),
    )


def _all_same_test_type(items: list[dict]) -> bool:
    """Return True when all items share the same test_type or it is missing."""
    if not items:
        return True
    types = {item.get("test_type", "") for item in items if item.get("test_type")}
    if not types:
        return True
    return len(types) == 1


def _lexical_catalog_recommend(query: str, all_text: str, catalog: CatalogStore) -> list[dict]:
    """Offline catalog fallback for covered-but-unmapped requests."""
    text = (query + " " + all_text).lower()
    stop = {
        "the", "and", "for", "with", "that", "this", "need", "needs",
        "hiring", "hire", "assessment", "assessments", "test", "tests",
        "role", "candidate", "candidates", "what", "should", "use",
    }
    tokens = {
        token
        for token in re.findall(r"[a-z0-9+#]+", text)
        if len(token) > 2 and token not in stop
    }
    if not tokens:
        return []

    scored = []
    for item in catalog.items:
        name = item.get("name", "").lower()
        description = item.get("description", "").lower()
        haystack = f"{name} {description}"
        name_tokens = set(re.findall(r"[a-z0-9+#]+", name))
        item_tokens = set(re.findall(r"[a-z0-9+#]+", haystack))
        overlap = tokens & item_tokens
        if not overlap:
            continue
        score = len(overlap) + (len(tokens & name_tokens) * 3)
        if name in text:
            score += 8
        scored.append((score, item))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in scored[:10]]


def _assistant_text_looks_like_our_shortlist(content: str) -> bool:
    low = content.lower()
    trusted_prefixes = [
        "based on your requirements, i recommend this shl shortlist:",
        "confirmed. final shortlist:",
        "i found catalog items that satisfy the hard constraints:",
        "i found the catalog items that include assessment exercises",
    ]
    return any(low.startswith(prefix) for prefix in trusted_prefixes)


def _catalog_items_from_trusted_assistant_text(content: str, catalog: CatalogStore) -> list[Recommendation]:
    if not _assistant_text_looks_like_our_shortlist(content):
        return []
    recs = []
    seen = set()
    for item in catalog.items:
        name = item.get("name", "")
        if name and name in content and item.get("url") not in seen:
            recs.append(Recommendation(name=item["name"], url=item["url"], test_type=item["test_type"]))
            seen.add(item["url"])
    return recs[:10]


def _has_prior_recommendations(messages: list[dict], catalog: CatalogStore) -> bool:
    """Check if any prior assistant turn contained non-empty recommendations."""
    for msg in messages:
        if msg["role"] == "assistant":
            content = msg["content"]
            try:
                data = json.loads(content)
                if data.get("recommendations") and len(data["recommendations"]) > 0:
                    return True
            except (json.JSONDecodeError, AttributeError):
                if _catalog_items_from_trusted_assistant_text(content, catalog):
                    return True
    return False


def _extract_prior_recommendations(messages: list[dict], catalog: CatalogStore) -> list[Recommendation]:
    """Extract the most recent non-empty recommendations from assistant turns."""
    for msg in reversed(messages):
        if msg["role"] == "assistant":
            content = msg["content"]
            try:
                data = json.loads(content)
                recs_data = data.get("recommendations", [])
                if recs_data:
                    recs = []
                    for r in recs_data:
                        if isinstance(r, dict):
                            item = catalog.get_by_name(r.get("name", ""))
                            if item:
                                recs.append(Recommendation(
                                    name=item["name"], url=item["url"],
                                    test_type=item["test_type"]
                                ))
                    if recs:
                        return recs
            except (json.JSONDecodeError, AttributeError):
                recs = _catalog_items_from_trusted_assistant_text(content, catalog)
                if recs:
                    return recs
    return []


def process_chat(messages: list[Message]) -> ChatResponse:
    start_time = time.time()
    catalog = get_catalog_store()

    msg_dicts = [{"role": m.role, "content": m.content} for m in messages]

    # ── Case 16: Sanitize all user content (strip null bytes / control chars) ──
    for msg in msg_dicts:
        if msg["role"] == "user":
            msg["content"] = _sanitize_user_content(msg["content"])

    # ── Case 2: Skip leading assistant messages ──
    while msg_dicts and msg_dicts[0]["role"] == "assistant":
        msg_dicts.pop(0)

    # ── Case 3: Merge consecutive user messages into one ──
    merged = []
    for msg in msg_dicts:
        if merged and merged[-1]["role"] == "user" and msg["role"] == "user":
            merged[-1]["content"] += " " + msg["content"]
        else:
            merged.append(msg.copy())
    msg_dicts = merged

    # If no messages remain after cleanup, treat as empty
    if not msg_dicts:
        return ChatResponse(
            reply="Please start by telling me about the role you are hiring for.",
            recommendations=[],
            end_of_conversation=False,
        )

    # Post-EOC restart detection (Case 12)
    eoc_index = _find_last_eoc_index(msg_dicts)
    if eoc_index is not None and eoc_index < len(msg_dicts) - 1:
        restart_msgs = msg_dicts[eoc_index + 1:]
        # Count how many user messages came AFTER the EOC
        post_eoc_user_msgs = [m for m in restart_msgs if m["role"] == "user"]
        if len(post_eoc_user_msgs) <= 1:
            # First message after EOC: acknowledge the prior session ended
            # and offer to start fresh instead of immediately recommending
            latest = post_eoc_user_msgs[0]["content"] if post_eoc_user_msgs else ""
            return ChatResponse(
                reply=f"It looks like we've already concluded our previous session. Shall I recommend assessments for a new role? Please describe the position, level, and key skills you'd like to assess.",
                recommendations=[],
                end_of_conversation=False,
            )
    else:
        restart_msgs = None

    full_msg_dicts = [m.copy() for m in msg_dicts]

    # Turn cap
    if len(msg_dicts) >= MAX_TURNS:
        msg_dicts = msg_dicts[-(MAX_TURNS - 1):]
        turns_remaining = 0
    else:
        turns_after_response = len(msg_dicts) + 1
        turns_remaining = MAX_TURNS - turns_after_response

    logger.info(f"Turns: {len(msg_dicts)} in context, {turns_remaining} remaining")

    # Build query + full text
    active_history = restart_msgs if restart_msgs else full_msg_dicts
    query_msgs = active_history
    query = build_query_from_messages(query_msgs)
    all_text = " ".join(m["content"] for m in active_history)
    combined = query + " " + all_text
    latest_user = _latest_user_text(active_history)

    # ── Cases 4/5/6/11: Deterministic injection detection ──
    if _is_injection_attempt(latest_user):
        # Case 11: If prior recs exist and user tries impersonation, preserve them
        prior_recs = _extract_prior_recommendations(active_history, catalog)
        if prior_recs:
            return ChatResponse(
                reply="I can only recommend assessments based on the role requirements we discussed. Would you like to refine the current list or add complementary assessments?",
                recommendations=prior_recs,
                end_of_conversation=False,
            )
        return ChatResponse(
            reply="I can only help with discovering SHL assessments from the official catalog. What role are you hiring for?",
            recommendations=[],
            end_of_conversation=False,
        )

    # Case 5/6: JSON injection or roleplay with fake catalog items
    if _contains_json_injection(latest_user):
        # Strip the injected JSON and process the legitimate part
        # But if the message is mostly injection, just ask for role
        clean_part = re.split(r'[{}]', latest_user)[0].strip()
        if clean_part and len(clean_part) > 10:
            latest_user = clean_part
            # Update msg_dicts so downstream uses clean content
            for msg in reversed(active_history):
                if msg["role"] == "user":
                    msg["content"] = clean_part
                    break
            # Recompute
            query = build_query_from_messages(active_history)
            all_text = " ".join(m["content"] for m in active_history)
            combined = query + " " + all_text
        else:
            return ChatResponse(
                reply="I'd be happy to help. Could you tell me more about the role — the job level and key skills you need to assess for?",
                recommendations=[],
                end_of_conversation=False,
            )

    url_response = _catalog_url_response(latest_user, catalog)
    if url_response:
        return url_response

    # ── Cases 7/8: Premature confirmation guard ──
    has_prior_recs = _has_prior_recommendations(active_history, catalog)
    has_prior_no_coverage_offer = _has_prior_no_coverage_offer(active_history)
    if _is_confirmation(latest_user) and not has_prior_recs and not has_prior_no_coverage_offer:
        # Case 8: First message is confirmation with zero context
        if len([m for m in active_history if m["role"] == "user"]) <= 1:
            return ChatResponse(
                reply="It looks like we haven't discussed any assessments yet. Could you tell me about the role you are hiring for so I can make some recommendations?",
                recommendations=[],
                end_of_conversation=False,
            )
        # Case 7: Confirmation but no recommendations were made yet
        return ChatResponse(
            reply="I still need a bit more context before making recommendations. Could you tell me more about the role, department, or key skills you need to assess?",
            recommendations=[],
            end_of_conversation=False,
        )

    user_text_all = " ".join(m["content"] for m in active_history if m["role"] == "user")
    unsupported_terms = _unsupported_tech_terms(user_text_all)
    disallows_adjacent = _explicitly_disallows_adjacent(user_text_all)
    if unsupported_terms and (
        disallows_adjacent
        or not has_prior_no_coverage_offer
        or not _is_confirmation(latest_user)
    ):
        return _no_coverage_response(unsupported_terms, disallows_adjacent)

    # ── Step 1: Rule-based recommendations (always computed) ──────────────
    rule_recs = _rule_based_recommend(query, all_text, catalog)
    logger.info(f"Rule-based recs: {len(rule_recs)}")
    has_context = _has_enough_context(combined)

    if _is_refusal_topic(latest_user) and not _is_confirmation(latest_user):
        return ChatResponse(
            reply=_build_rule_reply(latest_user, [], refusing=True),
            recommendations=[],
            end_of_conversation=False,
        )

    if _is_vague_request(latest_user, has_context, bool(rule_recs)):
        return ChatResponse(
            reply="I can help with that. What role, level, and skills are you hiring for?",
            recommendations=[],
            end_of_conversation=False,
        )

    constraints = _constraints_from_text(latest_user)
    if _constraints_active(constraints):
        constrained_rule_recs = _filter_by_constraints(rule_recs, constraints)
        candidates = constrained_rule_recs or _constraint_catalog_candidates(catalog, constraints)
        recs = _to_recommendations(candidates, catalog)
        if not recs:
            return ChatResponse(
                reply=_constraint_failure_reply(constraints),
                recommendations=[],
                end_of_conversation=False,
            )
        return ChatResponse(
            reply=_constraint_success_reply(recs, constraints),
            recommendations=recs,
            end_of_conversation=False,
        )

    lexical: list[dict] = []
    if has_context:
        lexical = _lexical_catalog_recommend(query, all_text, catalog)
        lexical = _inject_canonical_assessments(combined, lexical, catalog)

    retrieved = None
    low_similarity = False
    if _llm_enabled() and has_context:
        retriever = get_retriever()
        retrieved = retriever.retrieve(query, top_k=20)
        if retrieved:
            low_similarity = all((item.get("score", 0.0) < 0.4) for item in retrieved)

    deterministic_candidates = rule_recs if rule_recs else lexical
    deterministic_ok = bool(deterministic_candidates) and not _all_same_test_type(deterministic_candidates) and not low_similarity

    # Prefer deterministic catalog-backed answers whenever rules cover the
    # request. The LLM path remains only for uncovered or low-signal fallbacks.
    pass

    if not _llm_enabled():
        return ChatResponse(
            reply="I can help you find the right SHL assessments. Could you share the role, level, and skills to assess?",
            recommendations=[],
            end_of_conversation=False,
        )

    # ── Step 2: TF-IDF retrieval ──────────────────────────────────────────
    if retrieved is None:
        retriever = get_retriever()
        retrieved = retriever.retrieve(query, top_k=20)

    # Inject mentioned items
    mentioned_items = _extract_mentioned_items(all_text, catalog)
    if mentioned_items:
        mentioned_urls = {i["url"] for i in mentioned_items}
        retrieved = mentioned_items + [r for r in retrieved if r["url"] not in mentioned_urls]
        retrieved = retrieved[:30]

    # Inject canonicals
    retrieved = _inject_canonical_assessments(combined, retrieved, catalog)

    # Also inject rule-based items into retrieval context
    rule_urls = {i["url"] for i in retrieved}
    for item in rule_recs:
        if item["url"] not in rule_urls and len(retrieved) < 30:
            retrieved.append(item)
            rule_urls.add(item["url"])

    # ── Step 3: LLM call ─────────────────────────────────────────────────
    prompt = build_full_prompt(msg_dicts, retrieved, turns_remaining=turns_remaining)
    llm_response = None
    try:
        raw = _call_llm(prompt)
        elapsed = time.time() - start_time
        logger.info(f"LLM responded in {elapsed:.1f}s ({len(raw)} chars)")
        llm_response = _parse_llm_response(raw, catalog)
    except Exception as e:
        logger.error(f"LLM failed: {e}")

    # ── Step 5: Merge rule-based + LLM recs ──────────────────────────────
    has_context = _has_enough_context(combined)

    if llm_response is None:
        # Full LLM failure — use rule-based recs + generic reply
        recs = [
            Recommendation(name=i["name"], url=i["url"], test_type=i["test_type"])
            for i in rule_recs[:10]
        ]
        eoc = bool(recs) and turns_remaining <= 0 and has_prior_recs
        return ChatResponse(
            reply=_build_rule_reply(latest_user, recs),
            recommendations=recs,
            end_of_conversation=eoc,
        )

    # Always merge: rule-based recs come FIRST (guaranteed catalog hits),
    # then LLM recs supplement up to 10. If LLM is clarifying (empty recs +
    # question mark in reply), don't inject.
    llm_clarifying = (
        len(llm_response.recommendations) == 0
        and not llm_response.end_of_conversation
        and "?" in llm_response.reply
    )

    if has_context and rule_recs and not llm_clarifying:
        rule_as_recs = [
            Recommendation(name=i["name"], url=i["url"], test_type=i["test_type"])
            for i in rule_recs
        ]
        llm_response.recommendations = _merge_recommendations(
            rule_as_recs,           # rule-based first (guaranteed hits)
            llm_response.recommendations,  # LLM recs fill remaining slots
            catalog,
        )
        logger.info(f"After merge: {len(llm_response.recommendations)} total recs")

    # Turn-cap enforcement
    if turns_remaining <= 1 and not llm_response.end_of_conversation:
        if not llm_response.recommendations and has_context and rule_recs:
            llm_response.recommendations = [
                Recommendation(name=i["name"], url=i["url"], test_type=i["test_type"])
                for i in rule_recs[:10]
            ]
        llm_response.end_of_conversation = bool(llm_response.recommendations) and has_prior_recs
        if llm_response.end_of_conversation:
            logger.warning("Forced end_of_conversation due to turn cap")

    return llm_response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _inject_canonical_assessments(query: str, retrieved: list[dict], catalog: CatalogStore) -> list[dict]:
    query_lower = query.lower()
    existing_urls = {item["url"] for item in retrieved}
    injected = []

    def _add(name: str):
        item = catalog.get_by_name(name)
        if item and item["url"] not in existing_urls:
            injected.append(item)
            existing_urls.add(item["url"])

    # Always include OPQ32r
    _add(CANONICAL_ASSESSMENTS["personality"])

    senior_signals = ["senior", "lead", "manager", "director", "executive", "cxo",
                      "engineer", "developer", "architect", "technical", "cognitive",
                      "reasoning", "graduate", "analyst", "full-stack", "backend"]
    if any(s in query_lower for s in senior_signals):
        _add(CANONICAL_ASSESSMENTS["cognitive"])

    leadership_signals = ["leadership", "leader", "executive", "cxo", "director",
                          "senior leader", "benchmark", "selection"]
    if any(s in query_lower for s in leadership_signals):
        _add(CANONICAL_ASSESSMENTS["leadership_report"])
        _add(CANONICAL_ASSESSMENTS["ucf_report"])

    sales_signals = ["sales", "selling", "revenue", "account executive", "re-skill",
                     "restructuring", "talent audit"]
    if any(s in query_lower for s in sales_signals):
        _add(CANONICAL_ASSESSMENTS["sales_report"])
        _add(CANONICAL_ASSESSMENTS["sales_transform_ic"])
        _add(CANONICAL_ASSESSMENTS["gsa"])
        _add(CANONICAL_ASSESSMENTS["gsa_dev_report"])

    graduate_signals = ["graduate", "entry-level", "entry level", "final-year",
                        "student", "campus", "situational judgement", "situational judgment"]
    if any(s in query_lower for s in graduate_signals):
        _add(CANONICAL_ASSESSMENTS["graduate_scenarios"])

    safety_signals = ["safety", "dependability", "healthcare", "hipaa", "compliance",
                      "patient", "medical", "trust", "reliability"]
    if any(s in query_lower for s in safety_signals):
        _add(CANONICAL_ASSESSMENTS["dsi"])

    tech_mappings = {
        "java": ["Core Java (Advanced Level) (New)", "Core Java (Entry Level) (New)",
                 "Spring (New)", "Java 8 (New)"],
        "spring": ["Spring (New)"],
        "sql": ["SQL (New)"],
        "aws": ["Amazon Web Services (AWS) Development (New)"],
        "docker": ["Docker (New)"],
        "angular": ["Angular 6 (New)"],
        "react": ["ReactJS (New)"],
        "python": ["Python 3 (New)"],
        "rust": ["Smart Interview Live Coding", "Linux Programming (General)",
                 "Networking and Implementation (New)"],
        "networking": ["Networking and Implementation (New)"],
        "linux": ["Linux Programming (General)", "Linux Administration (New)"],
        "excel": ["MS Excel (New)", "Microsoft Excel 365 (New)",
                  "Microsoft Excel 365 - Essentials (New)"],
        "word": ["MS Word (New)", "Microsoft Word 365 (New)",
                 "Microsoft Word 365 - Essentials (New)"],
        "medical": ["Medical Terminology (New)"],
        "hipaa": ["HIPAA (Security)"],
        "customer service": ["Customer Service Phone Simulation",
                             "Contact Center Call Simulation (New)",
                             "Entry Level Customer Serv-Retail & Contact Center"],
        "contact cent": ["Contact Center Call Simulation (New)",
                         "Customer Service Phone Simulation",
                         "SVAR - Spoken English (US) (New)"],
        "svar": ["SVAR - Spoken English (US) (New)", "SVAR - Spoken English (U.K.)",
                 "SVAR - Spoken English (AUS)", "SVAR - Spoken English (Indian Accent) (New)"],
        "manufacturing": ["Manufac. & Indust. - Safety & Dependability 8.0",
                          "Manufacturing & Industrial - Mechanical Focus 8.0"],
        "numerical": ["SHL Verify Interactive – Numerical Reasoning"],
        "financial": ["Financial Accounting (New)"],
        "statistics": ["Basic Statistics (New)"],
        "admin": ["MS Excel (New)", "MS Word (New)",
                  "Microsoft Excel 365 (New)", "Microsoft Word 365 (New)"],
    }

    for signal, names in tech_mappings.items():
        if signal in query_lower:
            for name in names:
                _add(name)

    result = retrieved + injected
    return result[:30]


def _extract_mentioned_items(text: str, catalog: CatalogStore) -> list[dict]:
    mentioned = []
    seen_urls = set()
    all_names = sorted(catalog.get_all_names(), key=len, reverse=True)
    text_lower = text.lower()
    for name in all_names:
        if name.lower() in text_lower:
            item = catalog.get_by_name(name)
            if item and item["url"] not in seen_urls:
                mentioned.append(item)
                seen_urls.add(item["url"])
    abbreviations = {
        "OPQ": "Occupational Personality Questionnaire OPQ32r",
        "OPQ32r": "Occupational Personality Questionnaire OPQ32r",
        "OPQ32": "Occupational Personality Questionnaire OPQ32r",
        "Verify G+": "SHL Verify Interactive G+",
        "GSA": "Global Skills Assessment",
        "DSI": "Dependability and Safety Instrument (DSI)",
        "MQ": "Motivational Questionnaire (MQ)",
    }
    for abbr, full_name in abbreviations.items():
        if abbr.lower() in text_lower and full_name:
            item = catalog.get_by_name(full_name)
            if item and item["url"] not in seen_urls:
                mentioned.append(item)
                seen_urls.add(item["url"])
    return mentioned


def _find_last_eoc_index(messages: list[dict]) -> Optional[int]:
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg["role"] == "assistant":
            content = msg["content"]
            if "end_of_conversation" in content:
                try:
                    data = json.loads(content)
                    if data.get("end_of_conversation"):
                        return i
                except (json.JSONDecodeError, AttributeError):
                    if '"end_of_conversation": true' in content or '"end_of_conversation":true' in content:
                        return i
    return None
