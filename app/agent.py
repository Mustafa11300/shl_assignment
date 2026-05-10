"""
Core agent controller for the SHL Assessment Recommender.

Orchestrates: query analysis → retrieval → LLM generation → response validation.
Handles all conversational behaviors: clarify, recommend, refine, compare, refuse.
Enforces turn cap (max 8) and 30-second timeout.
"""

import json
import os
import re
import logging
import time
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from app.catalog import CatalogStore
from app.retriever import CatalogRetriever
from app.prompts import build_full_prompt, build_query_from_messages
from app.models import ChatResponse, Recommendation, Message

logger = logging.getLogger(__name__)

# Global singletons (initialized once at startup)
_catalog_store: Optional[CatalogStore] = None
_retriever: Optional[CatalogRetriever] = None

# Maximum total turns (user + assistant messages combined)
MAX_TURNS = 8


def get_catalog_store() -> CatalogStore:
    """Get or initialize the global CatalogStore singleton."""
    global _catalog_store
    if _catalog_store is None:
        _catalog_store = CatalogStore()
        logger.info(f"Loaded catalog with {len(_catalog_store)} items")
    return _catalog_store


def get_retriever() -> CatalogRetriever:
    """Get or initialize the global CatalogRetriever singleton."""
    global _retriever
    if _retriever is None:
        _retriever = CatalogRetriever(get_catalog_store())
        logger.info("FAISS retriever initialized")
    return _retriever


def _call_gemini(prompt_messages: list[dict]) -> str:
    """Call Gemini API with the constructed prompt.
    
    Uses exponential backoff for rate limit errors.
    Model is configurable via GEMINI_MODEL env var (default: gemini-2.5-flash).
    Disables thinking mode for faster responses (stay under 30s timeout).
    
    Args:
        prompt_messages: List of message dicts with 'role' and 'content'.
        
    Returns:
        Raw text response from the LLM.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")
    
    from google import genai
    from google.genai import types
    
    client = genai.Client(api_key=api_key)
    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    
    # Build the prompt — Gemini uses a different format
    system_instruction = prompt_messages[0]["content"] if prompt_messages[0]["role"] == "system" else ""
    user_content = prompt_messages[1]["content"] if len(prompt_messages) > 1 else ""
    
    # Build config — disable thinking for speed (stay under 30s)
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.3,
        max_output_tokens=4096,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )
    
    # Retry with backoff for rate limits
    max_retries = 2  # Fewer retries to stay under 30s
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=user_content,
                config=config,
            )
            return response.text
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                wait_time = (2 ** attempt) * 3  # 3s, 6s
                logger.warning(f"Rate limited (attempt {attempt+1}/{max_retries}), waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
    
    raise RuntimeError(f"Failed after {max_retries} retries due to rate limiting")


def _parse_llm_response(raw_response: str, catalog: CatalogStore) -> ChatResponse:
    """Parse and validate the LLM's JSON response.
    
    Handles common LLM output issues:
    - Markdown code blocks around JSON
    - Missing fields with sensible defaults
    - Invalid URLs (removes them)
    - Invalid test_type codes (fixes them)
    
    Args:
        raw_response: Raw text from the LLM.
        catalog: CatalogStore for URL validation.
        
    Returns:
        Validated ChatResponse.
    """
    # Strip markdown code blocks if present
    text = raw_response.strip()
    if text.startswith("```"):
        # Remove opening ```json or ```
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        # Remove closing ```
        text = re.sub(r"\n?```\s*$", "", text)
    
    # Try to extract JSON from response if it has extra text around it
    if not text.startswith("{"):
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            text = json_match.group(0)
    
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        logger.error(f"Raw response: {text[:500]}")
        # Fallback response
        return ChatResponse(
            reply="I can help you find the right SHL assessments. Could you tell me more about the role you're hiring for?",
            recommendations=[],
            end_of_conversation=False,
        )
    
    # Validate and clean recommendations
    valid_recs = []
    for rec in data.get("recommendations", []) or []:
        if not isinstance(rec, dict):
            continue
        
        name = rec.get("name", "")
        url = rec.get("url", "")
        
        # Validate URL exists in catalog
        if url and catalog.url_exists(url):
            # Use catalog data as ground truth for name and test_type
            catalog_item = catalog.get_by_url(url)
            if catalog_item:
                valid_recs.append(Recommendation(
                    name=catalog_item["name"],
                    url=catalog_item["url"],
                    test_type=catalog_item["test_type"],
                ))
        elif name:
            # Try to find by name if URL doesn't match
            catalog_item = catalog.get_by_name(name)
            if catalog_item:
                valid_recs.append(Recommendation(
                    name=catalog_item["name"],
                    url=catalog_item["url"],
                    test_type=catalog_item["test_type"],
                ))
            else:
                # Try substring match
                matches = catalog.search_by_name(name)
                if matches:
                    best = matches[0]
                    valid_recs.append(Recommendation(
                        name=best["name"],
                        url=best["url"],
                        test_type=best["test_type"],
                    ))
    
    # Deduplicate by URL
    seen_urls = set()
    deduped_recs = []
    for rec in valid_recs:
        if rec.url not in seen_urls:
            seen_urls.add(rec.url)
            deduped_recs.append(rec)
    
    # Cap at 10
    deduped_recs = deduped_recs[:10]
    
    return ChatResponse(
        reply=data.get("reply", "I can help you find the right SHL assessments. Could you tell me more about the role?"),
        recommendations=deduped_recs,
        end_of_conversation=data.get("end_of_conversation", False),
    )


def process_chat(messages: list[Message]) -> ChatResponse:
    """Process a chat request and return the agent's response.
    
    This is the main entry point for each /chat call. It:
    1. Counts turns and calculates remaining budget
    2. Extracts a search query from the conversation history
    3. Retrieves relevant catalog items via FAISS
    4. Builds a turn-aware prompt with retrieved context
    5. Calls the LLM
    6. Validates and returns the response
    
    Args:
        messages: Full conversation history as Message objects.
        
    Returns:
        Validated ChatResponse with reply, recommendations, and end_of_conversation.
    """
    start_time = time.time()
    catalog = get_catalog_store()
    retriever = get_retriever()
    
    # Convert messages to dicts
    msg_dicts = [{"role": m.role, "content": m.content} for m in messages]
    
    # Step 0: Calculate turn budget
    # Total messages so far + 1 (this response) = total turns used
    current_turn_count = len(msg_dicts)  # messages received
    turns_after_response = current_turn_count + 1  # after we respond
    turns_remaining = MAX_TURNS - turns_after_response
    logger.info(f"Turn budget: {current_turn_count} messages received, {turns_remaining} turns remaining after response")
    
    # Step 1: Extract search query from conversation
    query = build_query_from_messages(msg_dicts)
    logger.info(f"Search query: {query[:200]}")
    
    # Step 2: Retrieve relevant catalog items
    retrieved = retriever.retrieve(query, top_k=20)
    logger.info(f"Retrieved {len(retrieved)} catalog items")
    
    # Step 3: Also retrieve items mentioned by name in the conversation
    # This helps with comparison and refinement requests
    all_text = " ".join(m["content"] for m in msg_dicts)
    mentioned_items = _extract_mentioned_items(all_text, catalog)
    
    # Merge mentioned items into retrieved (at the top)
    if mentioned_items:
        mentioned_urls = {item["url"] for item in mentioned_items}
        retrieved = mentioned_items + [r for r in retrieved if r["url"] not in mentioned_urls]
        retrieved = retrieved[:25]  # Controlled context size
    
    # Step 4: Build prompt and call LLM
    prompt = build_full_prompt(msg_dicts, retrieved, turns_remaining=turns_remaining)
    
    try:
        raw_response = _call_gemini(prompt)
        elapsed = time.time() - start_time
        logger.info(f"LLM response in {elapsed:.1f}s, length: {len(raw_response)} chars")
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return ChatResponse(
            reply="I'm having trouble processing your request. Could you try again?",
            recommendations=[],
            end_of_conversation=False,
        )
    
    # Step 5: Parse and validate
    response = _parse_llm_response(raw_response, catalog)
    
    # Step 6: Enforce turn cap — if we're at the last turn and have no recs, 
    # force end_of_conversation
    if turns_remaining <= 1 and not response.end_of_conversation:
        response.end_of_conversation = True
        logger.warning("Forced end_of_conversation due to turn cap")
    
    return response


def _extract_mentioned_items(text: str, catalog: CatalogStore) -> list[dict]:
    """Extract catalog items mentioned by name in the conversation text.
    
    This ensures that when users mention specific assessments (for comparison
    or refinement), those items are always in the retrieval context.
    
    Args:
        text: Combined conversation text.
        catalog: CatalogStore for lookups.
        
    Returns:
        List of catalog items mentioned in the text.
    """
    mentioned = []
    seen_urls = set()
    
    # Check for known assessment names in the text
    # Use longer names first to avoid partial matches
    all_names = sorted(catalog.get_all_names(), key=len, reverse=True)
    text_lower = text.lower()
    
    for name in all_names:
        if name.lower() in text_lower:
            item = catalog.get_by_name(name)
            if item and item["url"] not in seen_urls:
                mentioned.append(item)
                seen_urls.add(item["url"])
    
    # Also check for common abbreviations
    abbreviations = {
        "OPQ": "Occupational Personality Questionnaire OPQ32r",
        "OPQ32r": "Occupational Personality Questionnaire OPQ32r",
        "OPQ32": "Occupational Personality Questionnaire OPQ32r",
        "Verify G+": "SHL Verify Interactive G+",
        "GSA": "Global Skills Assessment",
        "DSI": "Dependability and Safety Instrument (DSI)",
        "MQ": "Motivational Questionnaire (MQ)",
        "SVAR": None,  # Multiple SVAR variants — handled by retrieval
    }
    
    for abbr, full_name in abbreviations.items():
        if abbr.lower() in text_lower and full_name:
            item = catalog.get_by_name(full_name)
            if item and item["url"] not in seen_urls:
                mentioned.append(item)
                seen_urls.add(item["url"])
    
    return mentioned
