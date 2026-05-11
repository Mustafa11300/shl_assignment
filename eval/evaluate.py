"""
Proper evaluation harness for the SHL Assessment Recommender.

Implements the assignment's grading approach:
- Parse each trace to extract: user messages (in order) + expected final recommendations
- Replay the trace by feeding our agent user messages one by one,
  using the AGENT's actual replies (not trace replies) to build conversation history
- Collect the LAST non-empty recommendations the agent provided
- Score with Recall@10

This mirrors how the real LLM harness works: the user side is fixed (from persona),
but the agent side is fully dynamic.
"""

import json
import re
import os
import sys
import time
import requests
from typing import Optional
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Catalog URL validation ─────────────────────────────────────────────────────

def load_catalog_urls(dataset_path: str) -> set:
    """Load all valid URLs from the dataset."""
    urls = set()
    try:
        with open(dataset_path) as f:
            data = json.load(f)
        for item in data:
            if "url" in item:
                urls.add(item["url"].strip())
    except Exception:
        pass
    return urls


# ── Trace parsing ──────────────────────────────────────────────────────────────

def parse_trace(filepath: str) -> dict:
    """Parse a conversation trace Markdown file.
    
    Extracts:
    - user_messages: ordered list of user message strings (what the simulated user says)
    - expected_names: list of assessment names in the FINAL recommendation table
    - expected_urls: list of URLs in the FINAL recommendation table
    """
    with open(filepath) as f:
        content = f.read()

    user_messages = []
    final_names = []
    final_urls = []

    # Split into turn blocks
    turn_blocks = re.split(r"###\s+Turn\s+\d+", content)

    for block in turn_blocks[1:]:
        # Extract User message (text after "> " markers)
        user_match = re.search(r"\*\*User\*\*\s*\n(.*?)(?=\*\*Agent\*\*|\Z)", block, re.DOTALL)
        if user_match:
            raw = user_match.group(1)
            # Remove "> " block-quote markers and join lines
            lines = [re.sub(r"^\s*>\s?", "", line) for line in raw.splitlines()]
            user_text = " ".join(l.strip() for l in lines if l.strip())
            if user_text:
                user_messages.append(user_text)

        # Extract Agent section — look for recommendation table
        agent_match = re.search(r"\*\*Agent\*\*\s*\n(.*?)(?=###\s+Turn|\Z)", block, re.DOTALL)
        if agent_match:
            agent_text = agent_match.group(1)
            recs = _parse_table(agent_text)
            if recs:
                final_names = [r["name"] for r in recs]
                final_urls = [r["url"] for r in recs]

    return {
        "user_messages": user_messages,
        "expected_names": final_names,
        "expected_urls": final_urls,
    }


def _parse_table(text: str) -> list[dict]:
    """Extract recommendations from a Markdown table in agent text."""
    recs = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if len(cells) < 2:
            continue
        if cells[0] in ("#", "---", "") or "---" in cells[1]:
            continue
        try:
            num = cells[0].strip()
            if not num.isdigit():
                continue
            name = cells[1].strip()
            url_match = re.search(r"<(https://www\.shl\.com/[^>]+)>", line)
            if url_match and name:
                recs.append({"name": name, "url": url_match.group(1)})
        except (IndexError, ValueError):
            continue
    return recs


# ── Conversation replay ────────────────────────────────────────────────────────

VALID_TEST_TYPES = {"A", "B", "C", "D", "E", "K", "P", "S"}


def replay(base_url: str, trace: dict, verbose: bool = True) -> dict:
    """Replay a trace against the live agent.

    Feeds user messages one-by-one. After each user message, our agent
    responds. We use the agent's ACTUAL reply (not the trace's) to build
    conversation history — exactly like the real LLM harness.

    Collects last non-empty recommendations the agent produced.
    Also validates schema on every response.
    """
    messages = []
    last_recs_urls = []
    last_recs_names = []
    schema_errors = []
    test_type_errors = []
    turns_taken = 0
    ended_early = False

    for i, user_msg in enumerate(trace["user_messages"]):
        messages.append({"role": "user", "content": user_msg})

        # 5s delay before every call = max 12 RPM, safely under Gemini's 15 RPM limit
        time.sleep(5)

        try:
            resp = requests.post(
                f"{base_url}/chat",
                json={"messages": messages},
                timeout=90,  # Allow time for retries inside the agent
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.Timeout:
            schema_errors.append(f"Turn {i+1}: TIMEOUT (>90s)")
            break
        except Exception as e:
            schema_errors.append(f"Turn {i+1}: HTTP error — {e}")
            break

        turns_taken += 1

        # ── Schema validation ──────────────────────────────────────────────
        if "reply" not in data:
            schema_errors.append(f"Turn {i+1}: missing 'reply'")
        if "recommendations" not in data:
            schema_errors.append(f"Turn {i+1}: missing 'recommendations'")
        if "end_of_conversation" not in data:
            schema_errors.append(f"Turn {i+1}: missing 'end_of_conversation'")

        recs = data.get("recommendations") or []

        # Validate rec count
        if len(recs) > 10:
            schema_errors.append(f"Turn {i+1}: {len(recs)} recommendations (max 10)")

        # Validate test_type codes
        for rec in recs:
            tt = rec.get("test_type", "")
            codes = {c.strip() for c in tt.split(",")}
            bad = codes - VALID_TEST_TYPES
            if bad:
                test_type_errors.append(f"Turn {i+1}: invalid test_type codes {bad} in '{rec.get('name')}'")

        # Track last non-empty recommendations
        if recs:
            last_recs_urls = [r.get("url", "") for r in recs]
            last_recs_names = [r.get("name", "") for r in recs]

        # Add agent reply to history (using actual reply, not trace)
        messages.append({
            "role": "assistant",
            "content": data.get("reply", ""),
        })

        if verbose:
            rec_count = len(recs)
            eoc = data.get("end_of_conversation", False)
            print(f"    Turn {i+1}: {rec_count} recs | EOC={eoc} | reply={data.get('reply','')[:60]}...")

        # Stop if agent ends the conversation
        if data.get("end_of_conversation", False):
            ended_early = (i + 1 < len(trace["user_messages"]))
            break

    return {
        "predicted_urls": last_recs_urls,
        "predicted_names": last_recs_names,
        "turns_taken": turns_taken,
        "schema_errors": schema_errors,
        "test_type_errors": test_type_errors,
        "ended_early": ended_early,
    }


# ── Recall@10 ──────────────────────────────────────────────────────────────────

def recall_at_k(predicted_urls: list[str], expected_urls: list[str], k: int = 10) -> float:
    if not expected_urls:
        return 1.0
    predicted_set = set(predicted_urls[:k])
    expected_set = set(expected_urls)
    return len(predicted_set & expected_set) / len(expected_set)


# ── URL hallucination check ────────────────────────────────────────────────────

def check_url_hallucination(base_url: str, catalog_urls: set) -> list[str]:
    """Fire a quick test request and verify all returned URLs are in catalog."""
    errors = []
    try:
        resp = requests.post(
            f"{base_url}/chat",
            json={"messages": [{"role": "user", "content": "I need assessments for a mid-level Java developer with Spring and SQL experience"}]},
            timeout=35,
        )
        resp.raise_for_status()
        data = resp.json()
        for rec in data.get("recommendations") or []:
            url = rec.get("url", "")
            if url and url not in catalog_urls:
                errors.append(f"Hallucinated URL: {url}")
    except Exception as e:
        errors.append(f"Request failed: {e}")
    return errors


# ── Main evaluation ────────────────────────────────────────────────────────────

def run_evaluation(base_url: str, traces_dir: str, dataset_path: Optional[str] = None) -> dict:
    results = {}
    total_recall = 0.0
    count = 0
    all_schema_errors = []
    all_test_type_errors = []

    # Load catalog for URL validation
    catalog_urls = set()
    if dataset_path and os.path.exists(dataset_path):
        catalog_urls = load_catalog_urls(dataset_path)
        print(f"Loaded {len(catalog_urls)} catalog URLs for validation")

    print("\n" + "="*60)
    print("CHECK 1: GET /health")
    print("="*60)
    try:
        resp = requests.get(f"{base_url}/health", timeout=10)
        body = resp.json()
        if resp.status_code == 200 and body == {"status": "ok"}:
            print("  ✅ PASS — 200 {'status': 'ok'}")
        else:
            print(f"  ❌ FAIL — status={resp.status_code} body={body}")
    except Exception as e:
        print(f"  ❌ FAIL — {e}")

    print("\n" + "="*60)
    print("CHECK 2: URL hallucination probe")
    print("="*60)
    if catalog_urls:
        hall_errors = check_url_hallucination(base_url, catalog_urls)
        if hall_errors:
            for e in hall_errors:
                print(f"  ❌ {e}")
        else:
            print("  ✅ PASS — All URLs from probe request exist in catalog")
    else:
        print("  ⚠️  SKIP — No catalog JSON provided (use --catalog flag)")

    print("\n" + "="*60)
    print("CHECK 3-5: Schema, test_type, and Recall@10 per trace")
    print("="*60)

    for i in range(1, 11):
        trace_file = os.path.join(traces_dir, f"C{i}.md")
        if not os.path.exists(trace_file):
            print(f"\nSkipping C{i}.md (not found)")
            continue

        print(f"\n{'─'*60}")
        print(f"Trace C{i}")
        print(f"{'─'*60}")

        trace = parse_trace(trace_file)
        print(f"  User messages: {len(trace['user_messages'])}")
        print(f"  Expected recs: {[n for n in trace['expected_names']]}")

        result = replay(base_url, trace, verbose=True)

        # Recall
        recall = recall_at_k(result["predicted_urls"], trace["expected_urls"])
        total_recall += recall
        count += 1

        # URL hallucination check on actual predictions
        hall_errors = []
        if catalog_urls:
            for url in result["predicted_urls"]:
                if url and url not in catalog_urls:
                    hall_errors.append(f"Hallucinated URL: {url}")

        print(f"  Predicted:     {result['predicted_names']}")
        print(f"  Recall@10:     {recall:.2f}")
        if result["schema_errors"]:
            print(f"  Schema errors: {result['schema_errors']}")
        if result["test_type_errors"]:
            print(f"  TestType err:  {result['test_type_errors']}")
        if hall_errors:
            print(f"  URL errors:    {hall_errors}")

        all_schema_errors.extend(result["schema_errors"])
        all_test_type_errors.extend(result["test_type_errors"])

        results[f"C{i}"] = {
            "recall": recall,
            "predicted": result["predicted_names"],
            "expected": trace["expected_names"],
            "turns": result["turns_taken"],
            "schema_errors": result["schema_errors"],
            "test_type_errors": result["test_type_errors"],
            "url_errors": hall_errors,
        }

        # Longer delay between traces to let token quotas recover
        time.sleep(15)

    mean_recall = total_recall / count if count > 0 else 0.0

    print(f"\n{'='*60}")
    print("FINAL REPORT")
    print(f"{'='*60}")
    print(f"  Mean Recall@10:  {mean_recall:.3f}  ({'✅ PASS' if mean_recall >= 0.5 else '❌ FAIL — need ≥0.5'})")
    print(f"  Schema errors:   {len(all_schema_errors)}  ({'✅ PASS' if not all_schema_errors else '❌ FAIL'})")
    print(f"  TestType errors: {len(all_test_type_errors)}  ({'✅ PASS' if not all_test_type_errors else '❌ FAIL'})")
    print(f"  Traces scored:   {count}/10")

    results["aggregate"] = {
        "mean_recall_at_10": mean_recall,
        "traces_evaluated": count,
        "schema_error_count": len(all_schema_errors),
        "test_type_error_count": len(all_test_type_errors),
        "passed_recall": mean_recall >= 0.5,
    }

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SHL Assessment Recommender")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--traces", default=os.path.expanduser("~/Downloads/GenAI_SampleConversations"),
                        help="Path to conversation traces directory")
    parser.add_argument("--catalog", default="", help="Path to catalog JSON for URL validation")
    args = parser.parse_args()

    results = run_evaluation(args.url, args.traces, args.catalog or None)

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    sys.exit(0 if results["aggregate"]["passed_recall"] else 1)
