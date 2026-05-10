"""
Evaluation harness for the SHL Assessment Recommender.

Parses the 10 sample conversation traces (C1-C10), extracts expected
recommendations, replays conversations against the agent, and computes
Recall@10 plus behavior probe metrics.
"""

import json
import re
import os
import sys
import requests
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_trace_file(filepath: str) -> dict:
    """Parse a conversation trace markdown file into structured format.
    
    Extracts:
    - turns: list of (role, content) tuples
    - final_recommendations: list of assessment names from the last recommendation table
    - final_urls: list of URLs from the last recommendation table
    
    Args:
        filepath: Path to a C*.md file.
        
    Returns:
        Dict with 'turns', 'final_recommendations', 'final_urls', 'final_eoc'.
    """
    with open(filepath, "r") as f:
        content = f.read()
    
    turns = []
    final_recs = []
    final_urls = []
    final_eoc = False
    
    # Split by turns
    turn_blocks = re.split(r"###\s+Turn\s+\d+", content)
    
    for block in turn_blocks[1:]:  # Skip the header
        # Extract user message
        user_match = re.search(r"\*\*User\*\*\s*\n\s*>(.*?)(?=\n\s*\*\*Agent\*\*|\Z)", block, re.DOTALL)
        if user_match:
            user_content = user_match.group(1).strip()
            # Clean up multi-line quotes
            user_content = re.sub(r"\n\s*>\s*", " ", user_content).strip()
            turns.append({"role": "user", "content": user_content})
        
        # Extract agent response 
        agent_match = re.search(r"\*\*Agent\*\*\s*\n(.*?)(?=###\s+Turn|\Z)", block, re.DOTALL)
        if agent_match:
            agent_text = agent_match.group(1).strip()
            
            # Extract recommendations from table
            table_recs = _parse_recommendation_table(agent_text)
            
            # Check if this has recommendations
            if table_recs:
                final_recs = [r["name"] for r in table_recs]
                final_urls = [r["url"] for r in table_recs]
            
            # Check end_of_conversation
            if "end_of_conversation`: **true**" in agent_text:
                final_eoc = True
            
            # Extract agent reply text (before the table)
            reply_text = re.split(r"\|.*\|.*\|", agent_text)[0].strip()
            reply_text = re.sub(r"_.*?_", "", reply_text).strip()
            if reply_text:
                turns.append({"role": "assistant", "content": reply_text})
    
    return {
        "turns": turns,
        "final_recommendations": final_recs,
        "final_urls": final_urls,
        "final_eoc": final_eoc,
    }


def _parse_recommendation_table(text: str) -> list[dict]:
    """Extract recommendations from a markdown table.
    
    Args:
        text: Block of text potentially containing a markdown table.
        
    Returns:
        List of dicts with 'name' and 'url'.
    """
    recs = []
    
    # Find table rows (lines starting with |)
    lines = text.split("\n")
    for line in lines:
        if not line.strip().startswith("|"):
            continue
        
        # Skip header and separator rows
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if len(cells) < 7:
            continue
        if cells[0] in ("#", "---", ""):
            continue
        if "---" in cells[1]:
            continue
        
        try:
            name = cells[1].strip()
            # Extract URL from the last relevant cell
            url_match = re.search(r"<(https://www\.shl\.com/[^>]+)>", line)
            if url_match:
                url = url_match.group(1)
                recs.append({"name": name, "url": url})
        except (IndexError, ValueError):
            continue
    
    return recs


def compute_recall_at_k(predicted_urls: list[str], expected_urls: list[str], k: int = 10) -> float:
    """Compute Recall@K.
    
    Recall@K = |intersection(predicted[:k], expected)| / |expected|
    
    Args:
        predicted_urls: Agent's recommended URLs.
        expected_urls: Expected (ground truth) URLs.
        k: Maximum number of predictions to consider.
        
    Returns:
        Recall@K score between 0.0 and 1.0.
    """
    if not expected_urls:
        return 1.0  # Nothing to recall
    
    predicted_set = set(predicted_urls[:k])
    expected_set = set(expected_urls)
    
    intersection = predicted_set & expected_set
    recall = len(intersection) / len(expected_set)
    
    return recall


def replay_conversation(base_url: str, trace: dict) -> dict:
    """Replay a conversation trace against the live API.
    
    Sends user messages sequentially, collecting agent responses.
    
    Args:
        base_url: Base URL of the deployed API (e.g., http://localhost:8000).
        trace: Parsed trace dict from parse_trace_file.
        
    Returns:
        Dict with 'predicted_urls', 'predicted_names', 'responses', 'error'.
    """
    messages = []
    responses = []
    predicted_urls = []
    predicted_names = []
    
    for turn in trace["turns"]:
        if turn["role"] == "user":
            messages.append(turn)
            
            # Call the API
            try:
                resp = requests.post(
                    f"{base_url}/chat",
                    json={"messages": messages},
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
                
                responses.append(data)
                
                # Track recommendations
                recs = data.get("recommendations", [])
                if recs:
                    predicted_urls = [r["url"] for r in recs]
                    predicted_names = [r["name"] for r in recs]
                
                # Add assistant response to history
                messages.append({
                    "role": "assistant",
                    "content": data.get("reply", ""),
                })
                
                # Check if conversation ended
                if data.get("end_of_conversation", False):
                    break
                    
            except Exception as e:
                return {
                    "predicted_urls": predicted_urls,
                    "predicted_names": predicted_names,
                    "responses": responses,
                    "error": str(e),
                }
    
    return {
        "predicted_urls": predicted_urls,
        "predicted_names": predicted_names,
        "responses": responses,
        "error": None,
    }


def run_evaluation(base_url: str, traces_dir: str) -> dict:
    """Run full evaluation against all conversation traces.
    
    Args:
        base_url: Base URL of the deployed API.
        traces_dir: Path to directory containing C1.md - C10.md.
        
    Returns:
        Dict with per-trace results and aggregate metrics.
    """
    results = {}
    total_recall = 0
    count = 0
    
    for i in range(1, 11):
        trace_file = os.path.join(traces_dir, f"C{i}.md")
        if not os.path.exists(trace_file):
            print(f"Skipping C{i}.md (not found)")
            continue
        
        print(f"\n{'='*60}")
        print(f"Evaluating C{i}...")
        print(f"{'='*60}")
        
        # Parse trace
        trace = parse_trace_file(trace_file)
        print(f"  User turns: {sum(1 for t in trace['turns'] if t['role'] == 'user')}")
        print(f"  Expected recs: {len(trace['final_recommendations'])}")
        for name in trace["final_recommendations"]:
            print(f"    - {name}")
        
        # Replay conversation
        result = replay_conversation(base_url, trace)
        
        if result["error"]:
            print(f"  ERROR: {result['error']}")
            results[f"C{i}"] = {"error": result["error"], "recall": 0.0}
            continue
        
        # Compute Recall@10
        recall = compute_recall_at_k(
            result["predicted_urls"],
            trace["final_urls"],
            k=10,
        )
        
        print(f"  Predicted recs: {len(result['predicted_names'])}")
        for name in result["predicted_names"]:
            print(f"    - {name}")
        print(f"  Recall@10: {recall:.2f}")
        
        results[f"C{i}"] = {
            "recall": recall,
            "predicted": result["predicted_names"],
            "expected": trace["final_recommendations"],
            "num_turns": len(result["responses"]),
        }
        
        total_recall += recall
        count += 1
    
    mean_recall = total_recall / count if count > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"AGGREGATE RESULTS")
    print(f"{'='*60}")
    print(f"Mean Recall@10: {mean_recall:.3f}")
    print(f"Traces evaluated: {count}")
    
    results["aggregate"] = {
        "mean_recall_at_10": mean_recall,
        "traces_evaluated": count,
    }
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate SHL Assessment Recommender")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--traces", default=os.path.expanduser("~/Downloads/GenAI_SampleConversations"),
                       help="Path to conversation traces directory")
    args = parser.parse_args()
    
    results = run_evaluation(args.url, args.traces)
    
    # Save results
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
